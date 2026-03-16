# Video Processing

This document describes the current video-processing toolkit in Disaster Clippy and the intended path forward for transcript acquisition, transcription, enrichment, and indexing.

The short version:
- keep full offline video transcription capability
- prefer acquiring existing transcript data before generating our own
- normalize all transcript sources into one internal segment format
- support both archived/offline video sources and still-live online sources

## Purpose

Video content can arrive in several forms:
- video-heavy ZIM archives
- archived snapshots of YouTube channels
- still-live YouTube channels that remain reachable online
- imported transcript text or subtitle files

Disaster Clippy should handle all of these without losing its offline-first design.

That means:
- the system must still work when air-gapped or using historical archives only
- the system should take advantage of better upstream transcript data when the internet is available

## Current Capabilities

Current reusable module:
- `offline_tools/video_analysis.py`

Current provided capabilities:
- `VideoZIMReader` for scanning `videos/*.json` entries in a ZIM and extracting video bytes
- `transcribe_with_timestamps()` for Faster-Whisper transcription from local media
- `group_segments_by_duration()` for transcript chunking
- `identify_topics_with_ollama()` for topic + keyword enrichment via local Ollama

This module is intentionally framework-agnostic and does not depend on orchestration runtime folders, batch manifests, or worker heartbeats.

## Strategic Direction

The project should move toward a layered video-processing model:

1. Acquire the best available transcript data
2. Generate transcript text locally only when acquisition fails or is impossible
3. Normalize all transcript sources into a shared internal schema
4. Chunk, enrich, and index transcripts in a source-agnostic way

This preserves full offline capability while improving quality and speed for still-live sources.

## Source Types

### 1. Offline/Archived Video Sources

Examples:
- video-heavy ZIM archives
- historical channel snapshots
- downloaded media with no live upstream source

Expected behavior:
- inspect local metadata and packaged text first
- if no transcript/subtitle data is present, run local ASR

### 2. Live-Backed Video Sources

Examples:
- a downloaded snapshot of a YouTube channel that is still public
- a local archive whose upstream video URLs still resolve

Expected behavior:
- try to fetch transcript/caption/metadata online first
- fall back to local ASR if online transcript acquisition fails

### 3. Imported Transcript Sources

Examples:
- pasted transcript text
- `.srt` or `.vtt` files
- exported subtitle JSON from another tool

Expected behavior:
- parse and normalize directly without running ASR

## Processing Model

### Acquire Before Generate

This is the core routing principle going forward.

Preferred order:
- packaged transcript or subtitle data inside the source
- online transcript/caption retrieval from the live upstream source
- local transcription from media bytes

Benefits:
- better text quality when captions already exist
- lower compute cost than transcribing everything
- faster ingestion for large video archives
- no loss of offline resilience

### Online vs Offline Routing

When connectivity is available and the source is still live:
- try online transcript acquisition first
- also pull current metadata when useful
- preserve the local media/transcription path as fallback

When connectivity is unavailable or the source is historical:
- skip online acquisition
- rely on packaged transcript data or local ASR

This gives the system the best available transcript quality without making internet access a requirement.

## Transcript Acquisition Layer

The system should grow a dedicated transcript acquisition layer in front of local transcription.

Suggested responsibilities:
- detect whether a video has live upstream identifiers such as YouTube video IDs
- inspect local source contents for `.srt`, `.vtt`, transcript JSON, or embedded caption text
- retrieve online transcripts when allowed and possible
- return a unified transcript result object regardless of source

Suggested acquisition outcomes:
- transcript acquired from packaged subtitles
- transcript acquired from online captions
- transcript acquired from imported text
- no transcript acquired, local ASR required

## Sluice-Inspired Online Capabilities

The `sluice` project suggests several useful ideas for online transcript acquisition, even if we do not port code directly:

- multi-strategy transcript retrieval instead of a single fetch path
- multiple YouTube client profiles/fallbacks
- support for more than one transcript XML format
- normalization into timestamped segments before any chunking
- short-lived result caching, including failed fetches
- clear failure categories such as disabled/private/unavailable/no transcript

Potential import path for Disaster Clippy:

1. Add a new module such as `offline_tools/youtube_transcript.py`
- parse YouTube URLs and video IDs
- attempt live transcript retrieval
- normalize caption blocks into common segment objects

2. Add optional online transcript acquisition to the video pipeline
- only enabled when connectivity and policy allow
- safe fallback to local ASR if unavailable

3. Keep external-service fallbacks optional
- if a hosted transcript provider is ever supported, treat it as optional enrichment, not a requirement

4. Cache acquisition results locally
- avoid repeatedly hammering live transcript endpoints during retries or source inspection

## Local Transcription Layer

Local transcription remains a first-class feature, not a legacy fallback.

Current runtime:
- `faster-whisper`

This path is required for:
- offline-only deployments
- dead or historical upstream sources
- videos with no captions
- archival scenarios where only media bytes remain

## Transcript Normalization

All transcript sources should normalize into the same internal schema before indexing.

Recommended segment schema:
- `start_sec`
- `end_sec`
- `text`
- `source_kind`
- `language`

Recommended `source_kind` values:
- `packaged_caption`
- `online_caption`
- `online_auto_caption`
- `local_asr`
- `imported_text`

This allows downstream logic to stay simple while preserving provenance and quality information.

## Translation Layer

Translation should happen at the transcript-text level, not at the raw video level.

Why this is the right boundary:
- all upstream sources eventually become text
- translation packs already operate on text
- translated transcript text is far smaller and cheaper to store than reprocessing media
- the same translation pipeline can handle English-to-other and other-to-English flows

Examples:
- English video -> acquire transcript -> translate to Spanish, French, Arabic, or any installed language
- Foreign-language video -> acquire transcript -> translate to English for indexing/search or to other supported languages for display

Recommended approach:
- store the original transcript text as the canonical layer
- optionally store one or more translated transcript layers
- keep provenance and language metadata for every layer

### Recommended Stored Text Layers

For each video or transcript chunk, preserve:
- original transcript text
- normalized segment/chunk timing
- translated variants when generated

Recommended fields:
- `original_language`
- `transcript_language`
- `text_original`
- `text_translated`
- `translation_target_language`
- `translation_model`
- `translation_generated_at`

This supports:
- native-language display when available
- English fallback for search and indexing
- retranslation later if language packs improve
- side-by-side debugging of original vs translated content

### Translation Timing

Preferred pipeline:
1. Acquire or generate transcript
2. Detect or confirm transcript language
3. Normalize to segment schema
4. Chunk for embedding/indexing
5. Translate chunks or transcript blocks as needed
6. Index original and/or translated text depending on search strategy

For quality, chunk-level translation will often work better than translating tiny caption fragments one by one.

The likely best practice is:
- keep canonical storage at segment level
- translate at chunk level for better context
- preserve links back to original segment timing

## Recommended Chunk Data Contract

For each transcript chunk, store:
- `source_id`
- `video_id`
- `video_title`
- `start_sec`
- `end_sec`
- `text`
- `topic`
- `keywords`
- `transcript_source`
- `language`

Useful optional fields:
- `video_url`
- `channel_name`
- `published_at`
- `quality_hint`
- `retrieved_at`
- `text_original`
- `text_translated`
- `original_language`
- `translation_target_language`

This supports:
- time-window attribution in search results
- provenance-aware ranking
- later reprocessing when a better transcript source becomes available
- multilingual retrieval and display from the same underlying transcript

## Data Model

The video pipeline should treat video content as a set of related text and metadata layers, not as one monolithic artifact.

Recommended layers:

### 1. Video Metadata

Describes the video itself and where it came from.

Suggested fields:
- `source_id`
- `video_id`
- `video_url`
- `video_title`
- `channel_name`
- `published_at`
- `duration_sec`
- `thumbnail_url`
- `source_type`
- `is_live_backed`

Purpose:
- identify the video
- preserve origin and attribution
- support later refresh or re-acquisition

### 2. Transcript Segments

Canonical transcript storage at the segment level.

Suggested fields:
- `video_id`
- `segment_id`
- `start_sec`
- `end_sec`
- `text`
- `language`
- `source_kind`
- `quality_hint`

Purpose:
- preserve source transcript truth
- keep exact timing
- support replay, validation, and re-chunking later

### 3. Translated Transcript Layers

Optional translated text derived from the canonical transcript.

Suggested fields:
- `video_id`
- `translation_target_language`
- `original_language`
- `translation_model`
- `translation_generated_at`
- `chunks` or `segments` containing:
  - `start_sec`
  - `end_sec`
  - `text_original`
  - `text_translated`

Purpose:
- support multilingual browsing and playback
- provide English fallback for non-English sources
- avoid rerunning translation repeatedly

### 4. Indexed Chunks

Search-ready chunks derived from transcript text.

Suggested fields:
- `source_id`
- `video_id`
- `chunk_id`
- `start_sec`
- `end_sec`
- `text`
- `text_original`
- `text_translated`
- `language`
- `transcript_source`
- `topic`
- `keywords`
- `embedding_model`
- `embedding_dimension`

Purpose:
- embedding and retrieval
- search result preview snippets
- topic-aware ranking
- exact time-window citation

### Recommended Relationships

- one video metadata record -> many transcript segments
- one video metadata record -> zero or more translated transcript layers
- one transcript source layer -> many indexed chunks

### Storage Notes

- video media is the large artifact
- transcript and translation layers are text-only and should remain relatively small
- preserving all text layers is usually worth the storage cost because it improves reprocessing, QA, multilingual support, and provenance tracking

## Chunking Direction

Current chunking is duration-based.

The next improvement should combine:
- time windows
- sentence or word-boundary awareness
- small overlap between chunks
- preserved `start_sec` and `end_sec`

That would improve embedding quality while keeping exact time references for playback and citation.

## Topic Enrichment

Topic extraction should remain a separate step after transcript acquisition/transcription and chunking.

Why:
- transcript capture should succeed even if enrichment fails
- retries become simpler
- indexing can proceed with fallback topic labels when needed

Current runtime:
- local Ollama via `identify_topics_with_ollama()`

## Suggested Integration Paths

1. Add a broader `video` or `video_archive` source path in `offline_tools/source_manager.py`
- detect video-heavy ZIMs via `zim_utils`
- support both archive-only and live-backed processing

2. Add transcript acquisition as a distinct pipeline stage
- inspect local subtitle/transcript assets
- optionally check live online transcript sources
- fall back to local ASR

3. Add a standalone admin job type
- input: `.zim` path or video source descriptor
- steps: inspect -> acquire transcript -> transcribe fallback -> chunk -> enrich -> write metadata and vectors

4. Add a CLI entrypoint for validation
- list videos
- inspect transcript source availability
- fetch online transcript when available
- transcribe one video locally
- output normalized chunk JSON for review

## Safety and Failure Rules

- Treat missing optional dependencies as explicit configuration errors
- If online acquisition fails, do not fail the whole pipeline when local ASR is possible
- If Ollama topic extraction fails, retain transcript with fallback topic labels
- Keep transcript acquisition, ASR, chunking, and enrichment as separate retryable stages
- Never make online acquisition mandatory for a source to be indexable

## Dependencies

Already present:
- `zimply-core`
- `requests`

Required for local transcription runtime:
- `faster-whisper`

Install when enabling transcription:

```bash
pip install faster-whisper
```

Future optional dependencies may be added for:
- online transcript acquisition
- subtitle parsing helpers
- live source metadata retrieval

## Near-Term Implementation Path

Recommended next steps:

1. Keep `offline_tools/video_analysis.py` as the base offline module
2. Add a transcript acquisition layer ahead of `transcribe_with_timestamps()`
3. Normalize all transcript inputs to one segment schema
4. Upgrade chunking to blend semantic boundaries with time-aware attribution
5. Add provenance fields to stored transcript chunks
6. Add transcript translation support using the existing language-pack system
7. Add admin/job visibility for whether transcript text came from:
- online captions
- packaged subtitles
- imported text
- local ASR

This path improves transcript quality for live-backed sources without weakening the offline mission of the project.
