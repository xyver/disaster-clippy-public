# Video Processing

This document describes the current video-processing implementation in Disaster Clippy and the near-term gaps that still remain.

The short version:
- transcript acquisition now happens before local ASR
- packaged transcripts and subtitles are already supported
- optional live YouTube transcript fetching is already implemented behind a config gate
- pasted URLs now have a URL-first preparation path with explicit outcome tracking
- local Faster-Whisper ASR remains the fallback path when media is available
- transcript normalization, English normalization, chunking, artifact writing, and optional topic enrichment are implemented

## Current Implementation

Implemented modules:
- `offline_tools/video_analysis.py`
- `offline_tools/transcript_acquisition.py`
- `offline_tools/youtube_transcript.py`
- `offline_tools/video_models.py`

Current data models in `offline_tools/video_models.py`:
- `VideoRecord`
- `TranscriptSegment`
- `TranscriptDocument`
- `TranscriptChunk`

Current reusable capabilities in `offline_tools/video_analysis.py`:
- `VideoZIMReader` for scanning `videos/*.json` entries in a ZIM and extracting video bytes
- `transcribe_with_timestamps()` for Faster-Whisper transcription from local media
- `group_segments_by_duration()` for time-window chunking
- `translate_transcript_to_english()` for transcript-level normalization into English
- `chunk_transcript_document()` for post-translation chunk generation
- `prepare_video_transcripts()` for the acquisition-first end-to-end flow
- `build_video_record_from_url()` for deriving a `VideoRecord` from a pasted URL
- `prepare_video_from_url()` for URL-first transcript preparation with explicit outcome states
- `identify_topics_with_ollama()` for optional topic and keyword enrichment
- artifact writers for transcript, chunk, and topic JSON outputs

Current reusable capabilities in `offline_tools/transcript_acquisition.py`:
- `acquire_best_transcript()` as the routing layer ahead of ASR
- packaged transcript detection from source-owned files
- normalization of loose transcript items into canonical segments
- parsing for `.json`, `.srt`, `.vtt`, and `.txt` transcript assets

Current reusable capabilities in `offline_tools/youtube_transcript.py`:
- `parse_youtube_video_id()` for common YouTube URL formats
- `fetch_youtube_transcript()` for optional live transcript retrieval
- multiple InnerTube client profiles/fallbacks
- parsing of standard caption XML and ASR-like XML
- short-lived in-memory caching of both successes and failures
- categorized fetch failures such as `disabled`, `unavailable`, `no_transcript`, `timeout`, and `fetch_failed`

## Current Processing Flow

The implemented pipeline order is:

1. Inspect the source folder for packaged transcript assets
2. If allowed, try live YouTube transcript retrieval from `video.source_url`
3. If no transcript was acquired and a local media path is provided, run Faster-Whisper ASR
4. Normalize the original transcript into a `TranscriptDocument`
5. Normalize or translate that transcript into English
6. Chunk the English transcript
7. Optionally enrich chunks with Ollama topics and keywords
8. Write source-owned artifacts into `raw_data/videos/<video_id>/`

This logic is implemented by `prepare_video_transcripts(...)`.

## URL-First Flow

There is now an explicit URL-first wrapper for pasted links:
- `prepare_video_from_url(...)`

This does not magically make all URLs work, but it does make the flow observable and predictable.

Current URL-first behavior:
1. Validate that a source URL was provided
2. Detect whether the URL is a recognized YouTube URL
3. Build a lightweight `VideoRecord` from the URL
4. Reuse the existing acquisition-first transcript pipeline
5. Return a clear outcome state and recommended next action

Current outcome values:
- `transcript_acquired_online`
- `transcript_acquired_packaged`
- `media_acquired_asr_used`
- `transcript_missing_media_required`
- `failed_unsupported_source`
- `failed_invalid_url`
- `failed_processing`

Current tracking fields include:
- `source_url`
- `video_id`
- `source_platform`
- `recognized_source`
- `transcript_fetch_allowed`
- `attempted_live_fetch`
- `asr_video_path_provided`
- `next_action`

This is the current answer to "paste a link and tell me what happened" even before automatic media download exists.

## Source Types Supported Today

### 1. Source-owned transcript assets

Supported today:
- `<source>/raw_data/<video_id>.*`
- `<source>/transcripts/<video_id>.*`
- `<source>/subtitles/<video_id>.*`
- `<source>/videos/<video_id>.*`
- `<source>/<video_id>.*`

Supported file types:
- `.json`
- `.srt`
- `.vtt`
- `.txt`

This is the most complete path today.

### 2. Live-backed YouTube sources

Supported today when both of these are true:
- `video.source_url` is a recognized YouTube URL
- `video_processing.allow_live_transcript_fetch` is enabled in local config

When enabled, the acquisition layer will try YouTube captions before falling back to ASR.

With the new URL-first wrapper, the system can now also report whether:
- the URL was recognized as YouTube
- live fetch was allowed
- live fetch was attempted
- local media is still required for ASR fallback

### 3. Local media ASR fallback

Supported today when `asr_video_path` is provided to `prepare_video_transcripts(...)`.

Important limitation:
- the pipeline does not currently download media from a YouTube URL
- if no packaged or live transcript exists, ASR still requires a local media file path
- the URL-first wrapper reports this explicitly as `transcript_missing_media_required`

## Config Gate For Live Fetch

Optional live transcript retrieval is controlled by:
- `admin/local_config.py`
- config key: `video_processing.allow_live_transcript_fetch`

Default:
- `False`

Behavior:
- if disabled, the pipeline only checks packaged transcript assets and then falls back to ASR when available
- if enabled, the pipeline may also attempt live YouTube transcript acquisition

## Current Artifact Layout

The current implementation writes outputs under:
- `raw_data/videos/<video_id>/`

Current stage artifacts:
- `transcript_original.json`
- `transcript_english.json`
- `chunks_english.json`
- `topics_english.json` when enrichment is enabled

This is more specific than the earlier design notes that described only a top-level `raw_data/` layout.

## Current Schema Direction

### Transcript segments

Current canonical segment fields:
- `segment_id`
- `start_sec`
- `end_sec`
- `text`
- `language`
- `source_kind`
- `original_segment_ids`

Current `source_kind` values in code:
- `packaged_caption`
- `online_caption`
- `local_asr`
- `imported_text`

Note:
- `online_auto_caption` is discussed in docs but is not currently emitted as a distinct value

### Transcript documents

Current transcript document fields:
- `video_id`
- `language`
- `source_kind`
- `full_text`
- `segments`
- `original_language`
- `translation_target_language`
- `translation_model`
- `translation_generated_at`
- `source_url`
- `retrieved_at`
- `metadata`

### Transcript chunks

Current chunk fields:
- `chunk_id`
- `video_id`
- `start_sec`
- `end_sec`
- `text`
- `language`
- `text_original`
- `text_translated`
- `transcript_source`
- `topic`
- `keywords`
- `original_segment_ids`

## Translation Behavior

Current implementation:
- original-language transcript is preserved as the canonical transcript layer
- a normalized English transcript is always produced for downstream processing
- if the source language is already English, the English transcript is copied/normalized rather than translated
- if the source language is non-English, `TranslationService` is used to translate segment text before chunking

Current pipeline order in code:
1. Acquire or generate transcript
2. Preserve original transcript
3. Translate or normalize to English
4. Chunk the English transcript
5. Optionally enrich topics

This means the implementation now follows translation-before-chunking.

## Topic Enrichment

Current runtime:
- local Ollama via `identify_topics_with_ollama()`

Current behavior:
- enrichment is optional
- if Ollama fails for a chunk, the pipeline keeps the chunk and falls back to a generic topic label with no keywords

## Admin Integration

There is already an admin route in `admin/routes/source_tools.py` for transcript preparation:
- `POST /prepare-video-transcript`

There is now also a URL-first admin route:
- `POST /prepare-video-from-url`

The job path currently:
- builds a `VideoRecord`
- calls `prepare_video_transcripts(...)`
- writes artifacts into the source-owned `raw_data/videos/<video_id>/` folder
- reports whether ASR was used and what transcript source was chosen

The URL-first job path currently:
- accepts a pasted `source_url`
- derives or reuses a `video_id`
- calls `prepare_video_from_url(...)`
- returns an explicit `outcome`
- returns a `next_action` when the URL alone is not enough

## What Is Still Missing

The current implementation is ahead of the original docs, but several planned pieces are still incomplete.

Not implemented yet:
- automatic media download from YouTube URLs for ASR fallback
- semantic or sentence-aware chunking beyond duration windows
- durable on-disk cache storage for transcript acquisition results
- broader source-manager integration for mixed-content auto-detection
- dedicated validation and QA coverage for video transcript artifacts
- full indexing/vector integration described in the plan
- richer transcript provenance values such as a separate `online_auto_caption`
- a dedicated CLI/debug harness for transcript inspection

## Next Steps

The most useful next implementation steps are:
- add a folder-batch runner so the system can scan a folder and process all local video files within
- add a ZIM-batch runner so the system can enumerate videos in a ZIM and process them in sequence
- add admin/job routes for both bulk modes so long runs have progress and resumability
- write batch summary artifacts so each run records which videos succeeded, failed, or still require local media

This would move the pipeline from "one video at a time" to practical collection-level processing.

## Practical Summary

What works today:
- packaged transcript acquisition
- optional live YouTube caption retrieval
- URL-first inspection and explicit pasted-link outcome tracking
- local ASR fallback from local media
- transcript normalization into shared dataclasses
- English transcript generation before chunking
- chunk artifact generation
- optional topic enrichment
- admin job submission for one video transcript-preparation flow

What does not work from URL alone yet:
- downloading an arbitrary YouTube video and then transcribing it locally when no captions are available
