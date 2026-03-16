# Video Processing Implementation Plan

This is the first implementation draft for expanding Disaster Clippy's video pipeline from offline ZIM transcription into a broader transcript acquisition, translation, and indexing system.

It is intentionally practical:
- what we already have
- what we can reuse
- where the new pieces should integrate
- what order to build in
- what decisions still need to be finalized

## Goal

Build a video-processing pipeline that:
- preserves full offline transcription capability
- acquires existing transcripts/captions before generating new ones
- supports archived/offline and still-live video sources
- reuses the existing language-pack system at the transcript layer
- stores transcript, translation, and index layers explicitly

## Decisions Locked In

These decisions are now the default implementation direction:

- all generated video-processing artifacts stay inside the source folder
- each source gets a `raw_data/` subfolder for intermediate and derived artifacts
- the original-language transcript is the canonical transcript layer
- transcripts are normalized to English before chunking and indexing
- chunking happens after translation, not before
- original transcript timing is the canonical timing layer
- translated and indexed layers may vary by a few seconds as long as they preserve lineage back to original segments
- the first implementation should support all three source families together:
  - video-heavy ZIMs
  - still-live YouTube-backed sources
  - direct blending of both through one shared pipeline

## Current Assets We Can Reuse

### 1. Offline video tooling

Primary file:
- `offline_tools/video_analysis.py`

Reusable pieces:
- `VideoZIMReader`
- `transcribe_with_timestamps()`
- `group_segments_by_duration()`
- `identify_topics_with_ollama()`

Fit:
- strong base for archive-only and offline-first processing
- already aligned with ZIM-style video bundles

Limitations:
- no transcript acquisition layer yet
- no subtitle parsing yet
- no transcript provenance model yet
- chunking is time-window based, not semantic-aware

### 2. Translation infrastructure

Primary files:
- `offline_tools/translation.py`
- `offline_tools/language_registry.py`
- `offline_tools/source_localizer.py`

Reusable pieces:
- MarianMT loading and language-pack detection
- translation caching pattern under `BACKUP_PATH/translations`
- batch translation flow
- localization/checkpointing patterns from `source_localizer.py`

Fit:
- very good base for transcript translation
- translation already works on text content, which matches the planned video model

Limitations:
- cache naming is article-oriented today
- translation service is mostly framed around HTML/article translation
- chat/query translation is still partial in docs and architecture

### 3. Background jobs and resumability

Primary files:
- `admin/job_manager.py`
- `admin/routes/source_tools.py`

Reusable pieces:
- background job submission
- conflict prevention by source
- checkpoint persistence in `BACKUP_PATH/_jobs`
- progress callbacks
- partial work file patterns

Fit:
- strong match for long-running video workflows
- especially useful for transcription, chunking, translation, and enrichment stages

Limitations:
- no dedicated video job types yet
- most route helpers assume document/source workflows, not media-oriented ones

### 4. Source management and validation

Primary files:
- `offline_tools/source_manager.py`
- `offline_tools/validation.py`

Reusable pieces:
- source-path conventions
- backup scanning
- metadata/index lifecycle concepts
- validation gates and actionable issues

Fit:
- useful for storing processed video outputs as first-class source artifacts

Limitations:
- current `_detect_source_type()` only recognizes `html` and `pdf`
- generic index flow does not yet have a `video` or `video_archive` path
- ZIM handling still centers on HTML extraction/import rather than transcript-native indexing

Direction:
- keep source handling hybrid rather than splitting video into a separate top-level source category
- the long-term goal is to point the system at a website, archive, or source folder and let it automatically detect and process mixed content types
- this should eventually cover HTML, PDFs, videos, subtitles/transcripts, and future content types through one preparation flow

### 5. Existing translation UI integration

Primary files:
- `admin/zim_server.py`
- `admin/local_config.py`

Reusable pieces:
- active-language configuration
- translation enable/disable flags
- translated-content badges and display cues

Fit:
- useful when later exposing translated transcript playback/browsing

## External Ideas Worth Adopting

From `sluice-main`, the main ideas worth bringing in are conceptual rather than 1:1 code reuse:
- transcript acquisition before ASR
- multi-strategy online transcript retrieval
- multiple fallback clients for live YouTube transcript access
- normalization into timestamped segments before chunking
- short-lived caching of transcript fetch results, including failures
- explicit transcript failure categories

We do not need the full Sluice app architecture.

## Recommended Architecture

Build the pipeline as four clear layers:

1. Transcript acquisition
- packaged subtitles/transcripts
- live online transcript retrieval
- imported transcript text

2. Local transcription fallback
- Faster-Whisper from media bytes when acquisition fails

3. Transcript normalization and storage
- canonical segment schema
- provenance fields
- optional translated layers

4. Chunking, enrichment, and indexing
- chunk generation
- topic extraction
- vector generation
- metadata/index output

Canonical order:

1. Acquire transcript in original language
2. Preserve full raw transcript with timing
3. Translate to English when needed
4. Chunk the normalized English transcript
5. Enrich and index the English chunks

This avoids damaging sentence structure by chunking before translation.

## Recommended Integration Strategy

### Phase 1: Add transcript acquisition without changing the whole source system

Add a new standalone module first:
- `offline_tools/transcript_acquisition.py`

Responsibilities:
- inspect local assets for transcript data
- detect YouTube/live identifiers
- attempt live transcript retrieval when allowed
- return normalized segment objects

Why start here:
- lowest-risk improvement
- keeps current `video_analysis.py` useful
- avoids overcommitting to a new source type too early

### Phase 2: Extend `video_analysis.py` around a shared segment model

Refactor `offline_tools/video_analysis.py` to:
- accept acquired transcript segments
- run local ASR only as fallback
- annotate transcript provenance
- output consistent chunk objects

Likely additions:
- `normalize_transcript_segments(...)`
- `load_packaged_subtitles(...)`
- `process_video_record(...)`

### Phase 3: Add video-aware caching

Extend the translation/transcript cache approach with video-specific caches under `BACKUP_PATH`.

Suggested new cache families:
- `transcripts/` for acquisition results
- `translations/<lang>/videos/` for translated transcript layers

Why:
- avoid repeated online fetch attempts
- avoid retranslating transcript chunks
- make retries and QA cheap

Inside each source folder, prefer source-owned artifacts first.

Recommended stage files under each source:
- `raw_data/transcript_original.json`
- `raw_data/transcript_english.json`
- `raw_data/chunks_english.json`
- `raw_data/topics_english.json`

For English-language videos, still generate `transcript_english.json` as a normalized downstream artifact so later stages do not need special-case branching.

Recommended contents:
- `raw_data/transcript_original.json`
  - canonical original-language transcript record
  - segment-level timing data
  - full transcript text assembled from the original segments
  - transcript provenance metadata
- `raw_data/transcript_english.json`
  - normalized English transcript record
  - translated text when source language is non-English
  - copied/normalized text when source language is already English
  - links back to original segments or segment ranges
- `raw_data/chunks_english.json`
  - post-translation chunk records used for embedding/indexing
  - approximate timing windows derived from source segments
  - source/provenance references
- `raw_data/topics_english.json`
  - optional enrichment outputs such as topic labels and keywords
  - references back to chunk IDs

### Phase 4: Introduce a dedicated video job path

Add new job helpers through `admin/routes/source_tools.py` and `admin/job_manager.py`.

Recommended first job types:
- `inspect_video_source`
- `acquire_transcripts`
- `transcribe_video`
- `translate_video_transcripts`
- `index_video_transcripts`
- `enrich_video_topics`

Recommended first combined flow:
- inspect -> acquire transcript -> transcribe fallback -> chunk -> optional translate -> enrich -> index

### Phase 5: Decide whether to formalize a new source type

Do not force this first.

Two paths are possible:

Option A:
- keep video processing as a specialized pipeline attached to existing source folders

Option B:
- add a true `video` or `video_archive` source type to `offline_tools/source_manager.py`

Current recommendation:
- start with Option A
- promote to Option B after the transcript pipeline stabilizes

Reason:
- `source_manager.py` currently assumes mostly HTML/PDF indexing
- a premature source-type expansion will create more churn than value

Updated direction:
- prefer a hybrid "prepare everything" pipeline over rigid source-type branching
- sources may contain mixed content such as HTML, PDFs, videos, and subtitles
- the preparation flow should scan the source, detect what is present, and run the correct tools automatically
- explicit sub-pipelines can still exist internally, but the user-facing model should remain unified

## Proposed New Modules

### `offline_tools/transcript_acquisition.py`

Purpose:
- choose the best available transcript source before ASR

Suggested functions:
- `detect_live_video_identity(...)`
- `load_packaged_transcript(...)`
- `load_packaged_subtitles(...)`
- `fetch_online_transcript(...)`
- `acquire_best_transcript(...)`

### `offline_tools/youtube_transcript.py`

Purpose:
- live YouTube-specific transcript retrieval

Suggested responsibilities:
- parse URLs/video IDs
- try multiple fetch strategies
- normalize standard and ASR-like transcript formats
- return categorized failures

Policy note:
- online transcript retrieval should be optional and controlled by connectivity plus explicit app/source policy
- the pipeline must still succeed without any online fetch capability

### `offline_tools/video_models.py`

Recommended early helper module:
- dataclasses for metadata records, transcript segments, translated layers, chunk records

This should be created early so the shared schema is centralized before multiple modules start defining overlapping structures.

## Reuse Mapping by Existing File

### `offline_tools/video_analysis.py`

Reuse:
- offline ASR
- ZIM video metadata extraction
- topic enrichment

Integrate by:
- calling transcript acquisition before `transcribe_with_timestamps()`
- replacing direct ASR-only assumptions with acquired-or-generated transcript handling

### `offline_tools/translation.py`

Reuse:
- model loading
- language-pack detection
- batch translation
- cache path conventions

Integrate by:
- adding transcript/chunk translation methods
- adding video-specific cache helpers
- reusing batch translation against transcript blocks before chunking

### `offline_tools/source_localizer.py`

Reuse:
- checkpointing approach
- phased workflow structure
- batch processing patterns

Integrate by:
- borrowing its staged processing model for long-running video pipelines

### `admin/job_manager.py`

Reuse:
- checkpoint model
- job queue
- progress reporting

Integrate by:
- creating video-specific job wrappers
- saving partial transcript/chunk outputs during long runs

### `admin/routes/source_tools.py`

Reuse:
- existing background job route style
- translation job submission patterns

Integrate by:
- adding video processing endpoints that match the current admin job UX

### `offline_tools/source_manager.py`

Reuse:
- source folder conventions
- validation lifecycle

Integrate later by:
- improving scan/detect logic so one source can route mixed content to the right processing tools
- exposing video pipeline outputs as standard source artifacts without forcing a separate source type

## Suggested Build Order

### Step 1
- finalize transcript segment schema
- finalize chunk schema
- finalize transcript provenance values
- finalize translated-layer storage shape
- finalize source-folder `raw_data/` file layout
- create `offline_tools/video_models.py` to centralize shared dataclasses and schema helpers
- define the config key(s) and policy shape for whether live online transcript retrieval is allowed

### Step 2
- build `offline_tools/transcript_acquisition.py`
- support packaged subtitles/transcripts first
- support online transcript retrieval second
- add an explicit policy/config check for whether live online retrieval is allowed

### Step 3
- update `offline_tools/video_analysis.py`
- route to acquisition first, ASR second
- add normalized chunk output

### Step 4
- add a simple CLI/debug harness for transcript inspection and processing
- validate acquisition, translation, fallback, and timing lineage before wiring everything into admin jobs

### Step 5
- extend `offline_tools/translation.py` for transcript/chunk translation
- add video transcript cache storage
- translate before chunking
- emit normalized English transcript artifacts even for English originals

### Step 6
- add first background job endpoints for video workflows
- keep them separate from generic indexing at first, but design them so they can later plug into a unified "prepare everything" pipeline

### Step 7
- add indexing integration and metadata output
- add validation and QA hooks

### Step 8
- expand source scanning so mixed-content sources can automatically route HTML, PDF, video, and transcript assets to the right preparation stages

## First-Draft Checklist

### Data model
- [ ] Finalize canonical transcript segment schema
- [ ] Finalize chunk schema
- [ ] Finalize transcript provenance values
- [ ] Finalize translated transcript layer schema
- [ ] Finalize `raw_data/` stage file names and contents
- [x] Transcript JSON files should store both segment arrays and assembled full-text fields
- [ ] Create `offline_tools/video_models.py`

### Acquisition
- [ ] Add `offline_tools/transcript_acquisition.py`
- [ ] Add packaged subtitle/transcript detection
- [ ] Add online transcript retrieval abstraction
- [ ] Add YouTube-specific transcript retrieval module
- [ ] Add categorized acquisition errors
- [ ] Add transcript acquisition cache
- [ ] Add policy/config gate for live online transcript retrieval

### Offline transcription
- [ ] Refactor `video_analysis.py` to use acquisition-first routing
- [ ] Keep Faster-Whisper fallback intact
- [ ] Preserve provenance on generated transcript segments

### Chunking and enrichment
- [ ] Upgrade chunking beyond pure duration windows
- [ ] Preserve start/end timestamps on all chunks
- [ ] Keep topic enrichment optional and retryable

### Translation
- [ ] Add transcript/chunk translation methods to `offline_tools/translation.py`
- [ ] Add translated transcript cache storage
- [ ] Preserve `text_original` and `text_translated`
- [ ] Track original and target language metadata
- [ ] Translate transcript before chunk generation
- [ ] Emit `transcript_english.json` for all videos, including English originals

### Policy and config
- [ ] Define config key(s) for whether live online transcript retrieval is allowed
- [ ] Decide where the setting lives and how modules read it
- [ ] Ensure acquisition code respects connectivity plus explicit policy/config

### CLI and debugging
- [ ] Add a CLI/debug entrypoint for transcript inspection
- [ ] Add a CLI path to test acquisition-first and ASR-fallback behavior
- [ ] Add a CLI path to inspect timing lineage across original transcript, English transcript, and chunks

### Jobs and admin
- [ ] Add first video-processing job endpoint
- [ ] Add checkpoint support for transcript acquisition/transcription
- [ ] Add admin visibility for transcript source and fallback usage
- [ ] Design toward a single "prepare everything" flow for mixed-content sources

### Validation and QA
- [ ] Add checks for transcript presence/source/provenance
- [ ] Add checks for translated transcript layers when present
- [ ] Add QA fixtures for online-first vs offline-fallback behavior

## Open Decisions

These should be finalized before implementation starts in earnest.

### 1. Should video become a first-class source_type, or stay part of a hybrid source-preparation flow?

Current recommendation:
- keep it hybrid
- improve mixed-content detection and routing instead of adding a separate top-level video source type first

### 2. How exact do translated/indexed timings need to be?

Current recommendation:
- original transcript segments are the canonical timed layer
- translated transcript layers can tolerate a few seconds of drift
- indexed chunks should use approximate time windows, not exact subtitle sync
- preserve references back to original segment IDs whenever practical

Timing policy:
- use precise timings where available on original transcript segments
- derive translated/indexed chunk windows from the underlying source segments
- optimize for stable lineage and "jump near this moment" behavior, not frame-accurate alignment

## Current Recommendation

Start with a narrow but high-value first implementation:
- packaged transcript/subtitle detection
- optional live YouTube transcript acquisition
- offline ASR fallback
- normalized segment schema
- translation before chunking
- chunk translation support
- source-owned `raw_data/` artifacts
- hybrid integration that can later plug into a unified "prepare everything" source workflow

English originals should still move through the same normalization pipeline, just without a translation step. That keeps all downstream processing uniform.

That gives the project the biggest architectural upgrade with the least churn.
