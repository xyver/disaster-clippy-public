# Video Processing Implementation Plan

This plan reflects the current implementation state of the video pipeline and narrows the remaining work to the parts that are still unfinished.

## Current Baseline

The following pieces already exist:
- `offline_tools/video_models.py`
- `offline_tools/transcript_acquisition.py`
- `offline_tools/youtube_transcript.py`
- `offline_tools/video_analysis.py` acquisition-first routing
- `offline_tools/video_analysis.py` URL-first wrapper and outcome tracking
- English transcript normalization before chunking
- source-owned artifact writing under `raw_data/videos/<video_id>/`
- admin job entrypoint in `admin/routes/source_tools.py`

The current pipeline already supports:
- packaged transcript/subtitle detection
- optional live YouTube transcript acquisition
- explicit URL-first preparation for pasted links
- Faster-Whisper fallback when local media is available
- transcript normalization into shared dataclasses
- English transcript generation
- chunk generation
- optional Ollama topic enrichment

## What Changed Since The First Draft

The earlier plan treated several pieces as future work that are now implemented:
- transcript acquisition layer
- YouTube transcript module
- shared video dataclasses
- acquisition-first routing in `video_analysis.py`
- URL-first wrapper around pasted links
- translation before chunking
- normalized English transcript artifacts
- source-owned raw-data artifacts
- first admin job path for video transcript preparation

Because of that, this document is now focused on remaining gaps rather than re-planning the whole foundation.

## Implemented Architecture

Current layered flow:

1. Transcript acquisition
- packaged `.json`, `.srt`, `.vtt`, and `.txt` assets
- optional live YouTube transcript retrieval

2. Local transcription fallback
- Faster-Whisper from `asr_video_path` when no transcript was acquired

3. Transcript normalization and storage
- `TranscriptDocument`, `TranscriptSegment`, and `TranscriptChunk`
- artifact writing to `raw_data/videos/<video_id>/`

4. Translation, chunking, and enrichment
- translate or normalize to English
- chunk after English normalization
- optionally enrich with Ollama topics/keywords

Cross-cutting entrypoint:
- URL-first preparation now wraps the existing pipeline and returns explicit outcome states for pasted links

## Remaining Priorities

### 1. Strengthen transcript provenance

Still needed:
- decide whether to emit a distinct `online_auto_caption` source kind
- standardize provenance metadata across packaged, online, and ASR sources
- make sure downstream indexing surfaces provenance cleanly

### 2. Improve chunk quality

Still needed:
- replace pure duration-window chunking with sentence-aware or semantic chunking
- preserve stable lineage back to original segment IDs
- decide whether overlap windows are needed for embeddings

### 3. Add durable caching

Still needed:
- on-disk cache for transcript fetch results
- optional persistence of failed fetch states to avoid repeated retries
- alignment between transcript caching and existing translation cache patterns

Current limitation:
- YouTube transcript caching is only in-memory with a short TTL

### 4. Extend online/live handling

Still needed:
- explicit connectivity checks or policy wiring around live fetch attempts
- clearer handling of non-YouTube live-backed sources
- decide whether media download should ever be added for ASR fallback from URL-only inputs

Current limitation:
- URL-only processing can fetch captions, but it cannot download media for ASR
- URL-only processing is now observable, but not yet self-sufficient

### 5. Add CLI and QA tooling

Still needed:
- transcript inspection CLI/debug entrypoint
- simple developer workflow to test packaged, online, and ASR paths
- QA fixtures for online-first versus offline fallback behavior

### 6. Broaden system integration

Still needed:
- validation hooks for transcript artifacts
- indexing/vector integration for transcript chunks
- mixed-content source scanning in `offline_tools/source_manager.py`
- eventual convergence into a unified source-preparation flow

### 7. Add bulk video processing

Still needed:
- folder-level batch processing for local media files
- ZIM-level batch processing using `VideoZIMReader`
- batch summary outputs and retry-friendly reporting
- admin/job wrappers for long-running bulk video preparation

## Current File Reuse Map

### `offline_tools/video_models.py`

Already used for:
- `VideoRecord`
- `TranscriptSegment`
- `TranscriptDocument`
- `TranscriptChunk`

### `offline_tools/transcript_acquisition.py`

Already used for:
- packaged transcript discovery
- transcript normalization
- optional live fetch routing

### `offline_tools/youtube_transcript.py`

Already used for:
- YouTube URL parsing
- InnerTube-based transcript retrieval
- short-lived caching
- categorized transcript fetch failures

### `offline_tools/video_analysis.py`

Already used for:
- acquisition-first transcript preparation
- URL-first result tracking for pasted links
- ASR fallback
- translation to English
- chunk generation
- topic enrichment
- artifact writing

### `admin/routes/source_tools.py`

Already used for:
- background job wrapper around `prepare_video_transcripts(...)`
- background job wrapper around `prepare_video_from_url(...)`

## Updated Build Order

### Step 1
- add validation and tests around the current acquisition-first and URL-first flows
- document the artifact contract and provenance behavior more explicitly

### Step 2
- improve chunking quality beyond duration-only grouping
- keep timing lineage stable while improving text boundaries

### Step 3
- add durable transcript cache storage
- align cache layout with source-owned artifact patterns and existing translation caching

### Step 4
- add CLI/debug tooling for transcript inspection and troubleshooting

### Step 5
- wire transcript chunks into indexing and validation flows

### Step 6
- expand mixed-content source detection so video assets participate in a unified preparation workflow

### Step 7
- add folder-batch and ZIM-batch runners on top of the existing per-video pipeline
- expose both through admin jobs with progress reporting and summary artifacts

## Updated Checklist

### Already implemented
- [x] Create `offline_tools/video_models.py`
- [x] Add `offline_tools/transcript_acquisition.py`
- [x] Add packaged subtitle/transcript detection
- [x] Add YouTube-specific transcript retrieval module
- [x] Add policy/config gate for live online transcript retrieval
- [x] Refactor `video_analysis.py` to use acquisition-first routing
- [x] Add URL-first preparation wrapper with explicit outcome states
- [x] Keep Faster-Whisper fallback intact
- [x] Preserve transcript JSON with both segments and full-text fields
- [x] Translate transcript before chunk generation
- [x] Emit `transcript_english.json` for English and non-English paths
- [x] Add first video-processing admin job endpoint
- [x] Add URL-first admin job endpoint

### Still to do
- [ ] Finalize transcript provenance values and metadata conventions
- [ ] Distinguish `online_caption` vs `online_auto_caption` if needed
- [ ] Add durable transcript acquisition cache storage
- [ ] Upgrade chunking beyond pure duration windows
- [ ] Add CLI/debug entrypoint for transcript inspection
- [ ] Add validation checks for transcript artifacts
- [ ] Add QA fixtures for online-first vs offline-fallback behavior
- [ ] Integrate transcript chunks into indexing/vector flows
- [ ] Improve mixed-content source scanning and routing
- [ ] Decide whether URL-only media download should be supported for ASR fallback
- [ ] Add folder-batch video processing
- [ ] Add ZIM-batch video processing
- [ ] Add batch summary and retry/reporting artifacts

## Open Decisions

### 1. Should URL-only processing ever download media for ASR fallback?

Current recommendation:
- do not assume this yet
- keep caption retrieval optional and lightweight
- only add media download if the product really needs URL-only no-caption transcription

### 2. Should `online_auto_caption` be a real first-class provenance value?

Current recommendation:
- likely yes, if the pipeline can reliably distinguish auto-generated YouTube captions from other caption tracks

### 3. Should video become a first-class `source_type`?

Current recommendation:
- not yet
- prefer hybrid mixed-content routing before introducing a rigid new top-level source type

## Current Recommendation

The foundation is in place. The next best work is not more architectural scaffolding; it is tightening the current implementation:
- improve chunk quality
- strengthen provenance
- add persistent caching
- add validation and QA
- decide how URL-first media acquisition should plug into the current outcome model
- connect transcript outputs to indexing and broader source preparation

That keeps momentum high while avoiding another round of speculative redesign.
