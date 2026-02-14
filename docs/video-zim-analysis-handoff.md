# Video ZIM Analysis Handoff

This document describes the tooling contributed from `video_zim_batch` and how to integrate it into Disaster Clippy.

## What Was Added

Core reusable module:
- `offline_tools/video_analysis.py`

Provided capabilities:
- `VideoZIMReader` for scanning `videos/*.json` entries in a ZIM and extracting video bytes
- `transcribe_with_timestamps()` for Faster-Whisper transcription
- `group_segments_by_duration()` for transcript chunking
- `identify_topics_with_ollama()` for topic + keyword enrichment via local Ollama

This is intentionally framework-agnostic and does not depend on orchestration runtime folders, batch manifests, or worker heartbeats.

## Why This Shape

- Keeps upstream surface area small and reviewable
- Avoids coupling to a specific planner/worker runtime
- Lets maintainers choose where this fits: CLI, job chain, or admin pipeline

## Suggested Integration Paths

1. Add a `video_zim` source type in `offline_tools/source_manager.py`
- Detect video-heavy ZIMs via `zim_utils` inspection
- Route indexing to a dedicated video pipeline that emits transcript chunks as documents

2. Add a standalone admin job type
- Input: `.zim` path + optional filters (`max_videos`, `min_duration`)
- Steps: inspect -> extract/transcribe -> enrich topics -> write `_metadata.json` and vectors

3. Add CLI entrypoint for local validation
- Script that can run:
  - list videos
  - transcribe one video
  - transcribe + topic annotate to JSON output

## Recommended Data Contract

For each transcript chunk, store:
- `source_id`
- `video_id`
- `video_title`
- `start_sec`, `end_sec`
- `text`
- `topic`
- `keywords`

This allows search and result attribution to point back to exact time windows.

## Dependencies

Already present:
- `zimply-core`
- `requests`

Required for transcription runtime:
- `faster-whisper`

Install when enabling transcription:

```bash
pip install faster-whisper
```

## Safety / Failure Notes

- Treat missing optional dependencies as explicit configuration errors
- If Ollama topic extraction fails, retain transcript with fallback topic labels rather than dropping the video
- Keep transcription and LLM enrichment as separate steps so failures are easier to retry
