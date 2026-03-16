"""
Transcript acquisition helpers for video sources.

This module handles the "acquire before generate" stage:
- inspect source-owned files for packaged transcript data
- normalize subtitles/transcripts into canonical segment objects
- report whether local ASR is still needed
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .video_models import TranscriptDocument, TranscriptSegment, VideoRecord
from .youtube_transcript import fetch_youtube_transcript, parse_youtube_video_id


@dataclass
class TranscriptAcquisitionResult:
    """Result of trying to acquire transcript data before ASR."""

    success: bool
    video_id: str
    source_kind: str = ""
    transcript: Optional[TranscriptDocument] = None
    requires_asr: bool = True
    error: str = ""
    warnings: List[str] = field(default_factory=list)
    source_path: str = ""


def is_live_transcript_fetch_allowed() -> bool:
    """Return whether online transcript retrieval is enabled by config."""
    try:
        from admin.local_config import get_local_config

        return get_local_config().is_live_transcript_fetch_enabled()
    except Exception:
        return False


def normalize_transcript_segments(
    items: Iterable[Dict[str, object]],
    *,
    language: str = "",
    source_kind: str = "",
    segment_prefix: str = "seg",
) -> List[TranscriptSegment]:
    """Convert loose segment dictionaries into canonical transcript segments."""
    segments: List[TranscriptSegment] = []

    for index, item in enumerate(items):
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        start_sec = _coerce_seconds(item.get("start_sec", item.get("start", item.get("offset", 0.0))))
        end_sec = _coerce_seconds(item.get("end_sec", item.get("end", start_sec)))
        if end_sec < start_sec:
            end_sec = start_sec

        segments.append(
            TranscriptSegment(
                segment_id=f"{segment_prefix}_{index + 1}",
                start_sec=round(start_sec, 3),
                end_sec=round(end_sec, 3),
                text=text,
                language=language,
                source_kind=source_kind,
            )
        )

    return segments


def acquire_best_transcript(
    source_path: str | Path,
    video: VideoRecord,
    *,
    allow_live_fetch: Optional[bool] = None,
) -> TranscriptAcquisitionResult:
    """
    Try source-owned transcript assets first.

    Online retrieval is intentionally not implemented here yet; this module
    establishes the acquisition boundary and policy gate before live fetchers
    are added.
    """
    source_root = Path(source_path)

    packaged = load_packaged_transcript(source_root, video)
    if packaged:
        packaged.source_url = video.source_url
        return TranscriptAcquisitionResult(
            success=True,
            video_id=video.video_id,
            source_kind=packaged.source_kind,
            transcript=packaged,
            requires_asr=False,
            source_path=str(source_root),
        )

    warnings: List[str] = []
    live_fetch_allowed = is_live_transcript_fetch_allowed() if allow_live_fetch is None else allow_live_fetch
    if live_fetch_allowed and video.source_url:
        youtube_video_id = parse_youtube_video_id(video.source_url)
        if youtube_video_id:
            live_result = fetch_youtube_transcript(youtube_video_id, language=video.language or "en")
            if live_result.success and live_result.transcript:
                live_result.transcript.source_url = video.source_url
                return TranscriptAcquisitionResult(
                    success=True,
                    video_id=video.video_id,
                    source_kind=live_result.transcript.source_kind,
                    transcript=live_result.transcript,
                    requires_asr=False,
                    warnings=["Live transcript fetched from YouTube"] if not live_result.from_cache else ["Live transcript served from cache"],
                    source_path=str(source_root),
                )
            if live_result.error:
                warnings.append(f"Live transcript fetch failed: {live_result.error_code or live_result.error}")
        else:
            warnings.append("Live transcript fetch enabled, but source URL is not a recognized YouTube video URL")

    return TranscriptAcquisitionResult(
        success=False,
        video_id=video.video_id,
        requires_asr=True,
        error="No packaged transcript or subtitle asset found",
        warnings=warnings,
        source_path=str(source_root),
    )


def load_packaged_transcript(source_root: Path, video: VideoRecord) -> Optional[TranscriptDocument]:
    """Look for source-owned transcript assets for a video."""
    for candidate in _candidate_transcript_paths(source_root, video.video_id):
        if not candidate.exists() or not candidate.is_file():
            continue

        suffix = candidate.suffix.lower()
        try:
            if suffix == ".json":
                transcript = _load_json_transcript(candidate, video.video_id)
            elif suffix == ".srt":
                transcript = _load_srt_transcript(candidate, video.video_id)
            elif suffix == ".vtt":
                transcript = _load_vtt_transcript(candidate, video.video_id)
            elif suffix == ".txt":
                transcript = _load_plain_text_transcript(candidate, video.video_id)
            else:
                transcript = None
        except Exception:
            transcript = None

        if transcript:
            transcript.metadata["source_file"] = str(candidate)
            return transcript

    return None


def _candidate_transcript_paths(source_root: Path, video_id: str) -> List[Path]:
    bases = [
        source_root / "raw_data",
        source_root / "transcripts",
        source_root / "subtitles",
        source_root / "videos",
        source_root,
    ]
    patterns = [
        f"{video_id}.json",
        f"{video_id}.srt",
        f"{video_id}.vtt",
        f"{video_id}.txt",
        f"{video_id}.transcript.json",
        f"{video_id}.captions.json",
    ]
    return [base / pattern for base in bases for pattern in patterns]


def _load_json_transcript(path: Path, video_id: str) -> Optional[TranscriptDocument]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        if "segments" in data and isinstance(data["segments"], list):
            language = str(data.get("language", ""))
            source_kind = str(data.get("source_kind", "packaged_caption"))
            segments = normalize_transcript_segments(
                data["segments"],
                language=language,
                source_kind=source_kind,
                segment_prefix=video_id,
            )
            full_text = str(data.get("full_text", "")) or _join_segments(segments)
            return TranscriptDocument(
                video_id=video_id,
                language=language,
                source_kind=source_kind,
                full_text=full_text,
                segments=segments,
                original_language=language,
                metadata={"format": "json"},
            )

        if "text" in data:
            text = str(data.get("text", "")).strip()
            if text:
                segments = [
                    TranscriptSegment(
                        segment_id=f"{video_id}_1",
                        start_sec=0.0,
                        end_sec=0.0,
                        text=text,
                        language=str(data.get("language", "")),
                        source_kind=str(data.get("source_kind", "imported_text")),
                    )
                ]
                return TranscriptDocument(
                    video_id=video_id,
                    language=str(data.get("language", "")),
                    source_kind=str(data.get("source_kind", "imported_text")),
                    full_text=text,
                    segments=segments,
                    original_language=str(data.get("language", "")),
                    metadata={"format": "json"},
                )

    return None


def _load_plain_text_transcript(path: Path, video_id: str) -> Optional[TranscriptDocument]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None

    segment = TranscriptSegment(
        segment_id=f"{video_id}_1",
        start_sec=0.0,
        end_sec=0.0,
        text=text,
        source_kind="imported_text",
    )
    return TranscriptDocument(
        video_id=video_id,
        language="",
        source_kind="imported_text",
        full_text=text,
        segments=[segment],
        metadata={"format": "txt"},
    )


def _load_srt_transcript(path: Path, video_id: str) -> Optional[TranscriptDocument]:
    content = path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"\r?\n\r?\n+", content.strip())
    items: List[Dict[str, object]] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        time_line = lines[1] if "-->" in lines[1] else lines[0]
        if "-->" not in time_line:
            continue
        start_raw, end_raw = [part.strip() for part in time_line.split("-->", 1)]
        text_lines = lines[2:] if time_line == lines[1] else lines[1:]
        text = " ".join(text_lines).strip()
        items.append(
            {
                "start_sec": _parse_timestamp(start_raw),
                "end_sec": _parse_timestamp(end_raw),
                "text": text,
            }
        )

    segments = normalize_transcript_segments(items, source_kind="packaged_caption", segment_prefix=video_id)
    if not segments:
        return None
    return TranscriptDocument(
        video_id=video_id,
        language="",
        source_kind="packaged_caption",
        full_text=_join_segments(segments),
        segments=segments,
        metadata={"format": "srt"},
    )


def _load_vtt_transcript(path: Path, video_id: str) -> Optional[TranscriptDocument]:
    content = path.read_text(encoding="utf-8-sig")
    lines = [line.rstrip() for line in content.splitlines()]
    items: List[Dict[str, object]] = []
    current_time = None
    current_text: List[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line == "WEBVTT":
            if current_time and current_text:
                items.append(
                    {
                        "start_sec": current_time[0],
                        "end_sec": current_time[1],
                        "text": " ".join(current_text).strip(),
                    }
                )
                current_time = None
                current_text = []
            continue

        if "-->" in line:
            if current_time and current_text:
                items.append(
                    {
                        "start_sec": current_time[0],
                        "end_sec": current_time[1],
                        "text": " ".join(current_text).strip(),
                    }
                )
                current_text = []

            start_raw, end_raw = [part.strip().split(" ", 1)[0] for part in line.split("-->", 1)]
            current_time = (_parse_timestamp(start_raw), _parse_timestamp(end_raw))
            continue

        if current_time:
            current_text.append(line)

    if current_time and current_text:
        items.append(
            {
                "start_sec": current_time[0],
                "end_sec": current_time[1],
                "text": " ".join(current_text).strip(),
            }
        )

    segments = normalize_transcript_segments(items, source_kind="packaged_caption", segment_prefix=video_id)
    if not segments:
        return None
    return TranscriptDocument(
        video_id=video_id,
        language="",
        source_kind="packaged_caption",
        full_text=_join_segments(segments),
        segments=segments,
        metadata={"format": "vtt"},
    )


def _join_segments(segments: List[TranscriptSegment]) -> str:
    return "\n\n".join(segment.text for segment in segments if segment.text.strip())


def _parse_timestamp(raw_value: str) -> float:
    cleaned = raw_value.replace(",", ".").strip()
    parts = cleaned.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = "0", parts[0], parts[1]
    else:
        return _coerce_seconds(cleaned)

    return (float(hours) * 3600.0) + (float(minutes) * 60.0) + float(seconds)


def _coerce_seconds(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return 0.0
