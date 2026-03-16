"""
Video analysis toolkit for video-heavy ZIM archives.

This module is intentionally standalone so maintainers can integrate it into
existing job pipelines without taking any orchestration-specific code.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .transcript_acquisition import acquire_best_transcript
from .translation import TranslationService
from .video_models import TranscriptChunk, TranscriptDocument, TranscriptSegment, VideoRecord


class VideoZIMReader:
    """Reader for YouTube-style ZIM archives that store video metadata + binary media."""

    def __init__(self, zim_path: str):
        self.zim_path = Path(zim_path)
        self._zim = None
        self._video_cache: Optional[List[VideoRecord]] = None

    def _open_zim(self):
        if self._zim is None:
            from zimply_core.zim_core import ZIMFile

            self._zim = ZIMFile(str(self.zim_path), "utf-8")
        return self._zim

    @staticmethod
    def parse_iso8601_duration(raw_value: Optional[str]) -> int:
        """Parse ISO-8601 video duration (PT3M42S) to total seconds."""
        if not raw_value or not isinstance(raw_value, str):
            return 0

        match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", raw_value)
        if not match:
            return 0

        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return (hours * 3600) + (minutes * 60) + seconds

    def list_videos(self) -> List[VideoRecord]:
        """Find videos by scanning `videos/*.json` metadata entries in the ZIM."""
        if self._video_cache is not None:
            return self._video_cache

        zim = self._open_zim()
        article_count = zim.header_fields.get("articleCount", 0)
        videos: List[VideoRecord] = []

        for idx in range(article_count):
            try:
                article = zim.get_article_by_id(idx)
                if article is None:
                    continue

                url = getattr(article, "url", "")
                if not (url.endswith(".json") and "videos/" in url):
                    continue

                video_id = url.replace("videos/", "").replace(".json", "")
                if "/" in video_id:
                    continue

                raw_content = article.data
                if isinstance(raw_content, bytes):
                    raw_content = raw_content.decode("utf-8")
                metadata = json.loads(raw_content)

                videos.append(
                    VideoRecord(
                        video_id=video_id,
                        title=metadata.get("title", "Unknown"),
                        description=metadata.get("description", ""),
                        duration_seconds=self.parse_iso8601_duration(
                            metadata.get("duration")
                        ),
                        video_path=metadata.get("videoPath"),
                        thumbnail_path=metadata.get("thumbnailPath"),
                        upload_date=metadata.get("uploadDate", ""),
                    )
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            except Exception:
                continue

        self._video_cache = videos
        return videos

    def get_video_content(self, video_path: str) -> Optional[bytes]:
        """Extract video bytes by exact or suffix URL match."""
        zim = self._open_zim()
        article_count = zim.header_fields.get("articleCount", 0)

        for idx in range(article_count):
            try:
                article = zim.get_article_by_id(idx)
                if article is None:
                    continue

                url = getattr(article, "url", "")
                if url == video_path or url.endswith(video_path):
                    data = article.data
                    if isinstance(data, bytes):
                        return data
            except Exception:
                continue

        return None

    def close(self) -> None:
        if self._zim:
            try:
                self._zim.close()
            except Exception:
                pass
            self._zim = None


def transcribe_with_timestamps(
    video_path: str,
    model_name: str = "small.en",
    device: str = "cuda",
    device_index: int = 0,
    compute_type: str = "int8",
    beam_size: int = 5,
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Transcribe media with Faster-Whisper and return simple segment dictionaries.

    Returns (segments, duration_seconds).
    """
    from faster_whisper import WhisperModel

    model = WhisperModel(
        model_name,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
    )
    segments, info = model.transcribe(video_path, beam_size=beam_size)

    output_segments: List[Dict[str, Any]] = []
    for segment in segments:
        output_segments.append(
            {
                "start_sec": round(segment.start, 1),
                "end_sec": round(segment.end, 1),
                "text": segment.text.strip(),
            }
        )

    return output_segments, info.duration


def normalize_segment_dicts(
    segments: Sequence[Dict[str, Any]],
    *,
    language: str = "",
    source_kind: str = "",
    segment_prefix: str = "seg",
) -> List[TranscriptSegment]:
    """Convert loose segment dictionaries into canonical transcript segments."""
    normalized: List[TranscriptSegment] = []
    for index, segment in enumerate(segments):
        text = str(segment.get("text", "")).strip()
        if not text:
            continue

        start = float(segment.get("start_sec", 0.0))
        end = float(segment.get("end_sec", start))
        if end < start:
            end = start

        normalized.append(
            TranscriptSegment(
                segment_id=f"{segment_prefix}_{index + 1}",
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                text=text,
                language=language,
                source_kind=source_kind,
            )
        )
    return normalized


def group_segments_by_duration(
    segments: Sequence[Dict[str, Any]],
    chunk_duration_seconds: float = 60.0,
) -> List[Dict[str, Any]]:
    """Merge short ASR segments into time-window chunks for later topic labeling."""
    chunks: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {"start_sec": 0.0, "end_sec": 0.0, "segments": [], "text": ""}

    for segment in segments:
        start = float(segment.get("start_sec", 0.0))
        end = float(segment.get("end_sec", start))
        text = str(segment.get("text", "")).strip()

        if current["segments"] and (start - float(current["start_sec"])) > chunk_duration_seconds:
            current["end_sec"] = float(current["segments"][-1]["end_sec"])
            current["text"] = " ".join(s["text"] for s in current["segments"]).strip()
            chunks.append(current)
            current = {"start_sec": start, "end_sec": end, "segments": [], "text": ""}

        if not current["segments"]:
            current["start_sec"] = start

        current["segments"].append({"start_sec": start, "end_sec": end, "text": text})

    if current["segments"]:
        current["end_sec"] = float(current["segments"][-1]["end_sec"])
        current["text"] = " ".join(s["text"] for s in current["segments"]).strip()
        chunks.append(current)

    return chunks


def ensure_raw_data_dir(source_path: str | Path) -> Path:
    """Ensure raw_data/ exists inside a source folder."""
    raw_dir = Path(source_path) / "raw_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def ensure_video_raw_data_dir(source_path: str | Path, video_id: str) -> Path:
    """Ensure raw_data/videos/<video_id>/ exists inside a source folder."""
    video_dir = ensure_raw_data_dir(source_path) / "videos" / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir


def write_transcript_artifact(
    source_path: str | Path,
    filename: str,
    transcript: TranscriptDocument,
) -> Path:
    """Write a transcript artifact into source-owned raw_data/."""
    raw_dir = ensure_video_raw_data_dir(source_path, transcript.video_id)
    artifact_path = raw_dir / filename
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(transcript.to_dict(), f, indent=2, ensure_ascii=False)
    return artifact_path


def write_chunk_artifact(
    source_path: str | Path,
    filename: str,
    chunks: Sequence[TranscriptChunk],
) -> Path:
    """Write chunk artifacts into source-owned raw_data/."""
    if not chunks:
        raise ValueError("Cannot write chunk artifact for empty chunk list")
    raw_dir = ensure_video_raw_data_dir(source_path, chunks[0].video_id)
    artifact_path = raw_dir / filename
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump([chunk.to_dict() for chunk in chunks], f, indent=2, ensure_ascii=False)
    return artifact_path


def write_topics_artifact(
    source_path: str | Path,
    filename: str,
    chunks_with_topics: Sequence[Dict[str, Any]] | Sequence[TranscriptChunk],
    *,
    video_id: str,
) -> Path:
    """Write topic-enriched chunk artifacts into source-owned raw_data/."""
    raw_dir = ensure_video_raw_data_dir(source_path, video_id)
    artifact_path = raw_dir / filename
    serializable = []
    for chunk in chunks_with_topics:
        if hasattr(chunk, "to_dict"):
            serializable.append(chunk.to_dict())
        else:
            serializable.append(dict(chunk))
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    return artifact_path


def translate_transcript_to_english(
    transcript: TranscriptDocument,
    *,
    source_language: str = "",
) -> TranscriptDocument:
    """Normalize a transcript into the English downstream artifact."""
    effective_language = source_language or transcript.language or transcript.original_language or "en"

    if effective_language == "en":
        return TranscriptDocument(
            video_id=transcript.video_id,
            language="en",
            source_kind=transcript.source_kind,
            full_text=transcript.full_text,
            segments=[
                TranscriptSegment(
                    segment_id=segment.segment_id,
                    start_sec=segment.start_sec,
                    end_sec=segment.end_sec,
                    text=segment.text,
                    language="en",
                    source_kind=segment.source_kind,
                    original_segment_ids=segment.original_segment_ids or [segment.segment_id],
                )
                for segment in transcript.segments
            ],
            original_language=effective_language,
            translation_target_language="en",
            source_url=transcript.source_url,
            retrieved_at=transcript.retrieved_at,
            metadata={**transcript.metadata, "normalized_without_translation": True},
        )

    translator = TranslationService(effective_language)
    translated_texts = translator.translate_batch_to_english([segment.text for segment in transcript.segments])
    translated_segments: List[TranscriptSegment] = []
    for segment, translated_text in zip(transcript.segments, translated_texts):
        translated_segments.append(
            TranscriptSegment(
                segment_id=segment.segment_id,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                text=translated_text,
                language="en",
                source_kind=segment.source_kind,
                original_segment_ids=segment.original_segment_ids or [segment.segment_id],
            )
        )

    return TranscriptDocument(
        video_id=transcript.video_id,
        language="en",
        source_kind=transcript.source_kind,
        full_text="\n\n".join(seg.text for seg in translated_segments if seg.text.strip()),
        segments=translated_segments,
        original_language=effective_language,
        translation_target_language="en",
        translation_model=f"MarianMT:{effective_language}->en" if effective_language != "en" else "",
        translation_generated_at=datetime.utcnow().isoformat() + "Z",
        source_url=transcript.source_url,
        retrieved_at=transcript.retrieved_at,
        metadata={**transcript.metadata},
    )


def chunk_transcript_document(
    transcript: TranscriptDocument,
    chunk_duration_seconds: float = 60.0,
) -> List[TranscriptChunk]:
    """Create English chunk artifacts from a normalized transcript document."""
    loose_chunks = group_segments_by_duration(
        [
            {"start_sec": seg.start_sec, "end_sec": seg.end_sec, "text": seg.text}
            for seg in transcript.segments
        ],
        chunk_duration_seconds=chunk_duration_seconds,
    )

    chunks: List[TranscriptChunk] = []
    for index, chunk in enumerate(loose_chunks):
        chunk_start = float(chunk["start_sec"])
        chunk_end = float(chunk["end_sec"])
        segment_ids = [
            segment.segment_id
            for segment in transcript.segments
            if segment.end_sec >= chunk_start and segment.start_sec <= chunk_end
        ]
        chunks.append(
            TranscriptChunk(
                chunk_id=f"{transcript.video_id}_chunk_{index + 1}",
                video_id=transcript.video_id,
                start_sec=chunk_start,
                end_sec=chunk_end,
                text=str(chunk.get("text", "")).strip(),
                language=transcript.language or "en",
                text_original=" ".join(
                    segment.text for segment in transcript.segments
                    if segment.segment_id in segment_ids
                ).strip(),
                text_translated=str(chunk.get("text", "")).strip(),
                transcript_source=transcript.source_kind,
                original_segment_ids=segment_ids,
            )
        )
    return chunks


def prepare_video_transcripts(
    source_path: str | Path,
    video: VideoRecord,
    *,
    asr_video_path: Optional[str] = None,
    asr_model_name: str = "small",
    chunk_duration_seconds: float = 60.0,
    write_artifacts: bool = True,
    allow_live_fetch: Optional[bool] = None,
    enrich_topics: bool = False,
    ollama_url: str = "http://localhost:11434",
    topic_model: str = "qwen2.5:7b",
) -> Dict[str, Any]:
    """
    Acquisition-first transcript preparation for one video.

    Current behavior:
    - try packaged transcript assets first
    - fall back to ASR if an explicit media path is provided
    - normalize an English transcript artifact
    - chunk after translation/normalization
    """
    acquisition = acquire_best_transcript(source_path, video, allow_live_fetch=allow_live_fetch)

    transcript_original: Optional[TranscriptDocument] = acquisition.transcript
    used_asr = False
    if transcript_original is None and asr_video_path:
        segment_dicts, duration_seconds = transcribe_with_timestamps(asr_video_path, model_name=asr_model_name)
        segments = normalize_segment_dicts(
            segment_dicts,
            language=video.language or "",
            source_kind="local_asr",
            segment_prefix=video.video_id,
        )
        transcript_original = TranscriptDocument(
            video_id=video.video_id,
            language=video.language or "",
            source_kind="local_asr",
            full_text="\n\n".join(segment.text for segment in segments if segment.text.strip()),
            segments=segments,
            original_language=video.language or "",
            source_url=video.source_url,
            retrieved_at=datetime.utcnow().isoformat() + "Z",
            metadata={"duration_seconds": duration_seconds, "source_file": asr_video_path},
        )
        used_asr = True

    if transcript_original is None:
        return {
            "success": False,
            "video_id": video.video_id,
            "requires_asr": acquisition.requires_asr,
            "error": acquisition.error or "Transcript acquisition failed",
            "warnings": acquisition.warnings,
        }

    transcript_english = translate_transcript_to_english(
        transcript_original,
        source_language=transcript_original.original_language or transcript_original.language or video.language,
    )
    chunks_english = chunk_transcript_document(transcript_english, chunk_duration_seconds=chunk_duration_seconds)
    topics_english: List[Dict[str, Any]] = []
    if enrich_topics and chunks_english:
        topics_english = identify_topics_with_ollama(
            [chunk.to_dict() for chunk in chunks_english],
            ollama_url=ollama_url,
            model=topic_model,
        )

    artifacts: Dict[str, str] = {}
    if write_artifacts:
        artifacts["transcript_original"] = str(write_transcript_artifact(source_path, "transcript_original.json", transcript_original))
        artifacts["transcript_english"] = str(write_transcript_artifact(source_path, "transcript_english.json", transcript_english))
        artifacts["chunks_english"] = str(write_chunk_artifact(source_path, "chunks_english.json", chunks_english))
        if topics_english:
            artifacts["topics_english"] = str(write_topics_artifact(source_path, "topics_english.json", topics_english, video_id=video.video_id))

    return {
        "success": True,
        "video_id": video.video_id,
        "source_kind": transcript_original.source_kind,
        "used_asr": used_asr,
        "requires_asr": False,
        "artifacts": artifacts,
        "transcript_original": transcript_original,
        "transcript_english": transcript_english,
        "chunks_english": chunks_english,
        "topics_english": topics_english,
        "warnings": acquisition.warnings,
    }


def identify_topics_with_ollama(
    chunks: List[Dict[str, Any]],
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen2.5:7b",
    timeout_seconds: int = 300,
) -> List[Dict[str, Any]]:
    """
    Add `topic` + `keywords` fields to each chunk via an Ollama generate call.
    """
    import requests

    for index, chunk in enumerate(chunks):
        text = str(chunk.get("text", "")).strip()
        if not text:
            chunk["topic"] = f"Segment {index + 1}"
            chunk["keywords"] = []
            continue

        prompt = (
            "Analyze this transcript segment and provide:\n"
            "1. A short topic title (3-7 words)\n"
            "2. Key items/concepts mentioned (comma-separated list)\n\n"
            f"Transcript ({chunk.get('start_sec', 0):.0f}s - {chunk.get('end_sec', 0):.0f}s):\n"
            f"\"{text[:1500]}\"\n\n"
            "Respond in this exact format:\n"
            "TOPIC: <title>\n"
            "KEYWORDS: <keyword1>, <keyword2>, ...\n"
        )

        try:
            response = requests.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_gpu": 1},
                },
                timeout=None if timeout_seconds <= 0 else timeout_seconds,
            )
            response.raise_for_status()
            raw_text = response.json().get("response", "")
        except Exception:
            chunk["topic"] = f"Segment {index + 1}"
            chunk["keywords"] = []
            continue

        topic = ""
        keywords: List[str] = []
        for line in raw_text.splitlines():
            clean = line.strip().lstrip("-* ").replace("**", "")
            topic_match = re.match(r"(?i)^topic\s*:\s*(.+)$", clean)
            keywords_match = re.match(r"(?i)^keywords?\s*:\s*(.+)$", clean)
            if topic_match:
                topic = topic_match.group(1).strip()
            elif keywords_match:
                keywords = [part.strip() for part in keywords_match.group(1).split(",") if part.strip()]

        chunk["topic"] = topic or f"Segment {index + 1}"
        chunk["keywords"] = keywords

    return chunks
