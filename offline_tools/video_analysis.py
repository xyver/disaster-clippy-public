"""
Video analysis toolkit for video-heavy ZIM archives.

This module is intentionally standalone so maintainers can integrate it into
existing job pipelines without taking any orchestration-specific code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class VideoRecord:
    """Lightweight metadata for one video entry in a ZIM."""

    video_id: str
    title: str
    description: str
    duration_seconds: int
    video_path: Optional[str]
    thumbnail_path: Optional[str]
    upload_date: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.video_id,
            "title": self.title,
            "description": self.description,
            "duration": self.duration_seconds,
            "video_path": self.video_path,
            "thumbnail_path": self.thumbnail_path,
            "upload_date": self.upload_date,
        }


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

