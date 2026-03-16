"""
Shared data models for video transcript acquisition and processing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VideoRecord:
    """Lightweight metadata for one video entry in a source."""

    video_id: str
    title: str
    description: str = ""
    duration_seconds: int = 0
    video_path: Optional[str] = None
    thumbnail_path: Optional[str] = None
    upload_date: str = ""
    source_url: str = ""
    channel_name: str = ""
    language: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptSegment:
    """Canonical timed transcript segment."""

    segment_id: str
    start_sec: float
    end_sec: float
    text: str
    language: str = ""
    source_kind: str = ""
    original_segment_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptDocument:
    """A normalized transcript artifact stored in raw_data."""

    video_id: str
    language: str
    source_kind: str
    full_text: str
    segments: List[TranscriptSegment] = field(default_factory=list)
    original_language: str = ""
    translation_target_language: str = ""
    translation_model: str = ""
    translation_generated_at: str = ""
    source_url: str = ""
    retrieved_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["segments"] = [segment.to_dict() for segment in self.segments]
        return data


@dataclass
class TranscriptChunk:
    """Post-translation chunk used for enrichment and indexing."""

    chunk_id: str
    video_id: str
    start_sec: float
    end_sec: float
    text: str
    language: str = "en"
    text_original: str = ""
    text_translated: str = ""
    transcript_source: str = ""
    topic: str = ""
    keywords: List[str] = field(default_factory=list)
    original_segment_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
