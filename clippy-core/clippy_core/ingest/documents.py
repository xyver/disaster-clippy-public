"""Document and Page: what a Chunker receives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Page:
    num: int      # 1-based PDF page number (what #page=N links use)
    text: str


@dataclass
class Document:
    doc_id: str
    path: Path
    title: str
    source_id: str
    edition: str = ""
    url: str = ""                  # public link to the original file, without #page
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: List[Page] = field(default_factory=list)

    def page_url(self, page: Optional[int]) -> str:
        if not self.url:
            return ""
        return f"{self.url}#page={page}" if page else self.url

    def base_metadata(self) -> Dict[str, Any]:
        """Metadata every chunk of this document starts with."""
        meta = {"doc_id": self.doc_id, "doc_title": self.title, "path": str(self.path)}
        if self.edition:
            meta["edition"] = self.edition
        meta.update(self.metadata)
        return meta
