"""
Core data types for clippy_core.

Metadata conventions
--------------------
Every chunk carries a free-form ``metadata`` dict. These keys have agreed
meanings and are used for citations, filters and evaluation. Fill in the ones
that apply to your documents; add any others you need.

    doc_id         stable id of the source document (e.g. "isu-tech-rules-singles")
    doc_title      human title of the document
    edition        edition / season / version (e.g. "2025-26")
    section_id     the citable unit inside the document (e.g. "Rule 611")
    section_title  heading text of that unit
    page_start     first PDF page (1-based) the chunk comes from
    page_end       last PDF page (1-based) the chunk comes from
    url            link to the original, ideally with "#page=N"

Anything else (discipline, level, language, ...) is passed through untouched
and can be used with ``filters=`` in search.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SearchMethod(Enum):
    """Method used for search."""
    SEMANTIC = "semantic"          # Embedding similarity only
    KEYWORD = "keyword"            # Full-text (BM25) only
    HYBRID = "hybrid"              # Both, fused with reciprocal rank fusion
    LOCAL_SEMANTIC = "local_semantic"  # Kept for compatibility with older callers


class ResponseMethod(Enum):
    """Method used for response generation."""
    CLOUD_LLM = "cloud_llm"
    LOCAL_LLM = "local_llm"
    SIMPLE = "simple"   # No LLM: extractive answer built from the passages


class DocType(Enum):
    """Document type classification."""
    GUIDE = "guide"
    ARTICLE = "article"
    REFERENCE = "reference"
    PRODUCT = "product"
    ACADEMIC = "academic"
    RULES = "rules"


@dataclass
class Chunk:
    """A unit of text ready to be indexed. Produced by a Chunker."""
    id: str
    content: str
    source_id: str
    title: str = ""
    url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single search result. The standard format returned by all stores."""
    id: str
    content: str
    source_id: str
    title: str = ""
    url: str = ""
    local_url: str = ""
    score: float = 0.0
    doc_type: str = "article"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- citation helpers -------------------------------------------------

    @property
    def page_label(self) -> str:
        start = self.metadata.get("page_start")
        end = self.metadata.get("page_end")
        if not start:
            return ""
        if end and end != start:
            return f"pp. {start}-{end}"
        return f"p. {start}"

    def citation(self) -> str:
        """Short human-readable citation, e.g. 'Tech Rules 2025-26 · Rule 611 · p. 147'."""
        m = self.metadata
        doc = m.get("doc_title") or self.title or self.source_id
        edition = m.get("edition")
        parts = [f"{doc} {edition}".strip() if edition else doc]
        if m.get("section_id"):
            parts.append(str(m["section_id"]))
        if self.page_label:
            parts.append(self.page_label)
        return " · ".join(p for p in parts if p)

    # ---- converters ------------------------------------------------------

    @classmethod
    def from_chromadb(cls, result: Dict[str, Any]) -> "SearchResult":
        metadata = result.get("metadata", {})
        return cls(
            id=result.get("id", ""),
            content=result.get("content", ""),
            source_id=metadata.get("source", ""),
            title=metadata.get("title", ""),
            url=metadata.get("url", ""),
            local_url=metadata.get("local_url", ""),
            score=result.get("score", 0.0),
            doc_type=metadata.get("doc_type", "article"),
            metadata=metadata,
        )

    @classmethod
    def from_pinecone(cls, match: Dict[str, Any]) -> "SearchResult":
        metadata = match.get("metadata", {})
        return cls(
            id=match.get("id", ""),
            content=metadata.get("content", metadata.get("text", "")),
            source_id=metadata.get("source", ""),
            title=metadata.get("title", ""),
            url=metadata.get("url", ""),
            local_url=metadata.get("local_url", ""),
            score=match.get("score", 0.0),
            doc_type=metadata.get("doc_type", "article"),
            metadata=metadata,
        )

    @classmethod
    def from_pgvector(cls, row: Dict[str, Any]) -> "SearchResult":
        metadata = row.get("metadata", {})
        return cls(
            id=row.get("id", ""),
            content=row.get("content", ""),
            source_id=row.get("source_id", ""),
            title=metadata.get("title", row.get("title", "")),
            url=row.get("url", ""),
            local_url=row.get("local_url", ""),
            score=row.get("similarity", row.get("score", 0.0)),
            doc_type=row.get("doc_type", "article"),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "local_url": self.local_url,
            "score": self.score,
            "doc_type": self.doc_type,
            "citation": self.citation(),
            "metadata": self.metadata,
        }


@dataclass
class SearchResponse:
    """Response from a search operation."""
    results: List[SearchResult]
    query: str
    method: SearchMethod
    total_results: int = 0
    error: Optional[str] = None

    def __post_init__(self):
        if self.total_results == 0:
            self.total_results = len(self.results)


@dataclass
class ChatMessage:
    """A single message in conversation history."""
    role: str  # "user" or "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        return cls(role="assistant", content=content)


@dataclass
class ChatResponse:
    """Response from chat generation."""
    text: str
    method: ResponseMethod
    sources_used: List[str] = field(default_factory=list)
    search_results: List[SearchResult] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "method": self.method.value,
            "sources_used": self.sources_used,
            "citations": [
                {"n": i, "citation": r.citation(), "url": r.url, "id": r.id}
                for i, r in enumerate(self.search_results, 1)
            ],
            "error": self.error,
        }


@dataclass
class SourceInfo:
    """Information about an available source."""
    id: str
    name: str
    count: int
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "count": self.count,
            "description": self.description,
        }
