"""
Core data types for clippy_core.

These are the types used by the chat and search APIs.
Kept separate from offline_tools/schemas.py which has source file structures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class SearchMethod(Enum):
    """Method used for search."""
    SEMANTIC = "semantic"           # Embedding-based vector search (online)
    LOCAL_SEMANTIC = "local_semantic"  # Local embeddings (offline)
    KEYWORD = "keyword"             # Simple keyword matching (fallback)


class ResponseMethod(Enum):
    """Method used for response generation."""
    CLOUD_LLM = "cloud_llm"    # OpenAI/Anthropic API
    LOCAL_LLM = "local_llm"    # Ollama/llama.cpp
    SIMPLE = "simple"          # No LLM, formatted response


class DocType(Enum):
    """Document type classification."""
    GUIDE = "guide"        # How-to, tutorial, step-by-step
    ARTICLE = "article"    # Informational content
    REFERENCE = "reference"  # Technical reference, specs
    PRODUCT = "product"    # Product page, equipment
    ACADEMIC = "academic"  # Research paper, study


@dataclass
class SearchResult:
    """
    A single search result from vector search.

    This is the standard format returned by all vector stores.
    """
    id: str
    content: str
    source_id: str
    title: str = ""
    url: str = ""
    local_url: str = ""
    score: float = 0.0
    doc_type: str = "article"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chromadb(cls, result: Dict[str, Any]) -> "SearchResult":
        """Create from ChromaDB result format."""
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
        """Create from Pinecone match format."""
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
        """Create from pgvector/Supabase row format."""
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
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "local_url": self.local_url,
            "score": self.score,
            "doc_type": self.doc_type,
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
