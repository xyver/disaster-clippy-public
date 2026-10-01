"""
Abstract base class for vector stores.

All vector store implementations (ChromaDB, Pinecone, pgvector) must implement
this interface to be usable with ChatService.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from ..schemas import SearchResult, SourceInfo


class VectorStoreBase(ABC):
    """
    Abstract interface for vector storage backends.

    Implementations:
    - ChromaDBStore: Local ChromaDB for offline/development
    - PineconeStore: Pinecone cloud for production
    - PgVectorStore: Supabase pgvector for integrated deployments
    """

    @abstractmethod
    async def search(
        self,
        query: str,
        n_results: int = 10,
        sources: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Semantic search for relevant documents.

        Args:
            query: Natural language search query
            n_results: Maximum number of results to return
            sources: Optional list of source IDs to filter by (None = all sources)

        Returns:
            List of SearchResult objects, sorted by relevance (highest first)
        """
        pass

    @abstractmethod
    async def get_sources(self) -> List[SourceInfo]:
        """
        List all available sources with document counts.

        Returns:
            List of SourceInfo objects
        """
        pass

    # Optional methods - implementations can override if supported

    async def search_keyword(
        self,
        query: str,
        n_results: int = 10,
        sources: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Keyword-based search (fallback for offline mode).

        Default implementation raises NotImplementedError.
        Override in stores that support keyword search.

        Args:
            query: Search query string
            n_results: Maximum number of results
            sources: Optional source filter

        Returns:
            List of SearchResult objects
        """
        raise NotImplementedError("This store doesn't support keyword search")

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a single document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document dict or None if not found
        """
        raise NotImplementedError("This store doesn't support document retrieval by ID")

    async def health_check(self) -> bool:
        """
        Check if the vector store is healthy and accessible.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Default: try to get sources as a health check
            await self.get_sources()
            return True
        except Exception:
            return False


class SyncVectorStoreBase(ABC):
    """
    Synchronous version of VectorStoreBase.

    For use with existing synchronous code (ChromaDB, etc.)
    ChatService will wrap these in async calls.
    """

    @abstractmethod
    def search(
        self,
        query: str,
        n_results: int = 10,
        sources: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Synchronous semantic search."""
        pass

    @abstractmethod
    def get_sources(self) -> List[SourceInfo]:
        """Synchronous source listing."""
        pass

    def search_keyword(
        self,
        query: str,
        n_results: int = 10,
        sources: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """Synchronous keyword search (optional)."""
        raise NotImplementedError("This store doesn't support keyword search")

    def health_check(self) -> bool:
        """Synchronous health check."""
        try:
            self.get_sources()
            return True
        except Exception:
            return False
