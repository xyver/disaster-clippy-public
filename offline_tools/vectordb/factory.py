"""
Factory for creating vector stores based on environment configuration.

Usage:
    from offline_tools.vectordb import get_vector_store

    # Uses VECTOR_DB_MODE from .env (defaults to 'local')
    store = get_vector_store()

    # Or specify explicitly
    store = get_vector_store(mode='pinecone')
"""

import os
from typing import Optional

from .store import VectorStore
from .metadata import MetadataIndex


def get_vector_store(mode: Optional[str] = None, **kwargs):
    """
    Factory function to get the appropriate vector store.

    Args:
        mode: 'local', 'pinecone', 'railway', or None (uses env var)
        **kwargs: Additional arguments passed to store constructor

    Returns:
        VectorStore or PineconeStore instance
    """
    if mode is None:
        mode = os.getenv("VECTOR_DB_MODE", "local").lower()

    if mode == "local":
        # VectorStore now uses BACKUP_PATH/chroma by default
        persist_dir = kwargs.pop("persist_dir", None)
        return VectorStore(persist_dir=persist_dir, **kwargs)

    elif mode in ("pinecone", "global"):
        # Import here to avoid requiring pinecone when not used
        # "global" is an alias for pinecone mode
        from .pinecone_store import PineconeStore
        return PineconeStore(**kwargs)

    elif mode == "railway":
        # Railway uses persistent volume at /data
        persist_dir = os.getenv("RAILWAY_VOLUME_PATH", "/data/chroma")
        return VectorStore(persist_dir=persist_dir, **kwargs)

    elif mode == "qdrant":
        raise NotImplementedError("Qdrant support coming soon")

    else:
        raise ValueError(f"Unknown vector store mode: {mode}")


def get_metadata_store(mode: Optional[str] = None) -> MetadataIndex:
    """
    Get metadata index for the current mode.

    The metadata index uses BACKUP_PATH by default.
    For global/pinecone mode, uses cloud metadata.
    """
    if mode is None:
        mode = os.getenv("VECTOR_DB_MODE", "local").lower()

    if mode == "railway":
        index_dir = os.getenv("RAILWAY_VOLUME_PATH", "/data")
        return MetadataIndex(index_dir=index_dir)

    if mode in ("pinecone", "global"):
        # For global mode, metadata is fetched from cloud/Pinecone
        # MetadataIndex will use BACKUP_PATH for local cache
        return MetadataIndex()

    # Default uses BACKUP_PATH (handled by MetadataIndex)
    return MetadataIndex()
