"""
Factory for creating vector stores based on environment configuration.

Supports dual-dimension architecture:
- 1536-dim: Online/global search (OpenAI embeddings, Pinecone)
- 768-dim: Offline/local search (all-mpnet-base-v2, local ChromaDB)

Usage:
    from offline_tools.vectordb import get_vector_store

    # Auto-select based on offline_mode setting
    store = get_vector_store()

    # Explicit dimension selection
    store = get_vector_store(dimension=768)  # Offline search
    store = get_vector_store(dimension=1536) # Online search

    # Or specify mode explicitly
    store = get_vector_store(mode='pinecone')
"""

import os
from typing import Optional

from .metadata import MetadataIndex

# VectorStore is optional (requires chromadb)
try:
    from .store import VectorStore, CHROMADB_AVAILABLE
except ImportError:
    VectorStore = None
    CHROMADB_AVAILABLE = False


def get_offline_mode() -> str:
    """Get the offline_mode setting from local_config"""
    try:
        from admin.local_config import get_local_config
        return get_local_config().get_offline_mode()
    except Exception:
        return "hybrid"


def get_default_dimension() -> int:
    """
    Get the default embedding dimension based on offline_mode and configured model.

    Returns:
        Dimension from configured embedding model, or default based on mode:
        - offline_only: 768 (all-mpnet-base-v2)
        - online: 1536 (OpenAI)
    """
    mode = get_offline_mode()

    # Try to get dimension from configured embedding model
    try:
        from admin.local_config import get_local_config
        from offline_tools.model_registry import AVAILABLE_MODELS
        config = get_local_config()
        model_id = config.get_embedding_model()
        if model_id and model_id in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[model_id].get("dimensions", 768)
    except Exception:
        pass

    # Fallback based on mode
    if mode == "offline_only":
        return 768
    return 1536


def get_chroma_path_for_dimension(dimension: int, base_path: str = None) -> str:
    """
    Get the ChromaDB path for a specific dimension.

    Args:
        dimension: Embedding dimension (384, 768, 1024, 1536, etc.)
        base_path: Base backup path (auto-detected if None)

    Returns:
        Path to the dimension-specific ChromaDB directory (e.g., chroma_db_768)
    """
    if base_path is None:
        try:
            from admin.local_config import get_local_config
            base_path = get_local_config().get_backup_folder()
        except Exception:
            pass

        if not base_path:
            base_path = os.getenv("BACKUP_PATH", "data")

    # Dynamic path for any dimension
    return os.path.join(base_path, f"chroma_db_{dimension}")


def get_vector_store(mode: Optional[str] = None, dimension: Optional[int] = None, **kwargs):
    """
    Factory function to get the appropriate vector store.

    Args:
        mode: 'local', 'local_768', 'local_1536', 'pinecone', 'railway', or None
        dimension: Explicit dimension (768 or 1536). Overrides mode-based selection.
        **kwargs: Additional arguments passed to store constructor

    Returns:
        VectorStore or PineconeStore instance

    Dimension selection priority:
    1. Explicit dimension parameter
    2. Mode-specific (local_768, local_1536)
    3. offline_mode setting (offline_only -> 768, otherwise -> 1536)
    """
    if mode is None:
        mode = os.getenv("VECTOR_DB_MODE", "local").lower()

    # Handle dimension-specific local modes
    if mode == "local_768":
        mode = "local"
        dimension = 768
    elif mode == "local_1536":
        mode = "local"
        dimension = 1536

    # Auto-detect dimension if not specified
    if dimension is None and mode == "local":
        dimension = get_default_dimension()

    if mode == "local":
        # Check if ChromaDB is available
        if VectorStore is None or not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. For cloud deployments, set VECTOR_DB_MODE=pinecone. "
                "For local development, install with: pip install chromadb"
            )
        # Get dimension-specific path
        persist_dir = kwargs.pop("persist_dir", None)
        if persist_dir is None:
            persist_dir = get_chroma_path_for_dimension(dimension or 1536)
        return VectorStore(persist_dir=persist_dir, **kwargs)

    elif mode in ("pinecone", "global", "railway"):
        # Cloud modes: pinecone, global, railway
        # All use Pinecone for vector storage with OpenAI embeddings (1536-dim)
        from .pinecone_store import PineconeStore
        return PineconeStore(**kwargs)

    elif mode == "qdrant":
        raise NotImplementedError("Qdrant support coming soon")

    else:
        raise ValueError(f"Unknown vector store mode: {mode}")


def get_vector_store_for_search(fallback: bool = True):
    """
    Get the appropriate vector store for search based on offline_mode.

    This is the main entry point for search operations.

    Args:
        fallback: If True in hybrid mode, returns tuple (primary, fallback)
                  If False, returns only primary store

    Returns:
        VectorStore instance, or tuple of (primary, fallback) if fallback=True in hybrid mode
    """
    offline_mode = get_offline_mode()
    local_dimension = get_default_dimension()  # Uses configured embedding model

    if offline_mode == "offline_only":
        # Offline only - use configured local embedding dimension
        return get_vector_store(mode="local", dimension=local_dimension)

    elif offline_mode == "online_only":
        # Online only - use 1536-dim (Pinecone or local)
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if pinecone_key:
            return get_vector_store(mode="pinecone")
        return get_vector_store(mode="local", dimension=1536)

    else:
        # Hybrid mode
        if fallback:
            # Return both stores for fallback chain
            pinecone_key = os.getenv("PINECONE_API_KEY")
            if pinecone_key:
                primary = get_vector_store(mode="pinecone")
            else:
                primary = get_vector_store(mode="local", dimension=1536)

            fallback_store = get_vector_store(mode="local", dimension=local_dimension)
            return (primary, fallback_store)
        else:
            # Just return primary
            pinecone_key = os.getenv("PINECONE_API_KEY")
            if pinecone_key:
                return get_vector_store(mode="pinecone")
            return get_vector_store(mode="local", dimension=1536)


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
