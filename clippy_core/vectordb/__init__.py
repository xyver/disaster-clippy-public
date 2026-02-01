"""
Vector Database Module for clippy_core.

Provides pluggable vector storage backends for semantic search.

Supported backends:
- ChromaDB (local/offline) - via chromadb.py or adapter to offline_tools
- Pinecone (cloud) - via pinecone.py or adapter to offline_tools
- pgvector (Supabase) - via pgvector.py (new implementation)

Usage:
    from clippy_core.vectordb import get_vector_store
    from clippy_core import ClippyConfig

    # Using config
    config = ClippyConfig(vector_db_mode="pgvector", pgvector_connection_string="...")
    store = get_vector_store(config)

    # Direct instantiation
    from clippy_core.vectordb.pgvector import PgVectorStore
    store = PgVectorStore(connection_string="...", table_name="source_vectors")
"""

from typing import Optional, Union
from .base import VectorStoreBase, SyncVectorStoreBase


def get_vector_store(config=None, mode: Optional[str] = None):
    """
    Factory function to get the appropriate vector store.

    Args:
        config: ClippyConfig instance (optional)
        mode: Override mode ("local", "pinecone", "pgvector")

    Returns:
        VectorStore instance (sync or async depending on backend)
    """
    # Determine mode
    if mode is None and config is not None:
        mode = config.vector_db_mode
    if mode is None:
        import os
        mode = os.getenv("VECTOR_DB_MODE", "local")

    mode = mode.lower()

    if mode in ("local", "chromadb"):
        # Try to use existing ChromaDB from offline_tools
        try:
            from offline_tools.vectordb.store import VectorStore as ChromaDBStore
            if config:
                from .factory import get_chroma_path
                persist_dir = get_chroma_path(config.embedding_dimension, config.backup_path)
                return ChromaDBStore(persist_dir=persist_dir)
            return ChromaDBStore()
        except ImportError:
            # Fall back to clippy_core's own implementation
            from .chromadb import ChromaDBStore
            if config:
                return ChromaDBStore(
                    persist_dir=config.backup_path,
                    dimension=config.embedding_dimension
                )
            return ChromaDBStore()

    elif mode in ("pinecone", "global"):
        # Try to use existing Pinecone from offline_tools
        try:
            from offline_tools.vectordb.pinecone_store import PineconeStore
            return PineconeStore()
        except ImportError:
            from .pinecone import PineconeStore
            return PineconeStore()

    elif mode == "pgvector":
        from .pgvector import PgVectorStore
        if config and config.pgvector_connection_string:
            return PgVectorStore(
                connection_string=config.pgvector_connection_string,
                table_name=config.pgvector_table_name,
                embedding_dimension=config.embedding_dimension
            )
        raise ValueError("pgvector mode requires pgvector_connection_string in config")

    else:
        raise ValueError(f"Unknown vector store mode: {mode}")


# Export base classes
__all__ = [
    "VectorStoreBase",
    "SyncVectorStoreBase",
    "get_vector_store",
]
