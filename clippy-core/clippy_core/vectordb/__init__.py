"""
Vector store backends for clippy_core.

- SQLiteHybridStore (default): one-file index, keyword + semantic search.
- PgVectorStore (optional): Postgres/Supabase with pgvector. Semantic only.

Any object with ``search(query, n_results=..., sources=...)`` returning
SearchResult objects (sync or async) works with ChatService. Subclass
VectorStoreBase / SyncVectorStoreBase to add a backend.
"""

from typing import Optional

from .base import SyncVectorStoreBase, VectorStoreBase
from .sqlite_hybrid import IndexMismatchError, SQLiteHybridStore


def get_vector_store(config=None, mode: Optional[str] = None, embedder=None):
    """Create the store named by ``mode`` or ``config.vector_db_mode``."""
    from ..config import ClippyConfig

    config = config or ClippyConfig.from_env()
    mode = (mode or config.vector_db_mode or "sqlite").lower()

    if mode == "sqlite":
        return SQLiteHybridStore(config.index_path, embedder=embedder,
                                 api_key=config.get_openai_api_key())
    if mode == "pgvector":
        from .pgvector import PgVectorStore
        return PgVectorStore(connection_string=config.pgvector_connection_string,
                             table_name=config.pgvector_table_name)
    raise ValueError(f"Unknown vector_db_mode: {mode!r} (expected 'sqlite' or 'pgvector')")


__all__ = ["VectorStoreBase", "SyncVectorStoreBase", "SQLiteHybridStore",
           "IndexMismatchError", "get_vector_store"]
