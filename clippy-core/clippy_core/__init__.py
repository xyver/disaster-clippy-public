"""
clippy_core - portable cited search and chat over your own documents.

Extracted from Disaster Clippy. Build a one-file index from PDFs, search it
with keyword + semantic ranking, and get answers that cite document,
section and page.

Quick start (Python):

    from clippy_core import ChatService, ClippyConfig
    from clippy_core.vectordb import SQLiteHybridStore

    config = ClippyConfig(index_path="index.sqlite")
    chat = ChatService(SQLiteHybridStore(config.index_path), config=config)
    print(chat.chat_sync("What is required for ...?").text)

Quick start (CLI):  clippy build --manifest sources.yaml
                    clippy ask "What is required for ...?"

See HANDOFF.md for architecture and extension points.
"""

from .config import ClippyConfig
from .context import ContextFormatter
from .schemas import (Chunk, ChatMessage, ChatResponse, DocType, ResponseMethod, SearchMethod,
                      SearchResponse, SearchResult, SourceInfo)
from .chat import ChatService
from .llm import LLMBackend
from .vectordb import SQLiteHybridStore, SyncVectorStoreBase, VectorStoreBase, get_vector_store

__version__ = "0.2.0"

__all__ = [
    "ChatService", "ClippyConfig", "ContextFormatter", "LLMBackend",
    "Chunk", "ChatMessage", "ChatResponse", "SearchResult", "SearchResponse", "SourceInfo",
    "SearchMethod", "ResponseMethod", "DocType",
    "VectorStoreBase", "SyncVectorStoreBase", "SQLiteHybridStore", "get_vector_store",
]
