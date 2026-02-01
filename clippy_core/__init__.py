"""
clippy_core - Extractable chat and search module from Disaster Clippy.

This module provides the core chat and search functionality that can be
used in other projects. It's designed to be copied as a folder and work
with minimal configuration.

Quick Start:
    from clippy_core import ChatService, ClippyConfig
    from clippy_core.vectordb.pgvector import PgVectorStore

    # Configure
    config = ClippyConfig(
        llm_provider="openai",
        llm_model="gpt-4o-mini"
    )

    # Create vector store (use your own backend)
    store = PgVectorStore(
        connection_string="postgresql://...",
        table_name="source_vectors"
    )

    # Create chat service
    chat = ChatService(config=config, vector_store=store)

    # Chat with custom prompt
    response = await chat.chat(
        message="How do I prepare for a hurricane?",
        sources=["fema-ready", "fl-building-code"],
        system_prompt="You are a preparedness advisor for Florida."
    )

Main Components:
    - ChatService: Main entry point for chat functionality
    - ClippyConfig: Configuration dataclass
    - SearchResult, ChatResponse: Data types
    - VectorStoreBase: Abstract interface for vector stores
    - PgVectorStore: Supabase pgvector implementation
"""

from .config import ClippyConfig
from .schemas import (
    SearchResult,
    SearchResponse,
    ChatMessage,
    ChatResponse,
    SourceInfo,
    SearchMethod,
    ResponseMethod,
    DocType,
)

# Import ChatService - may fail if dependencies not installed
try:
    from .chat import ChatService
except ImportError as e:
    ChatService = None
    _chat_import_error = str(e)

# Import vector store utilities
from .vectordb import VectorStoreBase, get_vector_store

__version__ = "0.1.0"

__all__ = [
    # Main interface
    "ChatService",
    "ClippyConfig",
    # Data types
    "SearchResult",
    "SearchResponse",
    "ChatMessage",
    "ChatResponse",
    "SourceInfo",
    # Enums
    "SearchMethod",
    "ResponseMethod",
    "DocType",
    # Vector store
    "VectorStoreBase",
    "get_vector_store",
]
