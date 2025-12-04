"""
Vector Database Module

Handles vector storage for semantic search using ChromaDB or Pinecone.
"""

from .store import VectorStore
from .pinecone_store import PineconeStore
from .factory import get_vector_store, get_metadata_store
from .metadata import (
    MetadataIndex,
    classify_doc_type,
    DOC_TYPE_GUIDE,
    DOC_TYPE_ARTICLE,
    DOC_TYPE_PRODUCT,
    DOC_TYPE_ACADEMIC
)

__all__ = [
    'VectorStore',
    'PineconeStore',
    'get_vector_store',
    'get_metadata_store',
    'MetadataIndex',
    'classify_doc_type',
    'DOC_TYPE_GUIDE',
    'DOC_TYPE_ARTICLE',
    'DOC_TYPE_PRODUCT',
    'DOC_TYPE_ACADEMIC'
]
