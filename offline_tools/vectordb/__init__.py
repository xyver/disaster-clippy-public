"""
Vector Database Module

Handles vector storage for semantic search using ChromaDB or Pinecone.
ChromaDB is optional - for cloud deployments, only Pinecone is available.
"""

# VectorStore is optional (requires chromadb which is not in cloud deployment)
try:
    from .store import VectorStore
except ImportError:
    VectorStore = None

from .pinecone_store import PineconeStore
from .factory import get_vector_store, get_metadata_store, get_vector_store_for_search
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
    'get_vector_store_for_search',
    'get_metadata_store',
    'MetadataIndex',
    'classify_doc_type',
    'DOC_TYPE_GUIDE',
    'DOC_TYPE_ARTICLE',
    'DOC_TYPE_PRODUCT',
    'DOC_TYPE_ACADEMIC'
]
