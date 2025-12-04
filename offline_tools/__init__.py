"""
Offline Tools Module

Core tools for backup, indexing, embedding, and source management.
"""

from .schemas import (
    SourceMetadata,
    DocumentMetadata,
    EmbeddingsFile,
    BackupManifest,
    get_source_file,
    get_documents_file,
    get_embeddings_file,
    get_backup_manifest_file,
)
from .embeddings import EmbeddingService
from .source_manager import SourceManager
from .registry import SourcePackRegistry, SourcePack, PackTier

__all__ = [
    # Schemas
    'SourceMetadata',
    'DocumentMetadata',
    'EmbeddingsFile',
    'BackupManifest',
    'get_source_file',
    'get_documents_file',
    'get_embeddings_file',
    'get_backup_manifest_file',
    # Services
    'EmbeddingService',
    'SourceManager',
    'SourcePackRegistry',
    'SourcePack',
    'PackTier',
]
