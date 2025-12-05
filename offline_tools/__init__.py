"""
Offline Tools Module

Core tools for backup, indexing, embedding, scraping, and source management.

Submodules:
- backup/     : HTML backup utilities
- cloud/      : Cloud storage (R2) integration
- scraper/    : Web scraping tools (Appropedia, MediaWiki, Fandom, PDF, etc.)
- vectordb/   : Vector database abstractions (ChromaDB, Pinecone)
"""

from .schemas import (
    # Schema classes
    SourceManifest,
    DocumentMetadata,
    MetadataFile,
    IndexFile,
    VectorsFile,
    BackupManifest,
    # File name functions
    get_manifest_file,
    get_metadata_file,
    get_index_file,
    get_vectors_file,
    get_backup_manifest_file,
    # Schema version
    CURRENT_SCHEMA_VERSION,
)
from .embeddings import EmbeddingService
from .source_manager import SourceManager
from .registry import SourcePackRegistry, SourcePack, PackTier

__all__ = [
    # Schema classes
    'SourceManifest',
    'DocumentMetadata',
    'MetadataFile',
    'IndexFile',
    'VectorsFile',
    'BackupManifest',
    # File name functions
    'get_manifest_file',
    'get_metadata_file',
    'get_index_file',
    'get_vectors_file',
    'get_backup_manifest_file',
    'CURRENT_SCHEMA_VERSION',
    # Services
    'EmbeddingService',
    'SourceManager',
    'SourcePackRegistry',
    'SourcePack',
    'PackTier',
]
