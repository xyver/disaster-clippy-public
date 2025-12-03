"""
Layered Metadata Schema Definitions

This module defines the file structure and schemas for source packages.
All files use the {source_id}_ prefix for consistency.

File Structure:
    {source_id}/
        {source_id}_source.json           # Source-level metadata
        {source_id}_documents.json        # Document metadata (quick scan)
        {source_id}_embeddings.json       # Vectors only (for ChromaDB import)
        {source_id}_backup_manifest.json  # Backup manifest (URL -> filename)
        {source_id}_manifest.json         # Distribution manifest
        {source_id}_categories/           # Category rollups (optional)
            {category}.json
        pages/                            # HTML backup files
        assets/                           # Images, CSS, etc.
"""

from typing import Dict, List, Optional, TypedDict, Any
from dataclasses import dataclass, field
from datetime import datetime


# =============================================================================
# FILE NAMING
# =============================================================================

def get_source_file(source_id: str) -> str:
    """Source-level metadata file"""
    return f"{source_id}_source.json"

def get_documents_file(source_id: str) -> str:
    """Document metadata file (quick scan)"""
    return f"{source_id}_documents.json"

def get_embeddings_file(source_id: str) -> str:
    """Embeddings file (vectors only)"""
    return f"{source_id}_embeddings.json"

def get_backup_manifest_file(source_id: str) -> str:
    """Backup manifest file"""
    return f"{source_id}_backup_manifest.json"

def get_distribution_manifest_file(source_id: str) -> str:
    """Distribution manifest file"""
    return f"{source_id}_manifest.json"

def get_categories_dir(source_id: str) -> str:
    """Categories directory"""
    return f"{source_id}_categories"

# Legacy file names (for migration)
LEGACY_FILES = {
    "metadata": "{source_id}_metadata.json",
    "index": "{source_id}_index.json",
    "backup_manifest": "manifest.json",
}


# =============================================================================
# SCHEMA: SOURCE-LEVEL METADATA
# =============================================================================

@dataclass
class CategoryStats:
    """Stats for a single category"""
    count: int = 0
    total_chars: int = 0


@dataclass
class SourceMetadata:
    """
    Source-level metadata ({source_id}_source.json)

    Contains high-level info about the source for quick browsing.
    """
    source_id: str
    name: str
    description: str = ""
    license: str = "Unknown"
    base_url: str = ""

    # Counts
    total_docs: int = 0
    total_chars: int = 0

    # Category breakdown
    categories: Dict[str, CategoryStats] = field(default_factory=dict)

    # Timestamps
    created_at: str = ""
    last_backup: str = ""
    last_indexed: str = ""

    # Schema version for future migrations
    schema_version: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "name": self.name,
            "description": self.description,
            "license": self.license,
            "base_url": self.base_url,
            "total_docs": self.total_docs,
            "total_chars": self.total_chars,
            "categories": {
                k: {"count": v.count, "total_chars": v.total_chars}
                for k, v in self.categories.items()
            },
            "created_at": self.created_at,
            "last_backup": self.last_backup,
            "last_indexed": self.last_indexed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceMetadata":
        categories = {}
        for k, v in data.get("categories", {}).items():
            if isinstance(v, dict):
                categories[k] = CategoryStats(
                    count=v.get("count", 0),
                    total_chars=v.get("total_chars", 0)
                )
            else:
                categories[k] = CategoryStats(count=v, total_chars=0)

        return cls(
            source_id=data.get("source_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            license=data.get("license", "Unknown"),
            base_url=data.get("base_url", ""),
            total_docs=data.get("total_docs", 0),
            total_chars=data.get("total_chars", 0),
            categories=categories,
            created_at=data.get("created_at", ""),
            last_backup=data.get("last_backup", ""),
            last_indexed=data.get("last_indexed", ""),
            schema_version=data.get("schema_version", 2),
        )


# =============================================================================
# SCHEMA: DOCUMENT METADATA
# =============================================================================

@dataclass
class DocumentMetadata:
    """
    Single document metadata entry.

    This is stored in {source_id}_documents.json for quick scanning
    without loading full content or embeddings.
    """
    doc_id: str
    title: str
    url: str
    content_hash: str
    char_count: int
    categories: List[str] = field(default_factory=list)
    scraped_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content_hash": self.content_hash,
            "char_count": self.char_count,
            "categories": self.categories,
            "scraped_at": self.scraped_at,
        }


@dataclass
class DocumentsFile:
    """
    Documents metadata file ({source_id}_documents.json)

    Contains metadata for all documents, keyed by doc_id.
    Used for quick scanning without loading embeddings.
    """
    schema_version: int = 2
    source_id: str = ""
    document_count: int = 0
    total_chars: int = 0
    last_updated: str = ""
    documents: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "document_count": self.document_count,
            "total_chars": self.total_chars,
            "last_updated": self.last_updated,
            "documents": self.documents,
        }


# =============================================================================
# SCHEMA: EMBEDDINGS
# =============================================================================

@dataclass
class EmbeddingsFile:
    """
    Embeddings file ({source_id}_embeddings.json)

    Contains ONLY vectors, no content duplication.
    Content is in pages/, metadata is in _documents.json.
    """
    schema_version: int = 2
    source_id: str = ""
    embedding_model: str = "text-embedding-3-small"
    dimensions: int = 1536
    document_count: int = 0
    created_at: str = ""

    # Vectors keyed by doc_id
    vectors: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "embedding_model": self.embedding_model,
            "dimensions": self.dimensions,
            "document_count": self.document_count,
            "created_at": self.created_at,
            "vectors": self.vectors,
        }


# =============================================================================
# SCHEMA: BACKUP MANIFEST
# =============================================================================

@dataclass
class BackupManifest:
    """
    Backup manifest ({source_id}_backup_manifest.json)

    Maps URLs to local filenames for the backup.
    """
    schema_version: int = 2
    source_id: str = ""
    base_url: str = ""
    scraper_type: str = ""
    created_at: str = ""
    last_updated: str = ""
    total_pages: int = 0
    total_size_bytes: int = 0

    # URL -> page info
    pages: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Asset URL -> asset info
    assets: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "base_url": self.base_url,
            "scraper_type": self.scraper_type,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "total_pages": self.total_pages,
            "total_size_bytes": self.total_size_bytes,
            "pages": self.pages,
            "assets": self.assets,
        }


# =============================================================================
# SCHEMA: DISTRIBUTION MANIFEST
# =============================================================================

@dataclass
class DistributionManifest:
    """
    Distribution manifest ({source_id}_manifest.json)

    Package info for distribution via R2.
    """
    schema_version: int = 2
    source_id: str = ""
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    created_at: str = ""

    # Content info
    document_count: int = 0
    has_embeddings: bool = False
    has_categories: bool = False

    # License and attribution
    license: str = "Unknown"
    attribution: str = ""
    base_url: str = ""

    # Files included in package
    files: List[str] = field(default_factory=list)

    # Size info
    total_size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "created_at": self.created_at,
            "document_count": self.document_count,
            "has_embeddings": self.has_embeddings,
            "has_categories": self.has_categories,
            "license": self.license,
            "attribution": self.attribution,
            "base_url": self.base_url,
            "files": self.files,
            "total_size_bytes": self.total_size_bytes,
        }


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_source_files(source_path: str, source_id: str) -> Dict[str, Any]:
    """
    Validate that a source has the required files in the new format.

    Returns:
        Dict with validation results:
        - is_valid: bool
        - has_source: bool (source-level metadata)
        - has_documents: bool (document metadata)
        - has_embeddings: bool (vectors)
        - has_backup_manifest: bool
        - has_distribution_manifest: bool
        - has_categories: bool
        - issues: List[str]
        - legacy_files: List[str] (old format files found)
    """
    from pathlib import Path

    path = Path(source_path)
    result = {
        "is_valid": False,
        "has_source": False,
        "has_documents": False,
        "has_embeddings": False,
        "has_backup_manifest": False,
        "has_distribution_manifest": False,
        "has_categories": False,
        "issues": [],
        "legacy_files": [],
    }

    # Check new format files
    result["has_source"] = (path / get_source_file(source_id)).exists()
    result["has_documents"] = (path / get_documents_file(source_id)).exists()
    result["has_embeddings"] = (path / get_embeddings_file(source_id)).exists()
    result["has_backup_manifest"] = (path / get_backup_manifest_file(source_id)).exists()
    result["has_distribution_manifest"] = (path / get_distribution_manifest_file(source_id)).exists()
    result["has_categories"] = (path / get_categories_dir(source_id)).exists()

    # Check for legacy files
    legacy_metadata = path / f"{source_id}_metadata.json"
    legacy_index = path / f"{source_id}_index.json"
    legacy_manifest = path / "manifest.json"

    if legacy_metadata.exists() and not result["has_documents"]:
        result["legacy_files"].append(str(legacy_metadata.name))
    if legacy_index.exists() and not result["has_embeddings"]:
        result["legacy_files"].append(str(legacy_index.name))
    if legacy_manifest.exists() and not result["has_backup_manifest"]:
        result["legacy_files"].append(str(legacy_manifest.name))

    # Determine validity
    # Minimum required: source metadata + documents + embeddings
    if result["has_source"] and result["has_documents"] and result["has_embeddings"]:
        result["is_valid"] = True
    else:
        if not result["has_source"]:
            result["issues"].append(f"Missing {get_source_file(source_id)}")
        if not result["has_documents"]:
            result["issues"].append(f"Missing {get_documents_file(source_id)}")
        if not result["has_embeddings"]:
            result["issues"].append(f"Missing {get_embeddings_file(source_id)}")

    if result["legacy_files"]:
        result["issues"].append(f"Legacy files need migration: {', '.join(result['legacy_files'])}")

    return result


# =============================================================================
# SCHEMA VERSION
# =============================================================================

CURRENT_SCHEMA_VERSION = 2

def get_schema_version(file_data: Dict[str, Any]) -> int:
    """Get schema version from file data, defaulting to 1 for old files"""
    return file_data.get("schema_version", 1)

def needs_migration(file_data: Dict[str, Any]) -> bool:
    """Check if file needs migration to current schema"""
    return get_schema_version(file_data) < CURRENT_SCHEMA_VERSION
