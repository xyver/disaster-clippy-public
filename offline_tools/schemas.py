"""
Layered Metadata Schema Definitions

This module defines the file structure and schemas for source packages.
Files use underscore prefix for standard names within each source folder.

File Structure:
    {source_id}/
        _manifest.json            # Source identity + distribution info
        _metadata.json            # Document metadata (quick scan)
        _index.json               # Full document content (for display)
        _vectors.json             # Embeddings only (for ChromaDB import)
        backup_manifest.json      # Backup manifest (URL -> filename)
        pages/                    # HTML backup files
        assets/                   # Images, CSS, etc.
"""

from typing import Dict, List, Optional, TypedDict, Any
from dataclasses import dataclass, field
from datetime import datetime


# =============================================================================
# FILE NAMING
# =============================================================================

def get_manifest_file() -> str:
    """Source manifest file (identity + distribution info)"""
    return "_manifest.json"

def get_metadata_file() -> str:
    """Document metadata file (quick scan)"""
    return "_metadata.json"

def get_index_file() -> str:
    """Full content index file (for display/scanning)"""
    return "_index.json"

def get_vectors_file(dimension: int = 1536) -> str:
    """
    Vectors file (embeddings only).

    Args:
        dimension: Embedding dimension (384, 768, 1024, or 1536)

    Returns:
        Filename for the vectors file (e.g., _vectors_768.json)
    """
    if dimension == 1536:
        return "_vectors.json"  # Default for backwards compatibility
    return f"_vectors_{dimension}.json"


def get_vectors_768_file() -> str:
    """768-dim vectors file for offline use (legacy helper)"""
    return "_vectors_768.json"


def get_vectors_1536_file() -> str:
    """1536-dim vectors file for online use (legacy helper)"""
    return "_vectors.json"


def get_vectors_384_file() -> str:
    """384-dim vectors file (MiniLM - lightweight)"""
    return "_vectors_384.json"


def get_vectors_1024_file() -> str:
    """1024-dim vectors file (E5-large - high quality)"""
    return "_vectors_1024.json"

def get_backup_manifest_file() -> str:
    """Backup manifest file (URL to filename mapping)"""
    return "backup_manifest.json"


# =============================================================================
# HTML FILENAME UTILITIES
# =============================================================================

def html_filename_to_url(filename: str, is_zim_source: bool = False) -> str:
    """
    Convert an HTML backup filename to a URL path.

    The scraper saves files like:
        Projects_Cooling_ACEvapCool.htm.html  (original was .htm)
        Projects_Cooling_SolarAC.html         (original was .html or no extension)

    This converts them to URL paths:
        /Projects/Cooling/ACEvapCool.htm
        /Projects/Cooling/SolarAC

    For ZIM sources, there are two formats:

    1. Wikipedia-style ZIMs: underscores are part of the article name
        Anatomical_terms_of_location.html -> Anatomical_terms_of_location

    2. WARC-style ZIMs (web archives): filenames contain full domain
        www.ready.gov_alerts.html -> https://www.ready.gov/alerts
        www.fema.gov_locations.html -> https://www.fema.gov/locations

    Args:
        filename: The HTML filename (e.g., "Projects_Cooling_Page.htm.html")
        is_zim_source: If True, check for WARC or Wikipedia-style ZIM format

    Returns:
        URL path or full URL for WARC-style sources
    """
    # Preserve .htm extension (scraper adds .html to all files including .htm)
    # Order matters: check .htm.html first, then .html
    url_path = filename.replace(".htm.html", ".htm").replace(".html", "")

    if is_zim_source:
        # Check for WARC-style ZIM (filename starts with domain like www.domain.com_)
        # These have a dot before the first underscore (domain pattern)
        first_underscore = url_path.find('_')
        if first_underscore > 0:
            prefix = url_path[:first_underscore]
            # Check if prefix looks like a domain (contains at least one dot)
            if '.' in prefix and not prefix.startswith('.'):
                # WARC-style: extract domain and path
                domain = prefix
                path_part = url_path[first_underscore + 1:] if first_underscore < len(url_path) - 1 else ''
                # Convert remaining underscores in path to slashes
                path = path_part.replace('_', '/')
                # Handle empty path (homepage)
                if not path:
                    return f"https://{domain}/"
                return f"https://{domain}/{path}"

        # Wikipedia-style ZIM: underscores are part of the article name
        return url_path
    else:
        # HTML scrapes: underscores represent path separators
        url_path = url_path.replace("_", "/")
        return f"/{url_path}"


def html_filename_to_title(filename: str) -> str:
    """
    Convert an HTML backup filename to a display title.

    Strips all extensions and converts underscores to spaces.

    Args:
        filename: The HTML filename (e.g., "Projects_Cooling_Page.htm.html")

    Returns:
        Display title (e.g., "Projects Cooling Page")
    """
    # Strip all extensions for display
    title = filename.replace(".htm.html", "").replace(".html", "").replace(".htm", "").replace("_", " ")
    return title


# =============================================================================
# TAG NORMALIZATION
# =============================================================================

# Irregular plurals and word variations to normalize
TAG_NORMALIZATIONS = {
    # Plurals -> singular
    "collectors": "collector",
    "systems": "system",
    "heaters": "heater",
    "cookers": "cooker",
    "panels": "panel",
    "projects": "project",
    "designs": "design",
    "plans": "plan",
    "houses": "house",
    "homes": "home",
    "buildings": "building",
    "pumps": "pump",
    "tanks": "tank",
    "batteries": "battery",
    "generators": "generator",
    "turbines": "turbine",
    "cells": "cell",
    "modules": "module",
    "inverters": "inverter",
    "controllers": "controller",
    "sensors": "sensor",
    "meters": "meter",
    "filters": "filter",
    "valves": "valve",
    "pipes": "pipe",
    "tubes": "tube",
    "coils": "coil",
    "fans": "fan",
    "vents": "vent",
    "windows": "window",
    "walls": "wall",
    "roofs": "roof",
    "floors": "floor",
    "doors": "door",
    "tools": "tool",
    "materials": "material",
    "sources": "source",
    "resources": "resource",
    "guides": "guide",
    "tutorials": "tutorial",
    "instructions": "instruction",
    "methods": "method",
    "techniques": "technique",
    "solutions": "solution",
    "applications": "application",
    "installations": "installation",
    "calculations": "calculation",
    "measurements": "measurement",
    "experiments": "experiment",
    "tests": "test",
    "results": "result",
    "costs": "cost",
    "savings": "saving",
    "benefits": "benefit",
    "improvements": "improvement",
    "upgrades": "upgrade",
    "modifications": "modification",
    "conversions": "conversion",
    "alternatives": "alternative",
    "options": "option",
    "features": "feature",
    "components": "component",
    "parts": "part",
    "units": "unit",
    "types": "type",
    "models": "model",
    "versions": "version",
    "examples": "example",
    "photos": "photo",
    "images": "image",
    "videos": "video",
    "diagrams": "diagram",
    "drawings": "drawing",
    "schematics": "schematic",
    "charts": "chart",
    "graphs": "graph",
    "tables": "table",
    "lists": "list",
    "links": "link",
    "references": "reference",
    "articles": "article",
    "pages": "page",
    "sections": "section",
    "chapters": "chapter",
    "books": "book",
    "reports": "report",
    "studies": "study",
    "papers": "paper",
    "notes": "note",
    "tips": "tip",
    "ideas": "idea",
    "concepts": "concept",
    "principles": "principle",
    "basics": "basic",
    "fundamentals": "fundamental",
    "standards": "standard",
    "codes": "code",
    "regulations": "regulation",
    "requirements": "requirement",
    "specifications": "specification",
    "ratings": "rating",
    "reviews": "review",
    "comments": "comment",
    "questions": "question",
    "answers": "answer",
    "problems": "problem",
    "issues": "issue",
    "challenges": "challenge",
    "obstacles": "obstacle",
    "barriers": "barrier",
    "limitations": "limitation",
    "drawbacks": "drawback",
    "advantages": "advantage",
    "disadvantages": "disadvantage",

    # Verb forms -> base noun
    "heated": "heater",
    "heating": "heater",
    "cooled": "cooler",
    "cooling": "cooler",
    "collected": "collector",
    "collecting": "collector",
    "powered": "power",
    "powering": "power",
    "stored": "storage",
    "storing": "storage",
    "pumped": "pump",
    "pumping": "pump",
    "filtered": "filter",
    "filtering": "filter",
    "insulated": "insulation",
    "insulating": "insulation",
    "installed": "installation",
    "installing": "installation",
    "designed": "design",
    "designing": "design",
    "built": "build",
    "building": "build",
    "constructed": "construction",
    "constructing": "construction",
    "measured": "measurement",
    "measuring": "measurement",
    "tested": "test",
    "testing": "test",
    "calculated": "calculation",
    "calculating": "calculation",
    "estimated": "estimate",
    "estimating": "estimate",
    "improved": "improvement",
    "improving": "improvement",
    "upgraded": "upgrade",
    "upgrading": "upgrade",
    "modified": "modification",
    "modifying": "modification",
    "converted": "conversion",
    "converting": "conversion",
    "connected": "connection",
    "connecting": "connection",
    "mounted": "mount",
    "mounting": "mount",
    "attached": "attachment",
    "attaching": "attachment",
    "sealed": "seal",
    "sealing": "seal",
    "glazed": "glazing",
    "glazing": "glazing",

    # Synonyms -> preferred term
    "collection": "collector",
    "collections": "collector",
    "home": "house",
    "residential": "house",
    "domestic": "house",
    "photovoltaic": "pv",
    "photovoltaics": "pv",
    "thermal": "heat",
    "electric": "electrical",
    "electricity": "electrical",
    "h2o": "water",
    "aqua": "water",
    "sunlight": "solar",
    "sunshine": "solar",
    "windmill": "wind",
    "hydroelectric": "hydro",
    "hydropower": "hydro",
    "geothermal": "geo",
    "biomass": "bio",
    "biofuel": "bio",
}


def normalize_tag(tag: str) -> str:
    """
    Normalize a tag/term to its canonical form.

    Handles:
    - Lowercase conversion
    - Plural -> singular
    - Verb forms -> base noun
    - Common synonyms -> preferred term
    - Basic -s, -es, -ies plural removal

    Args:
        tag: The tag to normalize

    Returns:
        Normalized tag string
    """
    if not tag:
        return tag

    # Lowercase
    tag = tag.lower().strip()

    # Check explicit mappings first
    if tag in TAG_NORMALIZATIONS:
        return TAG_NORMALIZATIONS[tag]

    # Basic plural handling for unmapped words
    if len(tag) > 3:
        # -ies -> -y (batteries -> battery, but not "series")
        if tag.endswith("ies") and tag not in ("series", "species"):
            return tag[:-3] + "y"
        # -es -> remove (boxes -> box, but not "types")
        if tag.endswith("es") and not tag.endswith("tes") and len(tag) > 4:
            base = tag[:-2]
            if base.endswith(("s", "x", "z", "ch", "sh")):
                return base
        # -s -> remove (simple plural)
        if tag.endswith("s") and not tag.endswith(("ss", "us", "is")):
            return tag[:-1]

    return tag


def normalize_tags(tags: list) -> list:
    """
    Normalize a list of tags and remove duplicates.

    Args:
        tags: List of tag strings

    Returns:
        List of normalized, deduplicated tags
    """
    seen = set()
    result = []
    for tag in tags:
        normalized = normalize_tag(tag)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


# =============================================================================
# SCHEMA: MANIFEST (Source Identity + Distribution)
# =============================================================================

@dataclass
class CategoryStats:
    """Stats for a single category"""
    count: int = 0
    total_chars: int = 0


@dataclass
class SourceManifest:
    """
    Source manifest file (_manifest.json)

    Contains source identity and distribution info.
    """
    # Identity
    source_id: str = ""
    name: str = ""
    description: str = ""

    # License and attribution
    license: str = "Unknown"
    license_verified: bool = False
    license_notes: str = ""  # Required if license is "Custom"
    attribution: str = ""
    base_url: str = ""

    # Human verification flags
    links_verified_offline: bool = False
    links_verified_online: bool = False

    # Language
    language: str = ""  # ISO 639-1 code (e.g., "en", "es")
    language_verified: bool = False

    # Tags for discovery
    tags: List[str] = field(default_factory=list)

    # Stats
    total_docs: int = 0
    total_chars: int = 0
    categories: Dict[str, CategoryStats] = field(default_factory=dict)

    # Distribution info
    version: str = "1.0.0"
    has_backup: bool = False
    has_metadata: bool = False
    has_index: bool = False
    has_vectors: bool = False

    # File sizes (for download estimates)
    backup_size_bytes: int = 0
    metadata_size_bytes: int = 0
    index_size_bytes: int = 0
    vectors_size_bytes: int = 0
    total_size_bytes: int = 0

    # Timestamps
    created_at: str = ""
    last_backup: str = ""
    last_indexed: str = ""

    # Schema version
    schema_version: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "name": self.name,
            "description": self.description,
            "license": self.license,
            "license_verified": self.license_verified,
            "license_notes": self.license_notes,
            "attribution": self.attribution,
            "base_url": self.base_url,
            "links_verified_offline": self.links_verified_offline,
            "links_verified_online": self.links_verified_online,
            "language": self.language,
            "language_verified": self.language_verified,
            "tags": self.tags,
            "total_docs": self.total_docs,
            "total_chars": self.total_chars,
            "categories": {
                k: {"count": v.count, "total_chars": v.total_chars}
                for k, v in self.categories.items()
            },
            "version": self.version,
            "has_backup": self.has_backup,
            "has_metadata": self.has_metadata,
            "has_index": self.has_index,
            "has_vectors": self.has_vectors,
            "backup_size_bytes": self.backup_size_bytes,
            "metadata_size_bytes": self.metadata_size_bytes,
            "index_size_bytes": self.index_size_bytes,
            "vectors_size_bytes": self.vectors_size_bytes,
            "total_size_bytes": self.total_size_bytes,
            "created_at": self.created_at,
            "last_backup": self.last_backup,
            "last_indexed": self.last_indexed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceManifest":
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
            license_verified=data.get("license_verified", False),
            license_notes=data.get("license_notes", ""),
            attribution=data.get("attribution", ""),
            base_url=data.get("base_url", ""),
            links_verified_offline=data.get("links_verified_offline", False),
            links_verified_online=data.get("links_verified_online", False),
            language=data.get("language", ""),
            language_verified=data.get("language_verified", False),
            tags=data.get("tags", []),
            total_docs=data.get("total_docs", 0),
            total_chars=data.get("total_chars", 0),
            categories=categories,
            version=data.get("version", "1.0.0"),
            has_backup=data.get("has_backup", False),
            has_metadata=data.get("has_metadata", False),
            has_index=data.get("has_index", False),
            has_vectors=data.get("has_vectors", False),
            backup_size_bytes=data.get("backup_size_bytes", 0),
            metadata_size_bytes=data.get("metadata_size_bytes", 0),
            index_size_bytes=data.get("index_size_bytes", 0),
            vectors_size_bytes=data.get("vectors_size_bytes", 0),
            total_size_bytes=data.get("total_size_bytes", 0),
            created_at=data.get("created_at", ""),
            last_backup=data.get("last_backup", ""),
            last_indexed=data.get("last_indexed", ""),
            schema_version=data.get("schema_version", 3),
        )


# =============================================================================
# SCHEMA: METADATA (Document Lookup)
# =============================================================================

@dataclass
class DocumentMetadata:
    """
    Single document metadata entry.

    Stored in _metadata.json for quick scanning without loading content.
    """
    doc_id: str
    title: str
    url: str
    content_hash: str
    char_count: int
    categories: List[str] = field(default_factory=list)
    doc_type: str = "article"  # article, guide, research, product
    scraped_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content_hash": self.content_hash,
            "char_count": self.char_count,
            "categories": self.categories,
            "doc_type": self.doc_type,
            "scraped_at": self.scraped_at,
        }


@dataclass
class MetadataFile:
    """
    Document metadata file (_metadata.json)

    Contains metadata for all documents, keyed by doc_id.
    Used for quick scanning, diffing, and sync operations.
    """
    schema_version: int = 3
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
# SCHEMA: INDEX (Full Content)
# =============================================================================

@dataclass
class IndexFile:
    """
    Full content index file (_index.json)

    Contains full document content for display and scanning.
    Separate from vectors to allow content browsing without embeddings.
    """
    schema_version: int = 3
    source_id: str = ""
    document_count: int = 0
    created_at: str = ""

    # Full document content keyed by doc_id
    # Each entry: {title, url, content, categories, doc_type, ...}
    documents: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_id": self.source_id,
            "document_count": self.document_count,
            "created_at": self.created_at,
            "documents": self.documents,
        }


# =============================================================================
# SCHEMA: VECTORS (Embeddings Only)
# =============================================================================

@dataclass
class VectorsFile:
    """
    Vectors file (_vectors.json or _vectors_768.json)

    Contains ONLY embedding vectors, no content duplication.
    Content is in _index.json, metadata is in _metadata.json.

    Two versions may exist:
    - _vectors.json: 1536-dim (OpenAI text-embedding-3-small) for online/global
    - _vectors_768.json: 768-dim (all-mpnet-base-v2) for offline/local
    """
    schema_version: int = 3
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

    @classmethod
    def for_768(cls, source_id: str = "") -> "VectorsFile":
        """Create a VectorsFile configured for 768-dim offline embeddings"""
        return cls(
            source_id=source_id,
            embedding_model="all-mpnet-base-v2",
            dimensions=768
        )

    @classmethod
    def for_1536(cls, source_id: str = "") -> "VectorsFile":
        """Create a VectorsFile configured for 1536-dim online embeddings"""
        return cls(
            source_id=source_id,
            embedding_model="text-embedding-3-small",
            dimensions=1536
        )


# =============================================================================
# SCHEMA: BACKUP MANIFEST
# =============================================================================

@dataclass
class BackupManifest:
    """
    Backup manifest (backup_manifest.json)

    Maps URLs to local filenames for the backup.
    """
    schema_version: int = 3
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
# VALIDATION HELPERS
# =============================================================================

def validate_source_files(source_path: str, source_id: str) -> Dict[str, Any]:
    """
    Validate that a source has the required files.

    Returns:
        Dict with validation results:
        - is_valid: bool
        - has_manifest: bool
        - has_metadata: bool
        - has_index: bool
        - has_vectors: bool (True if any dimension exists)
        - has_vectors_384: bool
        - has_vectors_768: bool
        - has_vectors_1024: bool
        - has_vectors_1536: bool
        - available_dimensions: List[int]
        - has_backup_manifest: bool
        - has_backup: bool (pages/ folder or ZIM file)
        - issues: List[str]
    """
    from pathlib import Path

    path = Path(source_path)
    result = {
        "is_valid": False,
        "has_manifest": False,
        "has_metadata": False,
        "has_index": False,
        "has_vectors": False,
        "has_vectors_384": False,
        "has_vectors_768": False,
        "has_vectors_1024": False,
        "has_vectors_1536": False,
        "available_dimensions": [],
        "has_backup_manifest": False,
        "has_backup": False,
        "issues": [],
    }

    # Check schema files
    result["has_manifest"] = (path / get_manifest_file()).exists()
    result["has_metadata"] = (path / get_metadata_file()).exists()
    result["has_index"] = (path / get_index_file()).exists()

    # Check all vector file dimensions
    result["has_vectors_1536"] = (path / get_vectors_1536_file()).exists()
    result["has_vectors_768"] = (path / get_vectors_768_file()).exists()
    result["has_vectors_384"] = (path / get_vectors_384_file()).exists()
    result["has_vectors_1024"] = (path / get_vectors_1024_file()).exists()

    # Track which dimensions are available
    if result["has_vectors_384"]:
        result["available_dimensions"].append(384)
    if result["has_vectors_768"]:
        result["available_dimensions"].append(768)
    if result["has_vectors_1024"]:
        result["available_dimensions"].append(1024)
    if result["has_vectors_1536"]:
        result["available_dimensions"].append(1536)

    result["has_vectors"] = len(result["available_dimensions"]) > 0

    result["has_backup_manifest"] = (path / get_backup_manifest_file()).exists()

    # Check for backup content
    pages_folder = path / "pages"
    if pages_folder.exists() and any(pages_folder.iterdir()):
        result["has_backup"] = True

    # Check for ZIM file at parent level
    parent = path.parent
    zim_file = parent / f"{source_id}.zim"
    if zim_file.exists():
        result["has_backup"] = True

    # Determine validity - minimum: manifest + (metadata or index) + vectors (any dimension)
    if result["has_manifest"] and (result["has_metadata"] or result["has_index"]) and result["has_vectors"]:
        result["is_valid"] = True
    else:
        if not result["has_manifest"]:
            result["issues"].append(f"Missing {get_manifest_file()}")
        if not result["has_metadata"] and not result["has_index"]:
            result["issues"].append(f"Missing {get_metadata_file()} or {get_index_file()}")
        if not result["has_vectors"]:
            result["issues"].append("Missing vectors file (no embeddings for any dimension)")

    return result


def is_valid_source(source_path: str, source_id: str) -> bool:
    """Check if a source has valid schema files."""
    from pathlib import Path
    path = Path(source_path)
    return (path / get_manifest_file()).exists()


# =============================================================================
# SCHEMA VERSION
# =============================================================================

CURRENT_SCHEMA_VERSION = 3

def get_schema_version(file_data: Dict[str, Any]) -> int:
    """Get schema version from file data"""
    return file_data.get("schema_version", CURRENT_SCHEMA_VERSION)

def needs_migration(file_data: Dict[str, Any]) -> bool:
    """Check if file needs migration to current schema"""
    return get_schema_version(file_data) < CURRENT_SCHEMA_VERSION


