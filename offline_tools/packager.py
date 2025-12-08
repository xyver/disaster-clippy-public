"""
Shared Pack Tools
Common functions for pack management used by both admin and useradmin.

Source of truth is the backup folder - sources are discovered from
_manifest.json files in each source subfolder.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import hashlib
import os

from .schemas import (
    get_manifest_file, get_metadata_file, get_index_file,
    get_vectors_file, get_backup_manifest_file, CURRENT_SCHEMA_VERSION,
    html_filename_to_url
)


# =============================================================================
# CONSTANTS - Hardcoded options for UI dropdowns
# =============================================================================

LICENSE_OPTIONS = [
    "CC-BY",
    "CC-BY-SA",
    "CC-BY-NC",
    "CC-BY-NC-SA",
    "CC0",
    "Public Domain",
    "MIT",
    "GPL",
    "Unknown"
]

PRIORITY_OPTIONS = [
    "LOCAL",
    "GOVERNMENT",
    "GUIDE",
    "RESEARCH",
    "ARTICLE",
    "PRODUCT"
]

SCRAPER_TYPES = [
    "mediawiki",
    "static",
    "pdf",
    "api",
    "substack",
    "kiwix"
]


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent


def get_backup_path() -> Path:
    """Get backup path from local_config (GUI setting) or BACKUP_PATH env var"""
    # Try local_config first (user's GUI setting)
    try:
        from admin.local_config import get_local_config
        config = get_local_config()
        backup_folder = config.get_backup_folder()
        if backup_folder:
            return Path(backup_folder)
    except ImportError:
        pass

    # Fallback to env var
    backup_path = os.getenv("BACKUP_PATH", "")
    if backup_path:
        return Path(backup_path)

    raise ValueError("No backup path configured. Set BACKUP_PATH env var or configure in admin settings.")


# =============================================================================
# BACKUP DETECTION - Unified logic for detecting backup type and status
# =============================================================================

def detect_backup_status(source_id: str, backup_folder: Optional[Path] = None) -> Dict[str, Any]:
    """
    Detect backup type and status for a source.

    This is the single source of truth for backup detection, used by:
    - useradmin/app.py (get_local_sources)
    - useradmin/cloud_upload.py (get_sources_for_upload)
    - sourcepacks/pack_tools.py (get_source_sync_status)

    Args:
        source_id: Source identifier
        backup_folder: Optional backup folder path. If not provided, uses get_backup_path()

    Returns:
        Dict with:
        - has_backup: bool - Whether backup files exist
        - backup_type: str or None - "zim", "html", or "pdf"
        - backup_path: str or None - Path to backup file/folder
        - backup_size_mb: float - Size in MB
    """
    if backup_folder is None:
        backup_folder = get_backup_path()

    result = {
        "has_backup": False,
        "backup_type": None,
        "backup_path": None,
        "backup_size_mb": 0.0
    }

    if not backup_folder:
        return result

    source_folder = Path(backup_folder) / source_id
    if not source_folder.exists():
        return result

    # Check ZIM file inside source folder
    zim_path = source_folder / f"{source_id}.zim"
    if not zim_path.exists():
        # Also check for any .zim file in the source folder
        zim_files = list(source_folder.glob("*.zim"))
        if zim_files:
            zim_path = zim_files[0]

    if zim_path.exists():
        result["has_backup"] = True
        result["backup_type"] = "zim"
        result["backup_path"] = str(zim_path)
        # Calculate FULL folder size for accurate package size
        try:
            total_size = sum(f.stat().st_size for f in source_folder.rglob('*') if f.is_file())
            result["backup_size_mb"] = round(total_size / (1024*1024), 2)
        except Exception:
            # Fallback to just ZIM size
            result["backup_size_mb"] = round(zim_path.stat().st_size / (1024*1024), 2)
        return result

    # Check HTML folder
    html_files = list(source_folder.glob("*.html")) + list(source_folder.glob("**/*.html"))
    pages_folder = source_folder / "pages"
    if html_files or pages_folder.exists():
        result["has_backup"] = True
        result["backup_type"] = "html"
        result["backup_path"] = str(source_folder)
        try:
            total_size = sum(f.stat().st_size for f in source_folder.rglob('*') if f.is_file())
            result["backup_size_mb"] = round(total_size / (1024*1024), 2)
        except Exception:
            pass
        return result

    # Check PDF collection (_collection.json is the definitive marker)
    collection_file = source_folder / "_collection.json"
    if collection_file.exists():
        result["has_backup"] = True
        result["backup_type"] = "pdf"
        result["backup_path"] = str(source_folder)
        try:
            total_size = sum(f.stat().st_size for f in source_folder.rglob('*') if f.is_file())
            result["backup_size_mb"] = round(total_size / (1024*1024), 2)
        except Exception:
            pass
        return result

    return result


# =============================================================================
# SUBMISSION VALIDATION - Unified logic for validating submitted files
# =============================================================================

def validate_submission_files(files: List[Dict[str, Any]], source_id: str = None) -> Dict[str, Any]:
    """
    Validate a list of submitted files against the schema requirements.

    This is the single source of truth for submission validation, used by:
    - admin/app.py (approve_submission)
    - Any other submission validation needs

    Args:
        files: List of file dicts with 'key' (path) and optionally 'size_mb'
               Files can be R2 keys like "submissions/timestamp_source/file.json"
               or local paths
        source_id: Optional source ID (extracted from files if not provided)

    Returns:
        Dict with:
        - is_valid: bool - Whether submission meets all requirements
        - missing: list - List of missing required items
        - has_backup: bool
        - has_manifest: bool
        - has_metadata: bool
        - has_vectors: bool
        - has_license: bool
        - backup_type: str or None - "zim", "html", "pdf", or "zip"
        - source_id: str - Detected or provided source_id
        - files_found: dict - Which specific files were found
    """
    result = {
        "is_valid": False,
        "missing": [],
        "has_backup": False,
        "has_manifest": False,
        "has_metadata": False,
        "has_vectors": False,
        "has_license": False,
        "backup_type": None,
        "source_id": source_id,
        "files_found": {
            "backup": None,
            "manifest": None,
            "metadata": None,
            "index": None,
            "vectors": None,
            "backup_manifest": None,
        }
    }

    # Schema file names
    manifest_file = get_manifest_file()        # _manifest.json
    metadata_file = get_metadata_file()        # _metadata.json
    index_file = get_index_file()              # _index.json
    vectors_file = get_vectors_file()          # _vectors.json
    backup_manifest_file = get_backup_manifest_file()  # backup_manifest.json

    # Extract filenames and categorize
    for f in files:
        key = f.get("key", f) if isinstance(f, dict) else str(f)
        filename = key.split("/")[-1] if "/" in key else key

        # Backup files (.zim or .zip)
        if filename.endswith('.zim'):
            result["has_backup"] = True
            result["backup_type"] = "zim"
            result["files_found"]["backup"] = filename
        elif filename.endswith('.zip'):
            result["has_backup"] = True
            if '-html.zip' in filename:
                result["backup_type"] = "html"
            elif '-pdf.zip' in filename:
                result["backup_type"] = "pdf"
            else:
                result["backup_type"] = "zip"
            result["files_found"]["backup"] = filename

        # _manifest.json
        elif filename == manifest_file:
            result["has_manifest"] = True
            result["files_found"]["manifest"] = filename

        # _metadata.json
        elif filename == metadata_file:
            result["has_metadata"] = True
            result["files_found"]["metadata"] = filename

        # _index.json
        elif filename == index_file:
            result["files_found"]["index"] = filename

        # _vectors.json
        elif filename == vectors_file:
            result["has_vectors"] = True
            result["files_found"]["vectors"] = filename

        # backup_manifest.json
        elif filename == backup_manifest_file:
            result["files_found"]["backup_manifest"] = filename

    # Build missing list
    if not result["has_backup"]:
        result["missing"].append("backup file (.zim or .zip)")
    if not result["has_manifest"]:
        result["missing"].append(f"manifest file ({manifest_file})")
    if not result["has_metadata"]:
        result["missing"].append(f"metadata file ({metadata_file})")
    if not result["has_vectors"]:
        result["missing"].append(f"vectors file ({vectors_file})")

    # Determine overall validity
    result["is_valid"] = len(result["missing"]) == 0

    return result


def validate_submission_content(
    source_config: Dict[str, Any],
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Validate the content of submission files (after downloading/loading them).

    Args:
        source_config: Loaded source config (from _manifest.json or submission_manifest.source_config)
        metadata: Optional loaded metadata

    Returns:
        Dict with:
        - is_valid: bool
        - missing: list
        - has_license: bool
        - license_verified: bool
        - license_value: str
    """
    result = {
        "is_valid": True,
        "missing": [],
        "has_license": False,
        "license_verified": False,
        "license_value": ""
    }

    # Check license
    license_val = source_config.get("license", "")
    result["license_value"] = license_val
    result["license_verified"] = source_config.get("license_verified", False)
    result["has_license"] = license_val and license_val.lower() not in ["unknown", ""]

    if not result["has_license"]:
        result["missing"].append("license in source config")
        result["is_valid"] = False

    return result


# =============================================================================
# UNIFIED SOURCE MANAGEMENT
# =============================================================================

def load_sources(prefer_config: bool = False) -> Dict[str, Any]:
    """
    Get source configuration constants.

    NOTE: Sources are now discovered from the backup folder, not from
    sources.json. This function returns the constants for UI dropdowns.
    The 'sources' dict is empty - actual sources are discovered at runtime.

    Returns:
        Configuration dict with empty sources and option constants
    """
    return {
        "sources": {},
        "license_options": LICENSE_OPTIONS,
        "priority_options": PRIORITY_OPTIONS,
        "scraper_types": SCRAPER_TYPES
    }


def load_master_metadata() -> Dict[str, Any]:
    """
    Load the master metadata index from backup folder.

    Returns:
        Master metadata dict with source counts and totals
    """
    master_file = get_backup_path() / "_master.json"

    if master_file.exists():
        try:
            with open(master_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load master metadata: {e}")

    return {"total_documents": 0, "sources": {}}


def get_source_sync_status(source_id: str) -> Dict[str, Any]:
    """
    Get sync status between indexed (vector DB) and backed up (HTML) content.

    Args:
        source_id: Source identifier

    Returns:
        Dict with indexed/backed up counts and what needs attention
    """
    backup_path = get_backup_path()

    result = {
        "indexed_count": 0,
        "indexed_urls": [],
        "backed_up_count": 0,
        "backed_up_urls": [],
        "needs_backup": [],
        "needs_indexing": [],
        "synced": []
    }

    source_folder = backup_path / source_id

    # Get indexed URLs from metadata
    metadata_file = source_folder / get_metadata_file()
    if metadata_file.exists():
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                result["indexed_urls"] = [
                    doc.get("url") for doc in meta.get("documents", {}).values()
                    if doc.get("url")
                ]
                result["indexed_count"] = len(result["indexed_urls"])
        except Exception:
            pass

    # Get backed up URLs from backup manifest
    manifest_file = source_folder / get_backup_manifest_file()
    if manifest_file.exists():
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                result["backed_up_urls"] = manifest.get("urls", [])
                result["backed_up_count"] = len(result["backed_up_urls"])
        except Exception:
            pass

    # Calculate what needs attention
    indexed_set = set(result["indexed_urls"])
    backed_up_set = set(result["backed_up_urls"])

    result["needs_backup"] = list(indexed_set - backed_up_set)
    result["needs_indexing"] = list(backed_up_set - indexed_set)
    result["synced"] = list(indexed_set & backed_up_set)

    return result


def generate_metadata_from_html(
    backup_path: str,
    source_id: str,
    save: bool = True
) -> Dict[str, Any]:
    """
    Scan an HTML backup folder and generate metadata.

    Args:
        backup_path: Path to the HTML backup folder (typically the pages/ subfolder)
        source_id: Source identifier
        save: Whether to save the metadata file

    Returns:
        Metadata dict with document info
    """
    from bs4 import BeautifulSoup

    html_path = Path(backup_path)
    if not html_path.exists() or not html_path.is_dir():
        raise ValueError(f"Backup path does not exist: {backup_path}")

    # Check for backup_manifest.json in parent folder (source folder)
    # The manifest has URL-to-filename mapping
    source_folder = html_path.parent if html_path.name == "pages" else html_path
    manifest_path = source_folder / get_backup_manifest_file()
    url_mapping = {}  # filename -> url path

    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            # Build reverse lookup: filename -> url_path
            for url_path, page_info in manifest_data.get("pages", {}).items():
                filename = page_info.get("filename", "")
                if filename:
                    url_mapping[filename] = url_path
            print(f"Loaded {len(url_mapping)} URL mappings from backup manifest")
        except Exception as e:
            print(f"Warning: Could not load backup manifest: {e}")

    documents = {}
    total_chars = 0

    # Scan HTML files
    html_files = list(html_path.rglob('*.html')) + list(html_path.rglob('*.htm'))

    for html_file in html_files:
        try:
            with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            soup = BeautifulSoup(content, 'html.parser')

            # Extract title
            title = "Untitled"
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            elif soup.h1:
                title = soup.h1.get_text(strip=True)

            # Get text content
            text = soup.get_text(separator=' ', strip=True)
            char_count = len(text)
            total_chars += char_count

            # Generate content hash
            content_hash = hashlib.md5(text.encode()).hexdigest()[:12]

            # Get filename relative to html_path
            rel_path = html_file.relative_to(html_path)
            filename = str(rel_path).replace('\\', '/')

            # Look up proper URL from backup manifest, or construct from filename
            if filename in url_mapping:
                url = url_mapping[filename]
            else:
                # Fallback: use centralized filename conversion
                url = html_filename_to_url(filename)

            # Build local_url for offline serving
            local_url = f"/backup/{source_id}/{filename}"

            # Use same ID format as indexer: MD5 hash of source_id:url
            doc_id = hashlib.md5(f"{source_id}:{url}".encode()).hexdigest()
            documents[doc_id] = {
                "title": title,
                "url": url,
                "local_url": local_url,
                "content_hash": content_hash,
                "scraped_at": datetime.now().isoformat(),
                "char_count": char_count,
                "categories": [],
                "doc_type": "article"
            }

        except Exception as e:
            print(f"Warning: Failed to process {html_file}: {e}")
            continue

    if not documents:
        raise ValueError(f"No HTML files found in {backup_path}")

    # Create metadata structure
    metadata = {
        "schema_version": 3,
        "source_id": source_id,
        "source_type": "html",
        "last_updated": datetime.now().isoformat(),
        "total_documents": len(documents),
        "document_count": len(documents),
        "total_chars": total_chars,
        "documents": documents
    }

    if save:
        save_metadata(source_id, metadata)

    return metadata


def save_metadata(source_id: str, metadata: Dict[str, Any]) -> Path:
    """Save metadata to the backup folder location (v3 format)"""
    backup_dir = get_backup_path()
    backup_dir.mkdir(parents=True, exist_ok=True)

    # Source-specific metadata goes in source subfolder
    source_dir = backup_dir / source_id
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = source_dir / get_metadata_file()
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # Update master file
    update_master_metadata(source_id, metadata)

    return metadata_file


def update_master_metadata(source_id: str, metadata: Dict[str, Any]) -> None:
    """Update the master metadata index in backup folder"""
    backup_dir = get_backup_path()
    backup_dir.mkdir(parents=True, exist_ok=True)
    master_file = backup_dir / "_master.json"

    if master_file.exists():
        with open(master_file, 'r', encoding='utf-8') as f:
            master = json.load(f)
    else:
        master = {"version": 2, "sources": {}}

    # Get tags from manifest if available
    source_dir = backup_dir / source_id
    tags = []
    manifest_path = source_dir / get_manifest_file()
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            tags = manifest_data.get("tags", [])
        except Exception:
            pass

    master["sources"][source_id] = {
        "count": metadata.get("total_documents", metadata.get("document_count", 0)),
        "chars": metadata.get("total_chars", 0),
        "last_sync": datetime.now().isoformat(),
        "file": f"{source_id}.json",
        "topics": tags
    }

    # Recalculate totals (use .get() for backwards compatibility with old entries)
    master["total_documents"] = sum(s.get("count", 0) for s in master["sources"].values())
    master["total_chars"] = sum(s.get("chars", 0) for s in master["sources"].values())
    master["last_updated"] = datetime.now().isoformat()

    with open(master_file, 'w', encoding='utf-8') as f:
        json.dump(master, f, indent=2)


def sync_master_metadata() -> Dict[str, Any]:
    """
    Sync _master.json by scanning the backup folder.

    This rebuilds the master index based on what's actually on disk:
    - Adds new sources that exist but aren't in master
    - Removes sources that are in master but no longer exist
    - Updates counts and tags from source files

    Call this:
    - On app startup
    - After source creation/deletion
    - After indexing completes
    - When tags are updated

    Returns:
        Dict with sync results (added, removed, updated counts)
    """
    backup_dir = get_backup_path()
    if not backup_dir.exists():
        return {"added": 0, "removed": 0, "updated": 0, "total": 0}

    master_file = backup_dir / "_master.json"

    # Load existing master or create new
    if master_file.exists():
        try:
            with open(master_file, 'r', encoding='utf-8') as f:
                master = json.load(f)
        except Exception:
            master = {"version": 2, "sources": {}}
    else:
        master = {"version": 2, "sources": {}}

    old_sources = set(master.get("sources", {}).keys())
    new_sources = {}

    added = 0
    removed = 0
    updated = 0

    # Scan all directories in backup folder
    for item in backup_dir.iterdir():
        if not item.is_dir():
            continue
        if item.name.startswith("_") or item.name.startswith("."):
            continue  # Skip special dirs like _master.json folder (if any)

        source_id = item.name
        source_dir = item

        # Check if this is a valid source (has manifest or metadata)
        manifest_path = source_dir / get_manifest_file()
        metadata_path = source_dir / get_metadata_file()
        vectors_path = source_dir / get_vectors_file()

        # Must have at least a manifest to be considered a source
        if not manifest_path.exists():
            # Check for legacy files or ZIM file as fallback
            zim_files = list(source_dir.glob("*.zim"))
            if not zim_files and not metadata_path.exists():
                continue  # Not a valid source

        # Get document count
        doc_count = 0
        total_chars = 0

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                doc_count = meta.get("document_count", len(meta.get("documents", {})))
                total_chars = meta.get("total_chars", 0)
            except Exception:
                pass

        # Fallback: check vectors file for count
        if doc_count == 0 and vectors_path.exists():
            try:
                with open(vectors_path, 'r', encoding='utf-8') as f:
                    vec = json.load(f)
                doc_count = vec.get("document_count", len(vec.get("vectors", {})))
            except Exception:
                pass

        # Get tags from manifest
        tags = []
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
                tags = manifest_data.get("tags", [])
            except Exception:
                pass

        # Build source entry
        existing = master.get("sources", {}).get(source_id, {})
        new_sources[source_id] = {
            "count": doc_count,
            "chars": total_chars,
            "last_sync": datetime.now().isoformat(),
            "file": f"{source_id}.json",
            "topics": tags
        }

        # Track changes
        if source_id not in old_sources:
            added += 1
        elif (existing.get("count", 0) != doc_count or
              existing.get("topics", []) != tags):
            updated += 1

    # Count removed sources
    current_sources = set(new_sources.keys())
    removed = len(old_sources - current_sources)

    # Build new master
    master["version"] = 2
    master["sources"] = new_sources
    master["total_documents"] = sum(s.get("count", 0) for s in new_sources.values())
    master["total_chars"] = sum(s.get("chars", 0) for s in new_sources.values())
    master["last_updated"] = datetime.now().isoformat()

    # Save
    with open(master_file, 'w', encoding='utf-8') as f:
        json.dump(master, f, indent=2)

    return {
        "added": added,
        "removed": removed,
        "updated": updated,
        "total": len(new_sources),
        "total_documents": master["total_documents"]
    }


def remove_from_master(source_id: str) -> bool:
    """
    Remove a source from _master.json.

    Call this when deleting a source.

    Returns:
        True if source was removed, False if not found
    """
    backup_dir = get_backup_path()
    master_file = backup_dir / "_master.json"

    if not master_file.exists():
        return False

    try:
        with open(master_file, 'r', encoding='utf-8') as f:
            master = json.load(f)

        if source_id not in master.get("sources", {}):
            return False

        del master["sources"][source_id]

        # Recalculate totals
        master["total_documents"] = sum(s.get("count", 0) for s in master["sources"].values())
        master["total_chars"] = sum(s.get("chars", 0) for s in master["sources"].values())
        master["last_updated"] = datetime.now().isoformat()

        with open(master_file, 'w', encoding='utf-8') as f:
            json.dump(master, f, indent=2)

        return True
    except Exception as e:
        print(f"Error removing {source_id} from master: {e}")
        return False


def load_metadata(source_id: str) -> Optional[Dict[str, Any]]:
    """Load metadata for a source from backup folder"""
    backup_dir = get_backup_path()
    source_dir = backup_dir / source_id

    metadata_path = source_dir / get_metadata_file()
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    return None


def index_html_to_chromadb(
    backup_path: str,
    source_id: str
) -> Dict[str, Any]:
    """
    Index HTML backup content to ChromaDB.

    Args:
        backup_path: Path to HTML backup folder
        source_id: Source identifier

    Returns:
        Dict with indexed count and status
    """
    from offline_tools.indexer import HTMLBackupIndexer

    indexer = HTMLBackupIndexer(
        backup_path=backup_path,
        source_id=source_id
    )

    return indexer.index()


def index_zim_to_chromadb(
    zim_path: str,
    source_id: str
) -> Dict[str, Any]:
    """
    Index ZIM file content to ChromaDB.

    Args:
        zim_path: Path to ZIM file
        source_id: Source identifier

    Returns:
        Dict with indexed count and status
    """
    from offline_tools.indexer import ZIMIndexer

    indexer = ZIMIndexer(
        zim_path=zim_path,
        source_id=source_id
    )

    return indexer.index()


def index_pdf_to_chromadb(
    pdf_path: str,
    source_id: str,
    skip_existing: bool = True
) -> Dict[str, Any]:
    """
    Index PDF file or folder content to ChromaDB.

    Args:
        pdf_path: Path to PDF file or folder containing PDFs
        source_id: Source identifier

    Returns:
        Dict with indexed count and status
    """
    from offline_tools.indexer import PDFIndexer

    indexer = PDFIndexer(
        source_path=pdf_path,
        source_id=source_id
    )

    return indexer.index(skip_existing=skip_existing)


def export_chromadb_index(source_id: str) -> Dict[str, Any]:
    """
    Export ChromaDB embeddings for a source to a portable format.

    Args:
        source_id: Source identifier

    Returns:
        Dict with embeddings and metadata for all documents
    """
    from offline_tools.vectordb.store import LocalVectorStore

    store = LocalVectorStore()

    # Get all documents for this source
    results = store.collection.get(
        where={"source": source_id},
        include=["embeddings", "metadatas", "documents"]
    )

    if not results.get("ids"):
        return {"source_id": source_id, "count": 0, "embeddings": []}

    # Build export structure
    export_data = {
        "version": 1,
        "source_id": source_id,
        "exported_at": datetime.now().isoformat(),
        "count": len(results["ids"]),
        "embedding_model": "default",  # Could be enhanced to detect model
        "documents": []
    }

    for i, doc_id in enumerate(results["ids"]):
        doc_export = {
            "id": doc_id,
            "embedding": results["embeddings"][i] if results.get("embeddings") else None,
            "metadata": results["metadatas"][i] if results.get("metadatas") else {},
            "content": results["documents"][i] if results.get("documents") else None
        }
        export_data["documents"].append(doc_export)

    return export_data


def save_index_export(source_id: str, export_data: Dict[str, Any]) -> Path:
    """Save exported index to file"""
    # Save in data/indexes folder
    index_dir = get_project_root() / "data" / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    index_file = index_dir / f"{source_id}_index.json"
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f)

    return index_file


def get_source_completeness(source_id: str, backup_folder: str = None) -> Dict[str, Any]:
    """
    Check completeness of a source - what's present and what's missing.

    Args:
        source_id: Source identifier
        backup_folder: Path to backup folder (optional)

    Returns:
        Dict with completeness status
    """
    status = {
        "source_id": source_id,
        "has_config": False,
        "has_metadata": False,
        "has_backup": False,
        "has_index": False,
        "backup_type": None,
        "backup_size_mb": 0,
        "document_count": 0,
        "indexed_count": 0,
        "license_verified": False,
        "is_complete": False,
        "missing": [],
        "ready_for_upload": False
    }

    root = get_project_root()
    backup_folder = get_backup_path()

    # Check source config from _manifest.json
    if backup_folder:
        source_file = Path(backup_folder) / source_id / get_manifest_file()
        if source_file.exists():
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    source_config = json.load(f)
                status["has_config"] = True
                status["license_verified"] = source_config.get("license_verified", False)
                if not status["license_verified"]:
                    status["missing"].append("verified license")
            except Exception:
                status["missing"].append("source config (file unreadable)")
        else:
            status["missing"].append("source config file")
    else:
        status["missing"].append("backup folder not configured")

    # Check metadata
    metadata = load_metadata(source_id)
    if metadata:
        status["has_metadata"] = True
        status["document_count"] = metadata.get("total_documents", metadata.get("document_count", 0))
    else:
        status["missing"].append("metadata")

    # Check backup using unified detection
    backup_status = detect_backup_status(source_id, backup_folder)
    status["has_backup"] = backup_status["has_backup"]
    status["backup_type"] = backup_status["backup_type"]
    status["backup_size_mb"] = backup_status["backup_size_mb"]

    if not status["has_backup"]:
        status["missing"].append("backup file")

    # Check ChromaDB index
    try:
        from offline_tools.vectordb.store import LocalVectorStore
        store = LocalVectorStore()
        results = store.collection.get(
            where={"source": source_id},
            include=[]
        )
        status["has_index"] = len(results.get("ids", [])) > 0
        status["indexed_count"] = len(results.get("ids", []))
    except Exception:
        pass

    # Determine completeness
    status["is_complete"] = (
        status["has_config"] and
        status["has_metadata"] and
        status["has_backup"] and
        status["license_verified"]
    )

    status["ready_for_upload"] = status["is_complete"]

    return status


def create_pack_manifest(
    source_id: str,
    source_config: Dict[str, Any],
    metadata: Dict[str, Any],
    backup_info: Dict[str, Any],
    approval_info: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a complete pack manifest for distribution (v3 format).

    Args:
        source_id: Source identifier
        source_config: Source configuration from _manifest.json
        metadata: Document metadata
        backup_info: Info about the backup file (type, size)
        approval_info: Optional approval information

    Returns:
        Complete manifest dict
    """
    manifest = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "source_id": source_id,
        "created_at": datetime.now().isoformat(),

        # Source info (flattened for quick access)
        "name": source_config.get("name", source_id),
        "description": source_config.get("description", ""),
        "base_url": source_config.get("base_url", source_config.get("url", "")),
        "license": source_config.get("license", "Unknown"),
        "license_verified": source_config.get("license_verified", False),
        "tags": source_config.get("tags", []),

        # Stats
        "total_docs": metadata.get("total_documents", metadata.get("document_count", 0)),
        "total_chars": metadata.get("total_chars", 0),

        # Backup info
        "source_type": backup_info.get("type"),
        "has_backup": True,
        "has_metadata": True,
        "has_index": True,
        "has_vectors": True,
        "backup_size_mb": backup_info.get("size_mb", 0),

        # Schema file names
        "files": {
            "manifest": get_manifest_file(),      # _manifest.json
            "metadata": get_metadata_file(),      # _metadata.json
            "index": get_index_file(),            # _index.json
            "vectors": get_vectors_file(),        # _vectors.json
            "backup": f"{source_id}.zim" if backup_info.get("type") == "zim" else f"{source_id}-html.zip",
        }
    }

    if approval_info:
        manifest["approval"] = {
            "status": "approved",
            "approved_at": datetime.now().isoformat(),
            "approved_by": approval_info.get("approved_by", "admin"),
            "message": approval_info.get("message", "")
        }

    return manifest
