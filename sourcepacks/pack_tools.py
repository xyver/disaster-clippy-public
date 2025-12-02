"""
Shared Pack Tools
Common functions for pack management used by both admin and useradmin.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import hashlib


def get_project_root() -> Path:
    """Get project root directory"""
    return Path(__file__).parent.parent


def generate_metadata_from_html(
    backup_path: str,
    source_id: str,
    save: bool = True
) -> Dict[str, Any]:
    """
    Scan an HTML backup folder and generate metadata.

    Args:
        backup_path: Path to the HTML backup folder
        source_id: Source identifier
        save: Whether to save the metadata file

    Returns:
        Metadata dict with document info
    """
    from bs4 import BeautifulSoup

    html_path = Path(backup_path)
    if not html_path.exists() or not html_path.is_dir():
        raise ValueError(f"Backup path does not exist: {backup_path}")

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

            # Relative path as URL
            rel_path = html_file.relative_to(html_path)
            url = str(rel_path).replace('\\', '/')

            doc_id = f"{source_id}_{content_hash}"
            documents[doc_id] = {
                "title": title,
                "url": url,
                "content_hash": content_hash,
                "scraped_at": datetime.now().isoformat(),
                "char_count": char_count
            }

        except Exception as e:
            print(f"Warning: Failed to process {html_file}: {e}")
            continue

    if not documents:
        raise ValueError(f"No HTML files found in {backup_path}")

    # Create metadata structure
    metadata = {
        "version": 2,
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
    """Save metadata to the standard location"""
    metadata_dir = get_project_root() / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = metadata_dir / f"{source_id}.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    # Update master file
    update_master_metadata(source_id, metadata)

    return metadata_file


def update_master_metadata(source_id: str, metadata: Dict[str, Any]) -> None:
    """Update the master metadata index"""
    metadata_dir = get_project_root() / "data" / "metadata"
    master_file = metadata_dir / "_master.json"

    if master_file.exists():
        with open(master_file, 'r', encoding='utf-8') as f:
            master = json.load(f)
    else:
        master = {"version": 2, "sources": {}}

    master["sources"][source_id] = {
        "count": metadata.get("total_documents", metadata.get("document_count", 0)),
        "chars": metadata.get("total_chars", 0),
        "last_sync": datetime.now().isoformat(),
        "file": f"{source_id}.json",
        "topics": []
    }

    # Recalculate totals
    master["total_documents"] = sum(s["count"] for s in master["sources"].values())
    master["total_chars"] = sum(s["chars"] for s in master["sources"].values())
    master["last_updated"] = datetime.now().isoformat()

    with open(master_file, 'w', encoding='utf-8') as f:
        json.dump(master, f, indent=2)


def load_metadata(source_id: str) -> Optional[Dict[str, Any]]:
    """Load metadata for a source"""
    metadata_path = get_project_root() / "data" / "metadata" / f"{source_id}.json"
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

    return indexer.index_all()


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

    return indexer.index_all()


def export_chromadb_index(source_id: str) -> Dict[str, Any]:
    """
    Export ChromaDB embeddings for a source to a portable format.

    Args:
        source_id: Source identifier

    Returns:
        Dict with embeddings and metadata for all documents
    """
    from vectordb.store import LocalVectorStore

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

    # Check sources.json config
    sources_path = root / "config" / "sources.json"
    if sources_path.exists():
        with open(sources_path, 'r', encoding='utf-8') as f:
            sources_data = json.load(f)
        if source_id in sources_data.get("sources", {}):
            status["has_config"] = True
            source_config = sources_data["sources"][source_id]
            status["license_verified"] = source_config.get("license_verified", False)
            if not status["license_verified"]:
                status["missing"].append("verified license")
        else:
            status["missing"].append("config in sources.json")
    else:
        status["missing"].append("sources.json file")

    # Check metadata
    metadata = load_metadata(source_id)
    if metadata:
        status["has_metadata"] = True
        status["document_count"] = metadata.get("total_documents", metadata.get("document_count", 0))
    else:
        status["missing"].append("metadata")

    # Check backup
    if backup_folder:
        html_path = Path(backup_folder) / source_id
        zim_path = Path(backup_folder) / f"{source_id}.zim"

        if zim_path.exists():
            status["has_backup"] = True
            status["backup_type"] = "zim"
            status["backup_size_mb"] = round(zim_path.stat().st_size / (1024*1024), 2)
        elif html_path.exists() and html_path.is_dir():
            html_files = list(html_path.rglob('*.html'))
            if html_files:
                status["has_backup"] = True
                status["backup_type"] = "html"
                try:
                    total_size = sum(f.stat().st_size for f in html_path.rglob('*') if f.is_file())
                    status["backup_size_mb"] = round(total_size / (1024*1024), 2)
                except Exception:
                    pass

        if not status["has_backup"]:
            status["missing"].append("backup file")

    # Check ChromaDB index
    try:
        from vectordb.store import LocalVectorStore
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
    Create a complete pack manifest for distribution.

    Args:
        source_id: Source identifier
        source_config: Source configuration from sources.json
        metadata: Document metadata
        backup_info: Info about the backup file (type, size)
        approval_info: Optional approval information

    Returns:
        Complete manifest dict
    """
    manifest = {
        "version": 1,
        "source_id": source_id,
        "created_at": datetime.now().isoformat(),

        # Source info
        "name": source_config.get("name", source_id),
        "description": source_config.get("description", ""),
        "url": source_config.get("url", ""),
        "license": source_config.get("license", "Unknown"),
        "license_verified": source_config.get("license_verified", False),

        # Pack info
        "pack_info": {
            "document_count": metadata.get("total_documents", metadata.get("document_count", 0)),
            "total_chars": metadata.get("total_chars", 0),
            "backup_type": backup_info.get("type"),
            "backup_size_mb": backup_info.get("size_mb", 0)
        },

        # Files in this pack
        "files": {
            "manifest": "manifest.json",
            "metadata": "metadata.json",
            "backup": f"{source_id}.zim" if backup_info.get("type") == "zim" else f"{source_id}-html.zip",
            "index": "index.json"  # Optional
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
