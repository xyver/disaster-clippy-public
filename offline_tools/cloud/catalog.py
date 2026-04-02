"""
Public catalog generation.

Builds published/catalog.json from local active source metadata and uploads to R2.
The catalog is the data source for the /packs page on the public site.

Source selection rules:
- Local _master.json is the authoritative list of active sources.
- R2 backups/ is used as a publish check: the source must actually exist in R2.
- Local _manifest.json provides the public-facing metadata for each source.

The catalog object must be publicly readable. Public access is a bucket-level
setting in Cloudflare R2 -- configure the bucket or use a public URL policy
separate from this code.
"""

import json
import logging
import os
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

CATALOG_KEY = "published/catalog.json"


def _looks_like_backup_url(url: str) -> bool:
    """Heuristic check for bucket/backup-hosted URLs."""
    if not url:
        return False
    lowered = url.lower()
    return (
        "/backups/" in lowered
        or ".r2.dev/" in lowered
        or ".r2.cloudflarestorage.com/" in lowered
    )


def _load_first_document_url(source_dir: Path) -> str:
    """Best-effort extraction of the first indexed document URL for a source."""
    index_path = source_dir / "_index.json"
    if not index_path.exists():
        return ""
    try:
        with open(index_path, "r", encoding="utf-8-sig") as f:
            index_data = json.load(f)
        documents = index_data.get("documents", {})
        if isinstance(documents, dict):
            for doc in documents.values():
                if isinstance(doc, dict) and doc.get("url"):
                    return str(doc["url"])
    except Exception as e:
        logger.warning("Could not read first document URL for %s: %s", source_dir.name, e)
    return ""


def _derive_live_url(source_id: str, source_dir: Path, manifest: Dict[str, Any]) -> str:
    """Pick the human-facing live/original URL for this source."""
    for key in ("live_url", "source_url", "original_url", "homepage_url", "canonical_url", "public_url"):
        value = str(manifest.get(key, "") or "").strip()
        if value:
            return value

    zim_meta = manifest.get("zim_metadata", {})
    if isinstance(zim_meta, dict):
        source_url = str(zim_meta.get("source_url", "") or "").strip()
        if source_url:
            return source_url

    base_url = str(manifest.get("base_url", "") or "").strip()
    if base_url and not _looks_like_backup_url(base_url):
        return base_url

    source_type = str(manifest.get("source_type", "") or "").strip().lower()
    if source_type == "pdf":
        return _load_first_document_url(source_dir) or base_url

    return base_url


def _derive_backup_url(source_id: str, manifest: Dict[str, Any]) -> str:
    """Pick the bucket/backup location for this source when one is available."""
    for key in ("backup_url", "emergency_backup_url", "archive_url"):
        value = str(manifest.get(key, "") or "").strip()
        if value:
            return value

    base_url = str(manifest.get("base_url", "") or "").strip()
    if _looks_like_backup_url(base_url):
        return base_url

    r2_public_url = str(os.getenv("R2_PUBLIC_URL", "") or "").strip().rstrip("/")
    if r2_public_url:
        return f"{r2_public_url}/backups/{source_id}"

    return ""


def _read_local_master(backup_path: Path) -> Dict[str, Any]:
    """Read local _master.json and return its parsed dict, or an empty shell."""
    master_path = backup_path / "_master.json"
    if not master_path.exists():
        logger.warning("Local _master.json not found at %s", master_path)
        return {"sources": {}}

    try:
        with open(master_path, "r", encoding="utf-8") as f:
            master = json.load(f)
    except Exception as e:
        logger.error("Could not read local _master.json at %s: %s", master_path, e)
        return {"sources": {}}

    if not isinstance(master, dict):
        logger.warning("Local _master.json has unexpected type: %s", type(master).__name__)
        return {"sources": {}}

    return master


def _read_local_manifest(source_dir: Path) -> Optional[Dict[str, Any]]:
    """Read _manifest.json for a source. Returns None if missing or unreadable."""
    manifest_path = source_dir / "_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        # Many existing manifests were written with a UTF-8 BOM.
        with open(manifest_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read manifest for %s: %s", source_dir.name, e)
        return None


def _build_catalog_entry(source_id: str, source_dir: Path, manifest: Dict[str, Any], master_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build a single catalog entry from manifest + _master metadata.

    Returns None if the source is not catalog-ready (missing description).
    """
    description = manifest.get("description", "").strip()
    if not description:
        return None

    live_url = _derive_live_url(source_id, source_dir, manifest)
    backup_url = _derive_backup_url(source_id, manifest)

    return {
        "source_id": source_id,
        "name": manifest.get("name", source_id),
        "description": description,
        "license": manifest.get("license", "Unknown"),
        "license_verified": manifest.get("license_verified", False),
        "tags": manifest.get("tags", []),
        "topics": master_entry.get("topics", []),
        "base_url": manifest.get("base_url", ""),
        "live_url": live_url,
        "backup_url": backup_url,
        "source_type": manifest.get("source_type", ""),
        "language": manifest.get("language", "en"),
        "doc_count": master_entry.get("count", manifest.get("total_docs", 0)),
        "size_bytes": master_entry.get("size_bytes", manifest.get("total_size_bytes", 0)),
        "last_updated": master_entry.get("last_sync", manifest.get("created_at", "")),
    }


def generate_public_catalog(source_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Build published/catalog.json from local manifests and upload to R2.

    Discovery:
      - If source_ids is None, uses local _master.json as the active source list.
      - If source_ids is given explicitly, uses that list (for testing or manual runs).
      - In both cases, sources must also exist in R2 backups/ to be included.

    Inclusion criteria (per source):
      - Source exists in local _master.json (unless source_ids explicitly provided)
      - Source exists in R2 backups/{source_id}/
      - Local _manifest.json exists under BACKUP_PATH/{source_id}/
      - description field is non-empty

    Args:
        source_ids: Optional explicit list of source IDs to include. If None,
                    discovers from R2 published/ folder.

    Returns:
        Dict with keys:
          included  - list of source_ids in the catalog
          skipped   - list of "{source_id}: reason" strings
          errors    - list of error strings
          uploaded  - bool, True if catalog was successfully uploaded to R2
          catalog_key - the R2 key the catalog was written to
    """
    from offline_tools.packager import get_backup_path
    from offline_tools.cloud.r2 import get_backups_storage

    backup_path = get_backup_path()
    storage = get_backups_storage()
    master = _read_local_master(backup_path)
    master_sources = master.get("sources", {})

    result: Dict[str, Any] = {
        "included": [],
        "skipped": [],
        "errors": [],
        "uploaded": False,
        "catalog_key": CATALOG_KEY,
    }

    if not storage.is_configured():
        result["errors"].append("R2 not configured -- cannot discover published sources")
        logger.error("R2 not configured, cannot discover published sources")
        return result

    files = storage.list_files("backups/")
    published_source_ids: set[str] = set()
    for f in files:
        parts = f["key"].split("/")
        # Keys look like "backups/{source_id}/..."
        if len(parts) >= 2 and parts[1] and not parts[1].startswith("_"):
            published_source_ids.add(parts[1])

    logger.info("Discovered %d published sources from R2", len(published_source_ids))

    # Discover active source IDs from local _master.json if not provided
    if source_ids is None:
        source_ids = list(master_sources.keys())
        logger.info("Loaded %d active sources from local _master.json", len(source_ids))

    # Build catalog entries
    entries = []
    for source_id in source_ids:
        if source_id not in published_source_ids:
            result["skipped"].append(f"{source_id}: not present in R2 backups/")
            continue

        source_dir = backup_path / source_id
        if not source_dir.exists():
            msg = f"{source_id}: local directory not found at {source_dir}"
            result["errors"].append(msg)
            logger.warning(msg)
            continue

        manifest = _read_local_manifest(source_dir)
        if manifest is None:
            result["skipped"].append(f"{source_id}: no _manifest.json")
            continue

        master_entry = master_sources.get(source_id, {})
        entry = _build_catalog_entry(source_id, source_dir, manifest, master_entry)
        if entry is None:
            result["skipped"].append(f"{source_id}: description is empty")
            continue

        entries.append(entry)
        result["included"].append(source_id)

    # Assemble catalog document
    catalog = {
        "version": 1,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source_count": len(entries),
        "sources": entries,
    }

    # Upload to R2
    if storage.is_configured():
        catalog_bytes = json.dumps(catalog, indent=2, ensure_ascii=False).encode("utf-8")
        ok = storage.upload_fileobj(BytesIO(catalog_bytes), CATALOG_KEY)
        result["uploaded"] = ok
        if ok:
            logger.info(
                "Uploaded catalog with %d sources to r2://%s/%s",
                len(entries),
                storage.config.bucket_name,
                CATALOG_KEY,
            )
        else:
            result["errors"].append(f"Upload failed for {CATALOG_KEY}")
            logger.error("Failed to upload catalog to %s", CATALOG_KEY)
    else:
        logger.warning("R2 not configured -- catalog built but not uploaded")

    return result
