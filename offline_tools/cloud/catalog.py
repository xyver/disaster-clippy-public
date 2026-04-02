"""
Public catalog generation.

Builds published/catalog.json from local source manifests and uploads to R2.
The catalog is the data source for the /packs page on the public site.

The R2 backups/ folder is the authoritative list of what has been published.
Only sources that exist under backups/ in R2 are included in the catalog.
Within that set, a source is only included if its local _manifest.json has a
non-empty description field.

The catalog object must be publicly readable. Public access is a bucket-level
setting in Cloudflare R2 -- configure the bucket or use a public URL policy
separate from this code.
"""

import json
import logging
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

CATALOG_KEY = "published/catalog.json"


def _read_local_manifest(source_dir: Path) -> Optional[Dict[str, Any]]:
    """Read _manifest.json for a source. Returns None if missing or unreadable."""
    manifest_path = source_dir / "_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Could not read manifest for %s: %s", source_dir.name, e)
        return None


def _build_catalog_entry(source_id: str, manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build a single catalog entry from a manifest dict.

    Returns None if the source is not catalog-ready (missing description).
    """
    description = manifest.get("description", "").strip()
    if not description:
        return None

    return {
        "source_id": source_id,
        "name": manifest.get("name", source_id),
        "description": description,
        "license": manifest.get("license", "Unknown"),
        "tags": manifest.get("tags", []),
        "base_url": manifest.get("base_url", ""),
        "source_type": manifest.get("source_type", ""),
        "language": manifest.get("language", "en"),
        "doc_count": manifest.get("total_docs", 0),
        "size_bytes": manifest.get("total_size_bytes", 0),
        "last_updated": manifest.get("created_at", ""),
    }


def generate_public_catalog(source_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Build published/catalog.json from local manifests and upload to R2.

    Discovery:
      - If source_ids is None, lists R2 backups/ to find published source IDs.
      - If source_ids is given explicitly, uses that list (for testing or manual runs).

    Inclusion criteria (per source):
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

    result: Dict[str, Any] = {
        "included": [],
        "skipped": [],
        "errors": [],
        "uploaded": False,
        "catalog_key": CATALOG_KEY,
    }

    # Discover published source IDs from R2 if not provided
    if source_ids is None:
        if not storage.is_configured():
            result["errors"].append("R2 not configured -- cannot discover published sources")
            logger.error("R2 not configured, cannot discover published sources")
            return result

        files = storage.list_files("backups/")
        seen: set = set()
        source_ids = []
        for f in files:
            parts = f["key"].split("/")
            # Keys look like "backups/{source_id}/..."
            if len(parts) >= 2 and parts[1] and not parts[1].startswith("_"):
                if parts[1] not in seen:
                    seen.add(parts[1])
                    source_ids.append(parts[1])

        logger.info("Discovered %d published sources from R2: %s", len(source_ids), source_ids)

    # Build catalog entries
    entries = []
    for source_id in source_ids:
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

        entry = _build_catalog_entry(source_id, manifest)
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
