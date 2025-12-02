"""
Source Packs API

Endpoints for browsing and downloading source packs.
These are used by both the parent server (to serve packs)
and local instances (to fetch available packs).
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List
from pathlib import Path
import json
import io
import zipfile

from .registry import SourcePackRegistry, PackTier

router = APIRouter(prefix="/api/v1/packs", tags=["Source Packs"])


@router.get("/")
async def list_packs(
    tier: Optional[str] = Query(None, description="Filter by tier: official, community, personal"),
    downloadable_only: bool = Query(True, description="Only show packs ready for download")
):
    """
    List available source packs.

    By default, only shows packs that are ready for download (official + community).
    Use downloadable_only=false to see all packs including incomplete ones.
    """
    registry = SourcePackRegistry()

    if downloadable_only:
        packs = registry.get_downloadable_packs()
    else:
        packs = registry.get_all_packs()

    # Filter by tier if specified
    if tier:
        try:
            tier_enum = PackTier(tier.lower())
            packs = [p for p in packs if p.tier == tier_enum]
        except ValueError:
            raise HTTPException(400, f"Invalid tier: {tier}. Use: official, community, personal")

    return {
        "packs": [p.to_dict() for p in packs],
        "count": len(packs)
    }


@router.get("/official")
async def list_official_packs():
    """
    List only OFFICIAL tier packs.

    These are fully verified sources with:
    - Verified licenses
    - Complete backups
    - 100% indexed
    """
    registry = SourcePackRegistry()
    packs = registry.get_official_packs()

    return {
        "packs": [p.to_dict() for p in packs],
        "count": len(packs),
        "tier": "official",
        "description": "Fully verified sources ready for offline use"
    }


@router.get("/health")
async def pack_health_report():
    """
    Get a health report showing pack completeness status.

    Useful for maintainers to see what sources need attention.
    """
    registry = SourcePackRegistry()
    return registry.get_pack_health_report()


@router.get("/{source_id}")
async def get_pack_details(source_id: str):
    """
    Get detailed information about a specific source pack.
    """
    registry = SourcePackRegistry()
    pack = registry.get_pack(source_id)

    if not pack:
        raise HTTPException(404, f"Source pack not found: {source_id}")

    return pack.to_dict()


@router.get("/{source_id}/metadata")
async def download_pack_metadata(source_id: str):
    """
    Download just the metadata for a source pack.

    This is the lightweight option - contains document listings
    but not the actual vectors. Users can use this to:
    - See what content is available
    - Re-scrape content themselves
    - Verify completeness
    """
    registry = SourcePackRegistry()
    pack = registry.get_pack(source_id)

    if not pack:
        raise HTTPException(404, f"Source pack not found: {source_id}")

    # Load the source's metadata file
    metadata_path = registry.base_dir / "data" / "metadata" / f"{source_id}.json"

    if not metadata_path.exists():
        raise HTTPException(404, f"Metadata file not found for: {source_id}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return metadata


@router.get("/{source_id}/manifest")
async def get_pack_manifest(source_id: str):
    """
    Get the manifest for a source pack.

    The manifest contains everything needed to install the pack:
    - Source configuration
    - License information
    - Document count
    - Download options
    """
    registry = SourcePackRegistry()
    pack = registry.get_pack(source_id)

    if not pack:
        raise HTTPException(404, f"Source pack not found: {source_id}")

    # Load sources.json for this source's config
    sources_config = registry._load_sources_config()
    source_config = sources_config.get("sources", {}).get(source_id, {})

    manifest = {
        "manifest_version": "1.0",
        "pack_id": source_id,
        "pack_info": pack.to_dict(),
        "source_config": source_config,
        "download_options": {
            "metadata_only": {
                "endpoint": f"/api/v1/packs/{source_id}/metadata",
                "description": "Document metadata, can re-scrape locally",
                "size_estimate": "~100KB"
            }
        }
    }

    # Add vectors option if we have them
    # (Future: export vectors to parquet)

    # Add backup option if available
    if pack.has_backup:
        manifest["download_options"]["full_backup"] = {
            "type": pack.backup_type,
            "size_mb": pack.backup_size_mb,
            "description": "Complete offline archive"
        }

    return manifest


@router.get("/{source_id}/download")
async def download_pack(
    source_id: str,
    include_vectors: bool = Query(False, description="Include vector embeddings"),
    include_backup: bool = Query(False, description="Include backup files")
):
    """
    Download a source pack as a ZIP file.

    Options:
    - Default: Metadata + source config only (~100KB)
    - include_vectors=true: Add vector embeddings (~1-10MB per 1000 docs)
    - include_backup=true: Add backup files (varies by source)
    """
    registry = SourcePackRegistry()
    pack = registry.get_pack(source_id)

    if not pack:
        raise HTTPException(404, f"Source pack not found: {source_id}")

    # Check if pack is downloadable
    if pack.tier == PackTier.PERSONAL:
        raise HTTPException(403, "Personal tier packs are not available for download")

    # Create in-memory ZIP file
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add manifest
        manifest = {
            "manifest_version": "1.0",
            "pack_id": source_id,
            "pack_info": pack.to_dict(),
            "created_at": str(pack.last_updated),
            "includes_vectors": include_vectors,
            "includes_backup": include_backup
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        # Add source config
        sources_config = registry._load_sources_config()
        source_config = sources_config.get("sources", {}).get(source_id, {})
        pack_sources = {"sources": {source_id: source_config}}
        zf.writestr("sources.json", json.dumps(pack_sources, indent=2))

        # Add metadata
        metadata_path = registry.base_dir / "data" / "metadata" / f"{source_id}.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                zf.writestr(f"metadata/{source_id}.json", f.read())

        # Add vectors if requested (future implementation)
        if include_vectors:
            # TODO: Export vectors to parquet format
            pass

        # Add backup info if requested
        if include_backup and pack.has_backup:
            # Note: For large backups, we'd want a separate download
            # For now, just include a pointer to the backup
            backup_info = {
                "backup_type": pack.backup_type,
                "backup_size_mb": pack.backup_size_mb,
                "note": "Large backup files should be downloaded separately"
            }
            zf.writestr("backup_info.json", json.dumps(backup_info, indent=2))

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={source_id}-pack.zip"
        }
    )
