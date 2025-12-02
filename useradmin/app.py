"""
Local User Admin Panel Routes
FastAPI router for managing local user settings and offline content
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import os

from .local_config import (
    get_local_config,
    scan_backup_folder,
    check_internet_available
)

# Create router

# Include cloud upload routes
from .cloud_upload import router as cloud_upload_router
router = APIRouter(prefix="/useradmin", tags=["User Admin"])
router.include_router(cloud_upload_router)

# Templates
templates_path = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))


# Pydantic models for requests
class BackupPathUpdate(BaseModel):
    path_type: str  # zim_folder, html_folder, pdf_folder
    path: str


class OfflineModeUpdate(BaseModel):
    mode: str  # online_only, hybrid, offline_only


class SelectedSourcesUpdate(BaseModel):
    sources: List[str]


class SettingsUpdate(BaseModel):
    backup_paths: Optional[Dict[str, str]] = None
    offline_mode: Optional[str] = None
    auto_fallback: Optional[bool] = None
    selected_sources: Optional[List[str]] = None
    cache_responses: Optional[bool] = None


# Routes

@router.get("/", response_class=HTMLResponse)
async def admin_home(request: Request):
    """Main admin panel page"""
    config = get_local_config()
    internet_available = check_internet_available()

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "config": config.config,
        "internet_available": internet_available
    })


@router.get("/api/settings")
async def get_settings():
    """Get all current settings"""
    config = get_local_config()
    return JSONResponse({
        "settings": config.config,
        "internet_available": check_internet_available()
    })


@router.post("/api/settings")
async def update_settings(updates: SettingsUpdate):
    """Update multiple settings at once"""
    config = get_local_config()

    if updates.backup_paths:
        for path_type, path in updates.backup_paths.items():
            config.set_backup_path(path_type, path)

    if updates.offline_mode:
        config.set_offline_mode(updates.offline_mode)

    if updates.auto_fallback is not None:
        config.set("auto_fallback", updates.auto_fallback)

    if updates.selected_sources is not None:
        config.set_selected_sources(updates.selected_sources)

    if updates.cache_responses is not None:
        config.set("cache_responses", updates.cache_responses)

    if config.save():
        return {"status": "success", "settings": config.config}
    else:
        raise HTTPException(status_code=500, detail="Failed to save settings")


@router.post("/api/backup-path")
async def update_backup_path(update: BackupPathUpdate):
    """Update a single backup path"""
    config = get_local_config()

    # Validate path exists
    if update.path and not os.path.isdir(update.path):
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {update.path}")

    config.set_backup_path(update.path_type, update.path)

    if config.save():
        return {"status": "success", "path_type": update.path_type, "path": update.path}
    else:
        raise HTTPException(status_code=500, detail="Failed to save settings")


@router.post("/api/offline-mode")
async def update_offline_mode(update: OfflineModeUpdate):
    """Update offline mode setting"""
    config = get_local_config()

    if update.mode not in ["online_only", "hybrid", "offline_only"]:
        raise HTTPException(status_code=400, detail="Invalid mode. Use: online_only, hybrid, offline_only")

    config.set_offline_mode(update.mode)

    if config.save():
        return {"status": "success", "mode": update.mode}
    else:
        raise HTTPException(status_code=500, detail="Failed to save settings")


@router.get("/api/scan-backups")
async def scan_backups():
    """Scan configured backup folders and return available files"""
    config = get_local_config()
    # Use the unified backup folder
    backup_folder = config.get_backup_folder()

    results = {
        "zim_files": [],
        "html_files": [],
        "pdf_files": []
    }

    if backup_folder:
        results["zim_files"] = scan_backup_folder(backup_folder, "zim")
        results["html_files"] = scan_backup_folder(backup_folder, "html")
        results["pdf_files"] = scan_backup_folder(backup_folder, "pdf")

    return results


@router.get("/api/status")
async def get_status():
    """Get current system status"""
    config = get_local_config()
    paths = config.get_backup_paths()

    # Count available backup files
    zim_count = len(scan_backup_folder(paths.get("zim_folder", ""), "zim"))
    html_count = len(scan_backup_folder(paths.get("html_folder", ""), "html"))
    pdf_count = len(scan_backup_folder(paths.get("pdf_folder", ""), "pdf"))

    return {
        "internet_available": check_internet_available(),
        "offline_mode": config.get_offline_mode(),
        "auto_fallback": config.get("auto_fallback", True),
        "backup_counts": {
            "zim": zim_count,
            "html": html_count,
            "pdf": pdf_count
        },
        "last_sync": config.get("last_sync"),
        "paths_configured": {
            "zim": bool(paths.get("zim_folder")),
            "html": bool(paths.get("html_folder")),
            "pdf": bool(paths.get("pdf_folder"))
        }
    }


@router.post("/api/validate-path")
async def validate_path(data: dict):
    """Validate that a path exists and is accessible"""
    path = data.get("path", "")

    if not path:
        return {"valid": False, "error": "No path provided"}

    if not os.path.exists(path):
        return {"valid": False, "error": "Path does not exist"}

    if not os.path.isdir(path):
        return {"valid": False, "error": "Path is not a directory"}

    try:
        # Check if we can list directory contents
        os.listdir(path)
        return {"valid": True, "path": path}
    except PermissionError:
        return {"valid": False, "error": "Permission denied"}


# =============================================================================
# SOURCE PACKS BROWSER - Browse and install packs from parent server
# =============================================================================

@router.get("/packs", response_class=HTMLResponse)
async def source_packs_page(request: Request):
    """Source packs browser page"""
    config = get_local_config()
    internet_available = check_internet_available()

    return templates.TemplateResponse("packs.html", {
        "request": request,
        "config": config.config,
        "internet_available": internet_available
    })


class InstallPackRequest(BaseModel):
    source_id: str
    parent_url: str = ""  # URL of parent server, empty = local


@router.post("/api/install-pack")
async def install_pack(request: InstallPackRequest):
    """
    Install a source pack from the parent server.

    Downloads metadata and optionally vectors/backups,
    then adds the source to the local database.
    """
    import httpx

    config = get_local_config()

    # Determine parent server URL
    parent_url = request.parent_url or os.getenv("PARENT_SERVER_URL", "")

    if not parent_url:
        # If no parent URL, try to use local packs API
        # This is for testing when running everything locally
        parent_url = "http://localhost:8000"

    try:
        # Fetch pack manifest from parent
        async with httpx.AsyncClient(timeout=30.0) as client:
            manifest_url = f"{parent_url}/api/v1/packs/{request.source_id}/manifest"
            response = await client.get(manifest_url)

            if response.status_code == 404:
                raise HTTPException(404, f"Pack not found: {request.source_id}")

            if response.status_code != 200:
                raise HTTPException(500, f"Failed to fetch pack: {response.status_code}")

            manifest = response.json()

            # Fetch metadata
            metadata_url = f"{parent_url}/api/v1/packs/{request.source_id}/metadata"
            meta_response = await client.get(metadata_url)

            if meta_response.status_code != 200:
                raise HTTPException(500, "Failed to fetch pack metadata")

            metadata = meta_response.json()

        # Save source config to local sources.json
        sources_path = Path(__file__).parent.parent / "config" / "sources.json"
        sources_path.parent.mkdir(parents=True, exist_ok=True)

        if sources_path.exists():
            with open(sources_path, 'r', encoding='utf-8') as f:
                local_sources = json.load(f)
        else:
            local_sources = {"version": 1, "sources": {}}

        # Add the source from the pack
        source_config = manifest.get("source_config", {})
        local_sources["sources"][request.source_id] = source_config

        with open(sources_path, 'w', encoding='utf-8') as f:
            json.dump(local_sources, f, indent=2)

        # Save metadata
        metadata_dir = Path(__file__).parent.parent / "data" / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = metadata_dir / f"{request.source_id}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # Update installed packs in local config
        installed = config.get("installed_packs", [])
        if request.source_id not in installed:
            installed.append(request.source_id)
            config.set("installed_packs", installed)
            config.save()

        return {
            "status": "success",
            "source_id": request.source_id,
            "document_count": manifest.get("pack_info", {}).get("document_count", 0),
            "message": f"Installed {request.source_id} pack successfully"
        }

    except httpx.RequestError as e:
        raise HTTPException(500, f"Network error: {str(e)}")


@router.get("/api/installed-packs")
async def get_installed_packs():
    """Get list of locally installed packs"""
    config = get_local_config()
    installed = config.get("installed_packs", [])

    # Load local sources config
    sources_path = Path(__file__).parent.parent / "config" / "sources.json"
    local_sources = {}

    if sources_path.exists():
        with open(sources_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            local_sources = data.get("sources", {})

    # Build installed packs list with details
    packs = []
    for source_id in installed:
        source_config = local_sources.get(source_id, {})

        # Check if metadata exists
        metadata_path = Path(__file__).parent.parent / "data" / "metadata" / f"{source_id}.json"
        doc_count = 0
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    doc_count = meta.get("total_documents", meta.get("document_count", 0))
            except Exception:
                pass

        packs.append({
            "source_id": source_id,
            "name": source_config.get("name", source_id),
            "license": source_config.get("license", "Unknown"),
            "document_count": doc_count,
            "installed": True
        })

    return {"packs": packs, "count": len(packs)}


@router.get("/api/local-sources")
async def get_local_sources():
    """
    Get ALL local sources with completeness status.
    Shows config, metadata, backup, and license verification status for each.
    """
    from .cloud_upload import get_backup_paths

    backup_paths = get_backup_paths()
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    # Load sources config
    sources_path = Path(__file__).parent.parent / "config" / "sources.json"
    if not sources_path.exists():
        return {"sources": [], "error": "No sources.json found"}

    with open(sources_path, 'r', encoding='utf-8') as f:
        sources_data = json.load(f)

    sources_config = sources_data.get("sources", {})

    # Check each source for completeness
    local_sources = []

    for source_id, config_data in sources_config.items():
        source_status = {
            "source_id": source_id,
            "name": config_data.get("name", source_id),
            "description": config_data.get("description", ""),
            "license": config_data.get("license", "Unknown"),
            "license_verified": config_data.get("license_verified", False),
            "has_config": True,
            "has_metadata": False,
            "has_backup": False,
            "backup_type": None,
            "backup_size_mb": 0,
            "document_count": 0,
            "is_complete": False,
            "missing": []
        }

        # Check for metadata
        metadata_path = Path(__file__).parent.parent / "data" / "metadata" / f"{source_id}.json"
        if metadata_path.exists():
            source_status["has_metadata"] = True
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    source_status["document_count"] = meta.get("total_documents", 0)
            except Exception:
                pass
        else:
            source_status["missing"].append("metadata")

        # Check for backup files (ZIM or HTML folder) in unified backup folder
        backup_found = False

        if backup_folder:
            # Check ZIM file
            zim_path = Path(backup_folder) / f"{source_id}.zim"
            if zim_path.exists():
                backup_found = True
                source_status["has_backup"] = True
                source_status["backup_type"] = "zim"
                source_status["backup_size_mb"] = round(zim_path.stat().st_size / (1024*1024), 2)

            # Check HTML folder
            if not backup_found:
                html_path = Path(backup_folder) / source_id
                if html_path.exists() and html_path.is_dir():
                    # Verify it has HTML content
                    html_files = list(html_path.rglob('*.html'))
                    if html_files:
                        backup_found = True
                        source_status["has_backup"] = True
                        source_status["backup_type"] = "html"
                        try:
                            total_size = sum(f.stat().st_size for f in html_path.rglob('*') if f.is_file())
                            source_status["backup_size_mb"] = round(total_size / (1024*1024), 2)
                        except Exception:
                            pass

        if not backup_found:
            source_status["missing"].append("backup file")
        elif source_status["backup_size_mb"] < 0.1:
            source_status["has_backup"] = False
            source_status["missing"].append("backup file (empty)")

        # Check license
        if not source_status["license_verified"]:
            source_status["missing"].append("verified license")

        # Determine if complete
        source_status["is_complete"] = (
            source_status["has_config"] and
            source_status["has_metadata"] and
            source_status["has_backup"] and
            source_status["backup_size_mb"] >= 0.1 and
            source_status["license_verified"]
        )

        local_sources.append(source_status)

    # Sort: complete first, then by name
    local_sources.sort(key=lambda x: (not x["is_complete"], x["name"]))

    return {
        "sources": local_sources,
        "total": len(local_sources),
        "complete_count": sum(1 for s in local_sources if s["is_complete"])
    }


@router.get("/api/available-packs")
async def get_available_packs():
    """
    Get packs available from the Global Cloud Backup (R2 backups/ folder).
    These are verified packs ready for download.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from storage.r2 import get_r2_storage
        storage = get_r2_storage()

        if not storage.is_configured():
            return {"packs": [], "connected": False, "error": "R2 not configured"}

        # Test connection
        conn_status = storage.test_connection()
        if not conn_status["connected"]:
            return {"packs": [], "connected": False, "error": conn_status.get("error", "Connection failed")}

        # List files in backups/ folder
        files = storage.list_files("backups/")

        # Group files by source_id (folder name)
        packs = {}
        for f in files:
            key = f["key"]
            parts = key.split("/")
            if len(parts) >= 2:
                source_id = parts[1]  # e.g., "backups/builditsolar/..."
                if source_id and source_id not in packs:
                    packs[source_id] = {
                        "source_id": source_id,
                        "name": source_id.replace("-", " ").replace("_", " ").title(),
                        "files": [],
                        "total_size_mb": 0,
                        "tier": "official",
                        "description": "Official backup from Global Cloud Storage"
                    }
                if source_id:
                    packs[source_id]["files"].append(f)
                    packs[source_id]["total_size_mb"] += f.get("size_mb", 0)

        # Convert to list
        pack_list = list(packs.values())

        # Try to load manifest for each pack if available
        for pack in pack_list:
            manifest_key = f"backups/{pack['source_id']}/manifest.json"
            if storage.file_exists(manifest_key):
                # Would need to download and parse - for now skip
                pass

        return {
            "packs": pack_list,
            "connected": True,
            "total": len(pack_list)
        }

    except ImportError:
        return {"packs": [], "connected": False, "error": "Storage module not available"}
    except Exception as e:
        return {"packs": [], "connected": False, "error": str(e)}


# Need to import json at the top
import json


class GenerateMetadataRequest(BaseModel):
    source_id: str


@router.post("/api/generate-metadata")
async def generate_metadata_from_backup(request: GenerateMetadataRequest):
    """
    Scan a local backup folder and generate metadata.json for it.
    This is required before a source can be submitted to the cloud.
    """
    from sourcepacks.pack_tools import generate_metadata_from_html

    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise HTTPException(400, "No backup folder configured. Set it in Settings first.")

    # Check for HTML backup folder
    html_path = Path(backup_folder) / request.source_id
    zim_path = Path(backup_folder) / f"{request.source_id}.zim"

    if html_path.exists() and html_path.is_dir():
        try:
            metadata = generate_metadata_from_html(
                backup_path=str(html_path),
                source_id=request.source_id,
                save=True
            )
            return {
                "status": "success",
                "source_id": request.source_id,
                "document_count": metadata.get("total_documents", 0),
                "total_chars": metadata.get("total_chars", 0),
                "source_type": "html",
                "message": f"Generated metadata for {request.source_id}: {metadata.get('total_documents', 0)} documents"
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to generate metadata: {e}")

    elif zim_path.exists():
        # ZIM requires special handling - create placeholder for now
        from datetime import datetime
        metadata = {
            "version": 2,
            "source_id": request.source_id,
            "source_type": "zim",
            "last_updated": datetime.now().isoformat(),
            "total_documents": 1,
            "document_count": 1,
            "total_chars": 0,
            "documents": {
                "zim_placeholder": {
                    "title": f"{request.source_id} ZIM Archive",
                    "url": str(zim_path),
                    "note": "ZIM files require full indexing via admin tools"
                }
            }
        }
        from sourcepacks.pack_tools import save_metadata
        save_metadata(request.source_id, metadata)

        return {
            "status": "success",
            "source_id": request.source_id,
            "document_count": 1,
            "source_type": "zim",
            "message": f"Created placeholder metadata for ZIM file. Use admin tools for full indexing."
        }
    else:
        raise HTTPException(404, f"No backup found for {request.source_id} in {backup_folder}")


class IndexSourceRequest(BaseModel):
    source_id: str


@router.post("/api/index-source")
async def index_source_to_chromadb(request: IndexSourceRequest):
    """
    Index a source's content into local ChromaDB.
    Requires metadata to already exist.
    """
    from sourcepacks.pack_tools import load_metadata, index_html_to_chromadb, index_zim_to_chromadb

    # Check metadata exists
    metadata = load_metadata(request.source_id)
    if not metadata:
        raise HTTPException(400, f"No metadata found for {request.source_id}. Generate metadata first.")

    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise HTTPException(400, "No backup folder configured")

    # Get backup path
    html_path = Path(backup_folder) / request.source_id
    zim_path = Path(backup_folder) / f"{request.source_id}.zim"

    if html_path.exists() and html_path.is_dir():
        try:
            result = index_html_to_chromadb(str(html_path), request.source_id)
            return {
                "status": "success",
                "source_id": request.source_id,
                "indexed_count": result.get("indexed", 0),
                "skipped_count": result.get("skipped", 0),
                "message": f"Indexed {result.get('indexed', 0)} documents to ChromaDB"
            }
        except ImportError as e:
            raise HTTPException(500, f"Indexer not available: {e}")
        except Exception as e:
            raise HTTPException(500, f"Indexing failed: {e}")

    elif zim_path.exists():
        try:
            result = index_zim_to_chromadb(str(zim_path), request.source_id)
            return {
                "status": "success",
                "source_id": request.source_id,
                "indexed_count": result.get("indexed", 0),
                "message": f"Indexed {result.get('indexed', 0)} documents from ZIM to ChromaDB"
            }
        except ImportError as e:
            raise HTTPException(500, f"ZIM indexer not available (install libzim): {e}")
        except Exception as e:
            raise HTTPException(500, f"ZIM indexing failed: {e}")

    else:
        raise HTTPException(404, f"No backup found for {request.source_id}")


@router.get("/api/source-status/{source_id}")
async def get_source_status(source_id: str):
    """
    Get detailed status of a source - what's present and what's missing.
    """
    from sourcepacks.pack_tools import get_source_completeness

    config = get_local_config()
    backup_folder = config.get_backup_folder()

    return get_source_completeness(source_id, backup_folder)


@router.post("/api/export-index/{source_id}")
async def export_source_index(source_id: str):
    """
    Export ChromaDB embeddings for a source to a portable JSON format.
    This can be included in pack submissions for Layer 1 sync.
    """
    from sourcepacks.pack_tools import export_chromadb_index, save_index_export

    try:
        export_data = export_chromadb_index(source_id)

        if export_data.get("count", 0) == 0:
            raise HTTPException(404, f"No indexed documents found for {source_id}")

        # Save to file
        index_file = save_index_export(source_id, export_data)

        return {
            "status": "success",
            "source_id": source_id,
            "document_count": export_data["count"],
            "index_path": str(index_file),
            "message": f"Exported {export_data['count']} embeddings to {index_file.name}"
        }

    except Exception as e:
        raise HTTPException(500, f"Export failed: {e}")
