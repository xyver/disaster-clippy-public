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

# Use shared source management functions
from sourcepacks.pack_tools import (
    load_master_metadata,
    get_source_sync_status
)

# Create router

# Include cloud upload routes
from .cloud_upload import router as cloud_upload_router
router = APIRouter(prefix="/useradmin", tags=["User Admin"])
router.include_router(cloud_upload_router)


# Known ZIM sources and their URLs (Kiwix naming convention)
ZIM_SOURCE_URLS = {
    "bitcoin": "https://en.bitcoin.it/wiki/",
    "wikipedia": "https://en.wikipedia.org/wiki/",
    "wiktionary": "https://en.wiktionary.org/wiki/",
    "wikihow": "https://www.wikihow.com/",
    "stackexchange": "https://stackexchange.com/",
    "stackoverflow": "https://stackoverflow.com/",
    "gutenberg": "https://www.gutenberg.org/",
    "ted": "https://www.ted.com/",
    "phet": "https://phet.colorado.edu/",
}


def _analyze_zim_file(zim_path: Path, source_id: str) -> Dict[str, Any]:
    """
    Analyze a ZIM file to extract metadata.

    Derives URL from filename (Kiwix convention: {project}_{lang}_{selection}_{date}.zim)
    and analyzes content for tags.
    """
    result = {
        "base_url": None,
        "tags": [],
        "description": None,
        "article_count": 0
    }

    # Parse ZIM filename for project info
    # Example: bitcoin_en_all_maxi_2021-03.zim
    filename = zim_path.stem  # bitcoin_en_all_maxi_2021-03
    parts = filename.split('_')

    if parts:
        project = parts[0].lower()

        # Look up known URL
        if project in ZIM_SOURCE_URLS:
            result["base_url"] = ZIM_SOURCE_URLS[project]
        else:
            # Try common wiki patterns
            result["base_url"] = f"https://{project}.org/"

    # Try to analyze ZIM content for better tags
    try:
        from zimply_core.zim_core import ZIMFile
        zim = ZIMFile(str(zim_path), 'utf-8')

        result["article_count"] = zim.header_fields.get('articleCount', 0)

        # Sample some article titles to infer topics
        sample_titles = []
        article_count = result["article_count"]

        # Sample articles spread across the archive
        sample_indices = [
            i for i in range(0, min(article_count, 500), 10)
        ]

        for i in sample_indices[:50]:
            try:
                article = zim.get_article_by_id(i)
                if article:
                    mimetype = getattr(article, 'mimetype', '')
                    if 'text/html' in str(mimetype).lower():
                        title = getattr(article, 'title', '')
                        url = getattr(article, 'url', '')
                        # Skip special pages
                        if title and not any(x in url.lower() for x in ['special:', 'file:', 'category:', 'template:']):
                            sample_titles.append(title.lower())
            except:
                continue

        # Infer tags from content
        result["tags"] = _infer_tags_from_titles(sample_titles, source_id)

        # Generate description
        if result["article_count"] > 0:
            project_name = source_id.replace('_', ' ').title()
            result["description"] = f"{project_name} offline archive with {result['article_count']:,} articles"

        # Close ZIM file to release the handle
        zim.close()

    except ImportError:
        # zimply-core not installed, use filename-based inference
        result["tags"] = _infer_tags_from_source_id(source_id)
    except Exception as e:
        print(f"ZIM analysis error: {e}")
        result["tags"] = _infer_tags_from_source_id(source_id)

    return result


def _infer_tags_from_titles(titles: List[str], source_id: str) -> List[str]:
    """Infer relevant tags from a list of article titles"""

    # Topic keywords to look for (more specific terms for better matching)
    topic_keywords = {
        "cryptocurrency": ["bitcoin", "btc", "blockchain", "wallet", "mining", "transaction", "satoshi", "crypto", "altcoin", "block", "bip", "segwit", "lightning"],
        "finance": ["money", "currency", "payment", "exchange", "trading", "investment", "price", "market"],
        "technology": ["software", "hardware", "computer", "network", "protocol", "algorithm", "node", "client"],
        "programming": ["code", "script", "api", "developer", "programming", "library", "json", "rpc"],
        "security": ["encryption", "key", "password", "secure", "private", "hash", "privacy", "signature"],
        "reference": ["wiki", "encyclopedia", "guide", "manual", "documentation", "reference", "history"],
        "energy": ["solar", "power", "electricity", "battery", "generator", "wind", "inverter", "panel"],
        "survival": ["emergency", "disaster", "first aid", "shelter", "water purif", "food storage", "prepar"],
        "agriculture": ["farming", "garden", "crop", "soil", "harvest", "plant", "seed", "compost", "irrigation"],
        "medical": ["health", "medicine", "treatment", "disease", "symptom", "therapy", "first aid", "wound"],
        "construction": ["building", "construction", "tool", "repair", "diy", "woodwork", "plumb", "wiring"],
        "cooking": ["recipe", "cook", "bake", "food", "kitchen", "preserv", "canning"],
    }

    # Count keyword matches
    topic_scores = {topic: 0 for topic in topic_keywords}
    title_text = ' '.join(titles)

    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            if keyword in title_text:
                topic_scores[topic] += title_text.count(keyword)

    # Get top scoring topics
    sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
    tags = [topic for topic, score in sorted_topics if score > 2][:5]

    # Always include source-based tag if no good matches
    if not tags:
        tags = _infer_tags_from_source_id(source_id)

    return tags


def _infer_tags_from_source_id(source_id: str) -> List[str]:
    """Infer tags from source ID when content analysis isn't possible"""
    source_lower = source_id.lower()

    tag_mappings = {
        "bitcoin": ["cryptocurrency", "finance", "technology"],
        "wikipedia": ["reference", "encyclopedia"],
        "wikihow": ["how-to", "guides", "diy"],
        "stackoverflow": ["programming", "technology", "reference"],
        "gutenberg": ["literature", "books", "reference"],
        "medical": ["medical", "health"],
        "survival": ["survival", "emergency"],
        "solar": ["energy", "solar", "diy"],
        "garden": ["agriculture", "gardening"],
        "food": ["food", "cooking", "preservation"],
    }

    for key, tags in tag_mappings.items():
        if key in source_lower:
            return tags

    return ["reference"]

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
    backup_folder: Optional[str] = None  # Unified backup folder path
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

    # Handle unified backup_folder (updates both local_settings.json and .env)
    if updates.backup_folder is not None:
        if updates.backup_folder and not os.path.isdir(updates.backup_folder):
            raise HTTPException(status_code=400, detail=f"Directory does not exist: {updates.backup_folder}")
        config.set_backup_folder(updates.backup_folder)

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
        # Reload vector store to use new path
        try:
            from app import reload_vector_store
            reload_vector_store()
            print(f"Vector store reloaded with new backup path: {update.path}")
        except Exception as e:
            print(f"Warning: Could not reload vector store: {e}")

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

    # If switching to offline mode and auto-start is enabled, start Ollama
    if update.mode == "offline_only" and config.is_ollama_enabled():
        ollama_config = config.get_ollama_config()
        if ollama_config.get("auto_start", True):
            try:
                from .ollama_manager import get_ollama_manager
                ollama = get_ollama_manager()
                if ollama.is_installed() and not ollama.is_running():
                    ollama.start()
            except Exception as e:
                print(f"Could not auto-start Ollama: {e}")

    if config.save():
        return {"status": "success", "mode": update.mode}
    else:
        raise HTTPException(status_code=500, detail="Failed to save settings")


# Ollama endpoints
class OllamaConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    url: Optional[str] = None
    model: Optional[str] = None
    auto_start: Optional[bool] = None


@router.get("/api/ollama/status")
async def get_ollama_status():
    """Get Ollama status and configuration"""
    config = get_local_config()
    ollama_config = config.get_ollama_config()

    try:
        from .ollama_manager import get_ollama_manager
        ollama = get_ollama_manager()
        status = ollama.get_status()
    except Exception as e:
        status = {
            "installed": False,
            "running": False,
            "error": str(e)
        }

    return {
        "config": ollama_config,
        "status": status
    }


@router.post("/api/ollama/config")
async def update_ollama_config(update: OllamaConfigUpdate):
    """Update Ollama configuration"""
    config = get_local_config()

    updates = {}
    if update.enabled is not None:
        updates["enabled"] = update.enabled
    if update.url is not None:
        updates["url"] = update.url
    if update.model is not None:
        updates["model"] = update.model
    if update.auto_start is not None:
        updates["auto_start"] = update.auto_start

    if updates:
        config.set_ollama_config(**updates)
        if config.save():
            # Reload Ollama manager with new config
            try:
                from .ollama_manager import reload_ollama_manager
                reload_ollama_manager()
            except Exception:
                pass
            return {"status": "success", "config": config.get_ollama_config()}
        else:
            raise HTTPException(status_code=500, detail="Failed to save config")

    return {"status": "no_changes", "config": config.get_ollama_config()}


@router.post("/api/ollama/start")
async def start_ollama():
    """Start portable Ollama server"""
    try:
        from .ollama_manager import get_ollama_manager
        ollama = get_ollama_manager()

        if not ollama.is_installed():
            raise HTTPException(400, "Portable Ollama not installed. Download the Ollama pack first.")

        if ollama.is_running():
            return {"status": "already_running", "url": ollama.url}

        if ollama.start():
            return {"status": "started", "url": ollama.url}
        else:
            raise HTTPException(500, "Failed to start Ollama")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error starting Ollama: {e}")


@router.post("/api/ollama/stop")
async def stop_ollama():
    """Stop Ollama server (if we started it)"""
    try:
        from .ollama_manager import get_ollama_manager
        ollama = get_ollama_manager()

        if ollama.stop():
            return {"status": "stopped"}
        else:
            return {"status": "not_running"}

    except Exception as e:
        raise HTTPException(500, f"Error stopping Ollama: {e}")


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

        # Save metadata to BACKUP_PATH/{source_id}/{source_id}_metadata.json
        backup_folder = config.get_backup_folder()
        if backup_folder:
            metadata_dir = Path(backup_folder) / request.source_id
        else:
            metadata_dir = Path(__file__).parent.parent / "data" / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = metadata_dir / f"{request.source_id}_metadata.json"
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

    # Build installed packs list with details
    backup_folder = config.get_backup_folder()
    packs = []
    for source_id in installed:
        # Load source config from _source.json
        source_config = {}
        if backup_folder:
            source_file = Path(backup_folder) / source_id / f"{source_id}_source.json"
            if source_file.exists():
                try:
                    with open(source_file, 'r', encoding='utf-8') as f:
                        source_config = json.load(f)
                except Exception:
                    pass

        # Check if metadata exists - try BACKUP_PATH first, then legacy
        doc_count = 0
        metadata_path = None
        if backup_folder:
            metadata_path = Path(backup_folder) / source_id / f"{source_id}_metadata.json"
        if not metadata_path or not metadata_path.exists():
            metadata_path = Path(__file__).parent.parent / "data" / "metadata" / f"{source_id}.json"

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
    Source of truth is the backup folder - discovers all sources from
    {source_id}_source.json files in each source subfolder.
    """
    from .cloud_upload import get_backup_paths

    backup_paths = get_backup_paths()
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    # Primary discovery: scan backup folder for all sources
    # Source of truth is {source_id}_source.json in each source folder
    sources_config = {}

    if backup_folder:
        backup_path = Path(backup_folder)
        if backup_path.exists():
            for source_folder in backup_path.iterdir():
                if source_folder.is_dir():
                    source_id = source_folder.name

                    # Check for v2 schema file first (preferred)
                    source_file = source_folder / f"{source_id}_source.json"
                    if source_file.exists():
                        try:
                            with open(source_file, 'r', encoding='utf-8') as f:
                                source_data = json.load(f)
                            sources_config[source_id] = {
                                "name": source_data.get("name", source_id),
                                "description": source_data.get("description", ""),
                                "license": source_data.get("license", "Unknown"),
                                "base_url": source_data.get("base_url", ""),
                                "license_verified": source_data.get("license_verified", False),
                            }
                        except Exception:
                            sources_config[source_id] = {"name": source_id}

                    # Also check for manifest or metadata files (v1 or downloaded packs)
                    elif (source_folder / f"{source_id}_manifest.json").exists():
                        try:
                            manifest_path = source_folder / f"{source_id}_manifest.json"
                            with open(manifest_path, 'r', encoding='utf-8') as f:
                                manifest = json.load(f)
                            sources_config[source_id] = {
                                "name": manifest.get("name", source_id),
                                "description": manifest.get("description", ""),
                                "license": manifest.get("license", "Unknown"),
                                "base_url": manifest.get("base_url", ""),
                            }
                        except Exception:
                            sources_config[source_id] = {"name": source_id}

                    # Check for pages folder or any content
                    elif (source_folder / "pages").exists() or list(source_folder.glob("*.html")):
                        sources_config[source_id] = {"name": source_id}

    if not sources_config:
        return {"sources": [], "total": 0, "complete_count": 0}

    # Check each source for completeness
    local_sources = []

    for source_id, config_data in sources_config.items():
        source_status = {
            "source_id": source_id,
            "name": config_data.get("name", source_id),
            "description": config_data.get("description", ""),
            "license": config_data.get("license", "Unknown"),
            "license_verified": config_data.get("license_verified", False),
            "base_url": config_data.get("base_url", ""),
            "has_config": True,
            "has_metadata": False,
            "has_backup": False,
            "has_embeddings": False,  # For offline search
            "backup_type": None,
            "backup_size_mb": 0,
            "document_count": 0,
            "is_complete": False,
            "production_ready": False,  # Ready for submission to global repo
            "missing": [],
            # V2 schema fields
            "schema_version": 1,
            "has_source_metadata": False,
            "has_documents_file": False,
            "has_embeddings_file": False,
        }

        # Check for metadata - try BACKUP_PATH/{source_id}/{source_id}_metadata.json first
        metadata_path = None
        if backup_folder:
            metadata_path = Path(backup_folder) / source_id / f"{source_id}_metadata.json"
        if not metadata_path or not metadata_path.exists():
            # Fallback to legacy location
            metadata_path = Path(__file__).parent.parent / "data" / "metadata" / f"{source_id}.json"

        if metadata_path.exists():
            source_status["has_metadata"] = True
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    # Try document_count first, then total_documents for legacy
                    source_status["document_count"] = meta.get("document_count", meta.get("total_documents", 0))
            except Exception:
                pass
        else:
            source_status["missing"].append("metadata")

        # Also check v2 documents file - this is the PRIMARY source for v2 schema
        if backup_folder:
            documents_file = Path(backup_folder) / source_id / f"{source_id}_documents.json"
            if documents_file.exists():
                # V2 documents file counts as metadata
                source_status["has_metadata"] = True
                if "metadata" in source_status["missing"]:
                    source_status["missing"].remove("metadata")
                try:
                    with open(documents_file, 'r', encoding='utf-8') as f:
                        docs_data = json.load(f)
                        doc_count = docs_data.get("document_count", 0)
                        if doc_count > source_status["document_count"]:
                            source_status["document_count"] = doc_count
                except Exception:
                    pass

        # Check for backup files using unified detection
        from sourcepacks.pack_tools import detect_backup_status
        backup_status = detect_backup_status(source_id, Path(backup_folder) if backup_folder else None)
        source_status["has_backup"] = backup_status["has_backup"]
        source_status["backup_type"] = backup_status["backup_type"]
        source_status["backup_size_mb"] = backup_status["backup_size_mb"]

        if not backup_status["has_backup"]:
            source_status["missing"].append("backup file")
        elif backup_status["backup_size_mb"] < 0.1:
            source_status["has_backup"] = False
            source_status["missing"].append("backup file (empty)")

        # Check for v2 schema files
        if backup_folder:
            source_folder = Path(backup_folder) / source_id
            source_file = source_folder / f"{source_id}_source.json"
            documents_file = source_folder / f"{source_id}_documents.json"
            embeddings_file = source_folder / f"{source_id}_embeddings.json"

            source_status["has_source_metadata"] = source_file.exists()
            source_status["has_documents_file"] = documents_file.exists()
            source_status["has_embeddings_file"] = embeddings_file.exists()

            # Determine schema version
            if source_status["has_source_metadata"] and source_status["has_documents_file"] and source_status["has_embeddings_file"]:
                source_status["schema_version"] = 2
                source_status["has_embeddings"] = True  # V2 embeddings file counts
            else:
                # Check legacy index file
                index_path = source_folder / f"{source_id}_index.json"
                if index_path.exists():
                    try:
                        with open(index_path, 'r', encoding='utf-8') as f:
                            index_data = json.load(f)
                        if index_data.get("documents"):
                            source_status["has_embeddings"] = True
                    except Exception:
                        pass

        if not source_status["has_embeddings"]:
            source_status["missing"].append("embeddings (for offline search)")

        # Try to get base_url from manifest files if not in config
        if not source_status["base_url"] and backup_folder:
            source_folder = Path(backup_folder) / source_id
            # Check distribution manifest first
            dist_manifest = source_folder / f"{source_id}_manifest.json"
            if dist_manifest.exists():
                try:
                    with open(dist_manifest, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    if manifest.get("base_url"):
                        source_status["base_url"] = manifest["base_url"]
                except Exception:
                    pass
            # Check backup manifest
            if not source_status["base_url"]:
                backup_manifest = source_folder / f"{source_id}_backup_manifest.json"
                if not backup_manifest.exists():
                    backup_manifest = source_folder / "manifest.json"
                if backup_manifest.exists():
                    try:
                        with open(backup_manifest, 'r', encoding='utf-8') as f:
                            manifest = json.load(f)
                        if manifest.get("base_url"):
                            source_status["base_url"] = manifest["base_url"]
                    except Exception:
                        pass

        # Check license
        license_val = source_status["license"]
        has_license = license_val and license_val.lower() not in ["unknown", ""]
        if not has_license:
            source_status["missing"].append("license")
        if not source_status["license_verified"]:
            source_status["missing"].append("verified license")

        # Determine if complete (basic local use)
        source_status["is_complete"] = (
            source_status["has_config"] and
            source_status["has_metadata"] and
            source_status["has_backup"] and
            source_status["backup_size_mb"] >= 0.1
        )

        # Determine if production ready (can be submitted to global repo)
        source_status["production_ready"] = (
            source_status["is_complete"] and
            source_status["schema_version"] == 2 and
            source_status["has_embeddings"] and
            has_license
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
        # Skip special files like _master.json, sources.json that are at root level
        skip_files = {"_master.json", "sources.json", "backups.json"}
        packs = {}
        for f in files:
            key = f["key"]
            parts = key.split("/")
            if len(parts) >= 2:
                source_id = parts[1]  # e.g., "backups/builditsolar/..."
                # Skip if source_id is a special file (not a folder)
                if source_id in skip_files or source_id.startswith("_"):
                    continue
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


class DownloadPackRequest(BaseModel):
    source_id: str


def _run_download_pack(source_id: str, progress_callback=None):
    """
    Background job function to download a source pack from R2.
    This runs in a separate thread via the job manager.
    """
    import json
    import zipfile
    from dotenv import load_dotenv
    load_dotenv()
    from storage.r2 import get_r2_storage

    def update_progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    update_progress(0, "Connecting to R2 storage...")
    storage = get_r2_storage()

    if not storage.is_configured():
        raise Exception("R2 storage not configured")

    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise Exception("No backup folder configured. Set it in Settings first.")

    source_folder = Path(backup_folder) / source_id
    source_folder.mkdir(parents=True, exist_ok=True)

    # List all files for this source in R2
    update_progress(5, "Listing files in R2...")
    prefix = f"backups/{source_id}/"
    files = storage.list_files(prefix)

    if not files:
        raise Exception(f"No files found for source: {source_id}")

    downloaded_files = []
    total_size = 0
    total_files = len(files)

    # Download files with progress
    for idx, f in enumerate(files):
        key = f["key"]
        # Get the relative path within the source folder
        relative_path = key.replace(prefix, "")
        if not relative_path:
            continue

        local_path = source_folder / relative_path

        # Create subdirectories if needed
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate progress (10-70% for downloads)
        pct = 10 + int((idx / total_files) * 60)
        update_progress(pct, f"Downloading {relative_path}...")

        # Download the file
        success = storage.download_file(key, str(local_path))
        if success:
            downloaded_files.append(relative_path)
            total_size += f.get("size_mb", 0)

    # Extract HTML zip files if present (70-80%)
    update_progress(70, "Extracting zip files...")
    html_zip_patterns = [
        source_folder / f"{source_id}-html.zip",
        source_folder / f"{source_id}.zip"
    ]
    for zip_path in html_zip_patterns:
        if zip_path.exists():
            try:
                print(f"Extracting HTML backup from {zip_path.name}...")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(source_folder)
                print(f"Extracted {zip_path.name} to {source_folder}")
            except Exception as e:
                print(f"Warning: Could not extract {zip_path.name}: {e}")

    # Import index data into ChromaDB if index file exists (80-95%)
    update_progress(80, "Importing to ChromaDB...")
    index_path = source_folder / f"{source_id}_index.json"
    indexed_count = 0
    if index_path.exists():
        try:
            from vectordb import get_vector_store

            # Load the index file
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            documents = index_data.get("documents", [])
            if documents:
                # Get or reload vector store
                store = get_vector_store()

                # Prepare data for ChromaDB
                ids = []
                embeddings = []
                contents = []
                metadatas = []

                for doc in documents:
                    ids.append(doc["id"])
                    embeddings.append(doc["embedding"])
                    contents.append(doc.get("content", ""))
                    metadatas.append(doc.get("metadata", {}))

                # Add to ChromaDB in batches
                batch_size = 500
                total_batches = (len(ids) + batch_size - 1) // batch_size
                for i, start in enumerate(range(0, len(ids), batch_size)):
                    batch_ids = ids[start:start+batch_size]
                    batch_embeddings = embeddings[start:start+batch_size]
                    batch_contents = contents[start:start+batch_size]
                    batch_metadatas = metadatas[start:start+batch_size]

                    # Update progress during batch import (85-95%)
                    pct = 85 + int((i / total_batches) * 10)
                    update_progress(pct, f"Importing batch {i+1}/{total_batches}...")

                    store.collection.upsert(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_contents,
                        metadatas=batch_metadatas
                    )

                indexed_count = len(ids)
                print(f"Imported {indexed_count} documents from {source_id} into ChromaDB")

        except Exception as e:
            print(f"Warning: Could not import index to ChromaDB: {e}")

    # Update installed packs in local config (95-100%)
    update_progress(95, "Updating local configuration...")
    installed = config.get("installed_packs", [])
    if source_id not in installed:
        installed.append(source_id)
        config.set("installed_packs", installed)
        config.save()

    # Create _source.json if it doesn't exist (so source appears in local sources list)
    source_file = source_folder / f"{source_id}_source.json"
    if not source_file.exists():
        from datetime import datetime
        # Try to get info from manifest if available
        manifest_path = source_folder / f"{source_id}_manifest.json"
        source_data = {
            "schema_version": 2,
            "source_id": source_id,
            "name": source_id.replace("_", " ").replace("-", " ").title(),
            "description": "",
            "license": "Unknown",
            "base_url": "",
            "total_docs": 0,
            "total_chars": 0,
            "categories": {},
            "created_at": datetime.now().isoformat(),
            "last_backup": "",
            "last_indexed": "",
            "license_verified": False
        }
        # Try to pull data from manifest if it exists
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as mf:
                    manifest = json.load(mf)
                    mc = manifest.get("source_config", {})
                    if mc.get("name"):
                        source_data["name"] = mc["name"]
                    if mc.get("description"):
                        source_data["description"] = mc["description"]
                    if mc.get("license"):
                        source_data["license"] = mc["license"]
                    if mc.get("base_url"):
                        source_data["base_url"] = mc["base_url"]
            except:
                pass
        with open(source_file, 'w', encoding='utf-8') as f:
            json.dump(source_data, f, indent=2)

    update_progress(100, "Download complete")

    return {
        "status": "success",
        "source_id": source_id,
        "files_downloaded": len(downloaded_files),
        "total_size_mb": round(total_size, 2),
        "indexed_documents": indexed_count,
        "message": f"Downloaded {len(downloaded_files)} files and indexed {indexed_count} documents"
    }


@router.post("/api/download-pack")
async def download_pack_from_r2(request: DownloadPackRequest):
    """
    Download a source pack from R2 cloud storage.
    Submits as background job - returns job_id for tracking.
    """
    try:
        # Check prerequisites before submitting job
        from dotenv import load_dotenv
        load_dotenv()
        from storage.r2 import get_r2_storage
        storage = get_r2_storage()

        if not storage.is_configured():
            raise HTTPException(400, "R2 storage not configured")

        config = get_local_config()
        backup_folder = config.get_backup_folder()

        if not backup_folder:
            raise HTTPException(400, "No backup folder configured. Set it in Settings first.")

        # Verify source exists in R2 before starting job
        prefix = f"backups/{request.source_id}/"
        files = storage.list_files(prefix)
        if not files:
            raise HTTPException(404, f"No files found for source: {request.source_id}")

        # Submit as background job
        from .job_manager import get_job_manager
        manager = get_job_manager()

        try:
            job_id = manager.submit(
                "download",
                request.source_id,
                _run_download_pack,
                request.source_id
            )
        except ValueError as e:
            # Job already running for this source
            raise HTTPException(409, str(e))

        return {
            "status": "submitted",
            "job_id": job_id,
            "source_id": request.source_id,
            "message": f"Download job started for {request.source_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to start download: {str(e)}")


class ReindexPackRequest(BaseModel):
    source_id: str


@router.post("/api/reindex-pack")
async def reindex_pack(request: ReindexPackRequest):
    """
    Re-index an already downloaded pack into ChromaDB.
    Use this if you downloaded a pack before indexing was added,
    or if you need to refresh the index.
    """
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise HTTPException(400, "No backup folder configured. Set it in Settings first.")

    source_id = request.source_id
    source_folder = Path(backup_folder) / source_id

    if not source_folder.exists():
        raise HTTPException(404, f"Source folder not found: {source_id}")

    # Look for index file
    index_path = source_folder / f"{source_id}_index.json"
    if not index_path.exists():
        raise HTTPException(404, f"Index file not found: {source_id}_index.json")

    try:
        from vectordb import get_vector_store

        # Load the index file
        with open(index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        documents = index_data.get("documents", [])
        if not documents:
            raise HTTPException(400, "Index file contains no documents")

        # Get vector store
        store = get_vector_store()

        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        contents = []
        metadatas = []

        for doc in documents:
            ids.append(doc["id"])
            embeddings.append(doc["embedding"])
            contents.append(doc.get("content", ""))
            metadatas.append(doc.get("metadata", {}))

        # Add to ChromaDB in batches
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_contents = contents[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]

            store.collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_contents,
                metadatas=batch_metadatas
            )

        # Update installed packs
        installed = config.get("installed_packs", [])
        if source_id not in installed:
            installed.append(source_id)
            config.set("installed_packs", installed)
            config.save()

        return {
            "status": "success",
            "source_id": source_id,
            "indexed_documents": len(ids),
            "message": f"Re-indexed {len(ids)} documents for {source_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Re-index failed: {str(e)}")


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

    # Check for backup folder and files
    source_path = Path(backup_folder) / request.source_id

    # Detect source type - check for ZIM first (in root or source folder)
    zim_path = Path(backup_folder) / f"{request.source_id}.zim"
    zim_in_folder = None
    if source_path.exists():
        zim_files = list(source_path.glob("*.zim"))
        if zim_files:
            zim_in_folder = zim_files[0]

    # Check for HTML files
    has_html = source_path.exists() and (
        list(source_path.glob("*.html")) or
        (source_path / "pages").exists() or
        (source_path / f"{request.source_id}_backup_manifest.json").exists() or
        (source_path / "manifest.json").exists()
    )

    # Check for PDF collection
    collection_file = source_path / "_collection.json" if source_path.exists() else None
    has_pdf = collection_file and collection_file.exists()

    if has_pdf:
        # PDF collection - metadata comes from _collection.json
        from datetime import datetime
        try:
            with open(collection_file, 'r', encoding='utf-8') as f:
                collection_data = json.load(f)
        except Exception as e:
            raise HTTPException(500, f"Failed to read _collection.json: {e}")

        documents = collection_data.get("documents", {})
        collection_info = collection_data.get("collection", {})

        # Calculate total chars from documents
        total_chars = sum(doc.get("char_count", 0) for doc in documents.values())

        metadata = {
            "version": 2,
            "source_id": request.source_id,
            "source_type": "pdf",
            "last_updated": datetime.now().isoformat(),
            "total_documents": len(documents),
            "document_count": len(documents),
            "total_chars": total_chars,
            "collection_name": collection_info.get("name", request.source_id),
            "pdf_info": {
                "collection_file": str(collection_file),
                "document_count": len(documents),
                "note": "PDF collection with pre-extracted text chunks"
            }
        }
        from sourcepacks.pack_tools import save_metadata
        save_metadata(request.source_id, metadata)

        return {
            "status": "success",
            "source_id": request.source_id,
            "document_count": len(documents),
            "total_chars": total_chars,
            "source_type": "pdf",
            "message": f"PDF collection: {len(documents)} documents, {total_chars:,} characters"
        }

    elif zim_path.exists() or zim_in_folder:
        # ZIM found - use the actual path
        actual_zim_path = zim_in_folder if zim_in_folder else zim_path
        from datetime import datetime

        # Analyze ZIM to get real article count
        zim_info = _analyze_zim_file(actual_zim_path, request.source_id)
        article_count = zim_info.get("article_count", 0)

        metadata = {
            "version": 2,
            "source_id": request.source_id,
            "source_type": "zim",
            "last_updated": datetime.now().isoformat(),
            "total_documents": article_count,
            "document_count": article_count,
            "total_chars": 0,
            "zim_info": {
                "path": str(actual_zim_path),
                "article_count": article_count,
                "note": "Use Create Index to extract and index content for search."
            }
        }
        from sourcepacks.pack_tools import save_metadata
        save_metadata(request.source_id, metadata)

        return {
            "status": "success",
            "source_id": request.source_id,
            "document_count": article_count,
            "total_chars": 0,
            "source_type": "zim",
            "zim_path": str(actual_zim_path),
            "message": f"ZIM contains {article_count:,} articles. Use Create Index to extract content."
        }

    elif has_html:
        # HTML backup
        try:
            metadata = generate_metadata_from_html(
                backup_path=str(source_path),
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

    else:
        raise HTTPException(404, f"No backup found for {request.source_id} in {backup_folder}. Expected HTML files, .zim file, or _collection.json for PDFs.")


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


# =============================================================================
# SOURCE TOOLS - Edit, index, and manage local sources
# =============================================================================

@router.get("/source-tools", response_class=HTMLResponse)
async def source_tools_page(request: Request):
    """Source tools page for editing and managing sources"""
    config = get_local_config()

    return templates.TemplateResponse("source_tools.html", {
        "request": request,
        "config": config.config
    })


class UpdateSourceConfigRequest(BaseModel):
    source_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    base_url: Optional[str] = None
    tags: Optional[List[str]] = None
    license_verified: Optional[bool] = None


@router.post("/api/update-source-config")
async def update_source_config(request: UpdateSourceConfigRequest):
    """Update source configuration in {source_id}_source.json"""
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise HTTPException(400, "No backup folder configured")

    source_file = Path(backup_folder) / request.source_id / f"{request.source_id}_source.json"

    # Load existing source config or create new one
    if source_file.exists():
        try:
            with open(source_file, 'r', encoding='utf-8') as f:
                source = json.load(f)
        except Exception:
            source = {"source_id": request.source_id}
    else:
        # Check if source folder exists at all
        source_folder = Path(backup_folder) / request.source_id
        if not source_folder.exists():
            raise HTTPException(404, f"Source folder not found: {request.source_id}")
        source = {"schema_version": 2, "source_id": request.source_id}

    # Update fields that were provided
    if request.name is not None:
        source["name"] = request.name
    if request.description is not None:
        source["description"] = request.description
    if request.license is not None:
        source["license"] = request.license
    if request.base_url is not None:
        source["base_url"] = request.base_url
    if request.tags is not None:
        source["tags"] = request.tags
    if request.license_verified is not None:
        source["license_verified"] = request.license_verified

    # Save to _source.json
    try:
        source_file.parent.mkdir(parents=True, exist_ok=True)
        with open(source_file, 'w', encoding='utf-8') as f:
            json.dump(source, f, indent=2)
        return {"status": "success", "source_id": request.source_id}
    except Exception as e:
        raise HTTPException(500, f"Failed to save source config: {e}")


@router.get("/api/detect-license/{source_id}")
async def detect_license(source_id: str):
    """Auto-detect license from source content"""
    try:
        from sourcepacks.source_manager import SourceManager

        manager = SourceManager()
        detected = manager.detect_license(source_id)

        if detected:
            return {
                "detected_license": detected,
                "confidence": "medium",
                "source_id": source_id
            }
        else:
            return {
                "detected_license": None,
                "message": "Could not detect license from content"
            }
    except Exception as e:
        raise HTTPException(500, f"License detection failed: {e}")


@router.get("/api/auto-detect/{source_id}")
async def auto_detect(source_id: str):
    """Auto-detect license, description, tags, and URL from source content"""
    try:
        from sourcepacks.source_manager import SourceManager

        config = get_local_config()
        backup_folder = config.get_backup_folder()

        result = {
            "source_id": source_id,
            "license": None,
            "description": None,
            "tags": [],
            "base_url": None
        }

        manager = SourceManager()

        # Detect license
        detected_license = manager.detect_license(source_id)
        if detected_license:
            result["license"] = detected_license

        # Suggest tags
        suggested_tags = manager.suggest_tags(source_id)
        if suggested_tags:
            result["tags"] = suggested_tags

        # Try to get description and URL from metadata or manifest
        if backup_folder:
            source_folder = Path(backup_folder) / source_id

            # Check for source metadata (v2)
            source_file = source_folder / f"{source_id}_source.json"
            if source_file.exists():
                try:
                    with open(source_file, 'r', encoding='utf-8') as f:
                        source_meta = json.load(f)
                    if source_meta.get("description"):
                        result["description"] = source_meta["description"]
                    if source_meta.get("base_url"):
                        result["base_url"] = source_meta["base_url"]
                except Exception:
                    pass

            # Check legacy metadata
            if not result["description"]:
                meta_file = source_folder / f"{source_id}_metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        if meta.get("description"):
                            result["description"] = meta["description"]
                        if meta.get("base_url"):
                            result["base_url"] = meta["base_url"]
                    except Exception:
                        pass

            # Check backup manifest for base_url
            if not result["base_url"]:
                manifest_file = source_folder / f"{source_id}_backup_manifest.json"
                if not manifest_file.exists():
                    manifest_file = source_folder / "manifest.json"
                if manifest_file.exists():
                    try:
                        with open(manifest_file, 'r', encoding='utf-8') as f:
                            manifest = json.load(f)
                        if manifest.get("base_url"):
                            result["base_url"] = manifest["base_url"]
                    except Exception:
                        pass

            # Check distribution manifest
            if not result["base_url"]:
                dist_manifest = source_folder / f"{source_id}_manifest.json"
                if dist_manifest.exists():
                    try:
                        with open(dist_manifest, 'r', encoding='utf-8') as f:
                            manifest = json.load(f)
                        if manifest.get("base_url"):
                            result["base_url"] = manifest["base_url"]
                    except Exception:
                        pass

            # For ZIM files, try to derive URL from filename and analyze content
            if not result["base_url"] or not result["tags"]:
                zim_files = list(source_folder.glob("*.zim"))
                if zim_files:
                    zim_info = _analyze_zim_file(zim_files[0], source_id)
                    if not result["base_url"] and zim_info.get("base_url"):
                        result["base_url"] = zim_info["base_url"]
                    if not result["tags"] and zim_info.get("tags"):
                        result["tags"] = zim_info["tags"]
                    if not result["description"] and zim_info.get("description"):
                        result["description"] = zim_info["description"]

        return result

    except Exception as e:
        raise HTTPException(500, f"Auto-detection failed: {e}")


class ScanBackupRequest(BaseModel):
    source_id: str


@router.post("/api/scan-backup")
async def scan_backup(request: ScanBackupRequest):
    """
    Scan an existing backup folder and create a backup manifest.

    Use this when you have HTML files in the pages/ folder but no
    proper backup manifest (e.g., after downloading from R2).
    """
    try:
        from sourcepacks.source_manager import SourceManager

        manager = SourceManager()
        result = manager.scan_backup(request.source_id)

        if result.get("success"):
            return {
                "status": "success",
                "source_id": request.source_id,
                "page_count": result.get("page_count", 0),
                "total_size_mb": result.get("total_size_mb", 0),
                "manifest_path": result.get("manifest_path"),
                "message": result.get("message", "Backup scanned successfully")
            }
        else:
            raise HTTPException(400, result.get("error", "Scan failed"))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Scan failed: {e}")


class CreateIndexRequest(BaseModel):
    source_id: str
    force: bool = False
    limit: int = 1000


def _run_create_index(source_id: str, limit: int, force: bool, progress_callback=None):
    """Background worker function for index creation."""
    from sourcepacks.source_manager import SourceManager

    if progress_callback:
        progress_callback(5, "Initializing indexer...")

    manager = SourceManager()
    skip_existing = not force

    if progress_callback:
        progress_callback(10, "Loading documents...")

    # Create adapter to convert (current, total, msg) to (percent, msg)
    def indexer_progress(current, total, message):
        if progress_callback and total > 0:
            # Scale from 10-95% (leaving room for init and finalization)
            percent = 10 + int((current / total) * 85)
            progress_callback(percent, message)

    result = manager.create_index(
        source_id,
        limit=limit,
        skip_existing=skip_existing,
        progress_callback=indexer_progress
    )

    if progress_callback:
        progress_callback(100, "Complete")

    if result.success:
        return {
            "status": "success",
            "source_id": source_id,
            "indexed_count": result.indexed_count,
            "total_chars": result.total_chars,
            "message": f"Indexed {result.indexed_count} documents"
        }
    else:
        raise Exception(result.error or "Index creation failed")


@router.post("/api/create-index")
async def create_index(request: CreateIndexRequest):
    """Create embeddings index for a source (v2 format) - runs as background job"""
    try:
        config = get_local_config()
        backup_folder = config.get_backup_folder()

        if not backup_folder:
            raise HTTPException(400, "No backup folder configured")

        # Submit as background job
        from .job_manager import get_job_manager
        manager = get_job_manager()

        try:
            job_id = manager.submit(
                "index",
                request.source_id,
                _run_create_index,
                request.source_id,
                request.limit,
                request.force
            )
        except ValueError as e:
            # Job already running for this source
            raise HTTPException(409, str(e))

        return {
            "status": "submitted",
            "job_id": job_id,
            "source_id": request.source_id,
            "message": f"Index job started for {request.source_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to start index job: {e}")


class ValidateSourceRequest(BaseModel):
    source_id: str
    require_v2: bool = False


@router.post("/api/validate-source")
async def validate_source(request: ValidateSourceRequest):
    """Validate source for distribution"""
    try:
        from sourcepacks.source_manager import SourceManager
        import json

        config = get_local_config()
        backup_folder = config.get_backup_folder()

        # Load source config from {source_id}_source.json (primary source of truth)
        source_config = {}
        if backup_folder:
            source_file = Path(backup_folder) / request.source_id / f"{request.source_id}_source.json"
            if source_file.exists():
                try:
                    with open(source_file, 'r', encoding='utf-8') as f:
                        source_config = json.load(f)
                except Exception:
                    pass

        manager = SourceManager()

        if request.require_v2:
            result = manager.validate_for_production(
                request.source_id,
                source_config,
                require_v2=True
            )
        else:
            result = manager.validate_source(request.source_id, source_config)

        # Convert dataclass to dict
        from dataclasses import asdict
        validation_dict = asdict(result)

        return {
            "status": "success",
            "source_id": request.source_id,
            "validation": validation_dict
        }

    except Exception as e:
        raise HTTPException(500, f"Validation failed: {e}")


class CleanupRequest(BaseModel):
    source_id: str


@router.post("/api/cleanup-redundant-files")
async def cleanup_redundant_files(request: CleanupRequest):
    """Delete redundant legacy files that have v2 equivalents"""
    try:
        from sourcepacks.source_manager import SourceManager

        manager = SourceManager()
        result = manager.cleanup_redundant_files(request.source_id)

        if result["errors"]:
            return {
                "status": "partial",
                "source_id": request.source_id,
                "deleted": result["deleted"],
                "errors": result["errors"],
                "freed_mb": result["freed_mb"]
            }

        return {
            "status": "success",
            "source_id": request.source_id,
            "deleted": result["deleted"],
            "freed_mb": result["freed_mb"],
            "message": f"Cleaned up {len(result['deleted'])} file(s), freed {result['freed_mb']} MB"
        }

    except Exception as e:
        raise HTTPException(500, f"Cleanup failed: {e}")


class DeleteSourceRequest(BaseModel):
    source_id: str
    delete_files: bool = True  # Also delete local backup files


@router.post("/api/delete-source")
async def delete_source(request: DeleteSourceRequest):
    """
    Completely remove a source from the system.

    - Deletes local backup folder (which is the source of truth)
    - Removes from installed_packs list
    - Removes from _master.json
    - Removes from ChromaDB
    """
    import shutil

    source_id = request.source_id
    deleted_items = []
    errors = []
    config = get_local_config()

    # 1. Remove from installed_packs in local config
    try:
        installed = config.get("installed_packs", [])
        if source_id in installed:
            installed.remove(source_id)
            config.set("installed_packs", installed)
            config.save()
            deleted_items.append("installed_packs entry")
    except Exception as e:
        errors.append(f"Failed to remove from installed_packs: {e}")

    # 2. Delete local backup folder if requested
    freed_mb = 0
    if request.delete_files:
        try:
            backup_folder = config.get_backup_folder()
            if backup_folder:
                source_folder = Path(backup_folder) / source_id
                if source_folder.exists():
                    # Calculate size before deleting
                    try:
                        freed_mb = sum(f.stat().st_size for f in source_folder.rglob('*') if f.is_file()) / (1024*1024)
                    except:
                        pass
                    shutil.rmtree(source_folder)
                    deleted_items.append(f"backup folder ({round(freed_mb, 2)} MB)")
        except Exception as e:
            errors.append(f"Failed to delete backup folder: {e}")

    # 3. Remove from _master.json
    try:
        backup_folder = config.get_backup_folder()
        if backup_folder:
            master_file = Path(backup_folder) / "_master.json"
            if master_file.exists():
                import json
                with open(master_file, 'r', encoding='utf-8') as f:
                    master_data = json.load(f)

                if source_id in master_data.get("sources", {}):
                    # Update totals before removing
                    source_info = master_data["sources"][source_id]
                    master_data["total_documents"] = master_data.get("total_documents", 0) - source_info.get("count", 0)
                    master_data["total_chars"] = master_data.get("total_chars", 0) - source_info.get("chars", 0)

                    del master_data["sources"][source_id]
                    master_data["last_updated"] = datetime.now().isoformat()

                    with open(master_file, 'w', encoding='utf-8') as f:
                        json.dump(master_data, f, indent=2)

                    deleted_items.append("_master.json entry")
    except Exception as e:
        errors.append(f"Failed to remove from _master.json: {e}")

    # 4. Try to remove from ChromaDB if it exists
    try:
        from vectordb import get_vector_store
        store = get_vector_store()
        # Delete all documents with this source_id
        results = store.collection.get(where={"source_id": source_id})
        if results and results.get("ids"):
            store.collection.delete(ids=results["ids"])
            deleted_items.append(f"ChromaDB entries ({len(results['ids'])} docs)")
    except Exception as e:
        # ChromaDB deletion is optional, don't treat as error
        pass

    if errors and not deleted_items:
        raise HTTPException(500, f"Delete failed: {'; '.join(errors)}")

    return {
        "status": "success" if not errors else "partial",
        "source_id": source_id,
        "deleted": deleted_items,
        "freed_mb": round(freed_mb, 2),
        "errors": errors if errors else None,
        "message": f"Deleted {source_id}: {', '.join(deleted_items)}"
    }


# =============================================================================
# JOB MANAGEMENT ENDPOINTS
# =============================================================================

from .job_manager import get_job_manager


@router.get("/api/jobs")
async def get_all_jobs(limit: int = 20):
    """Get all jobs (recent history)."""
    manager = get_job_manager()
    return {
        "jobs": manager.get_all_jobs(limit=limit)
    }


@router.get("/api/jobs/active")
async def get_active_jobs(source_id: str = None):
    """Get currently active (pending/running) jobs."""
    manager = get_job_manager()
    return {
        "jobs": manager.get_active_jobs(source_id=source_id)
    }


@router.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job."""
    manager = get_job_manager()
    status = manager.get_status(job_id)
    if not status:
        raise HTTPException(404, f"Job {job_id} not found")
    return status


@router.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Request cancellation of a job."""
    manager = get_job_manager()
    if manager.cancel(job_id):
        return {"status": "cancelled", "job_id": job_id}
    raise HTTPException(400, f"Could not cancel job {job_id}")
