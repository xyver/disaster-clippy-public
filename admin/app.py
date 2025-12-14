"""
Local User Admin Panel Routes
FastAPI router for managing local user settings and offline content
"""

from fastapi import APIRouter, Request, HTTPException, Depends
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
from offline_tools.packager import (
    load_master_metadata,
    get_source_sync_status
)

# Schema file naming
from offline_tools.schemas import get_manifest_file, get_metadata_file
from offline_tools.validation import SYSTEM_FOLDERS


# =============================================================================
# VECTOR_DB_MODE - Controls deployment mode and access levels
# =============================================================================
#
# VECTOR_DB_MODE has 3 values:
#   - local:    Admin UI visible, local ChromaDB, R2 backups read + submissions R/W
#   - pinecone: Admin UI blocked (public mode), Pinecone cloud search only
#   - global:   Admin UI visible, Pinecone R/W, R2 full access
#

def get_vector_db_mode() -> str:
    """
    Get the current deployment mode from VECTOR_DB_MODE environment variable.

    Returns:
        "local" - Local admin with ChromaDB (default)
        "pinecone" - Public deployment with cloud search only
        "global" - Global admin with full cloud write access
    """
    return os.getenv("VECTOR_DB_MODE", "local").lower()


def is_public_mode() -> bool:
    """
    Check if running in public mode (VECTOR_DB_MODE=pinecone).
    In public mode, admin UI is blocked - only chat is available.
    """
    return get_vector_db_mode() == "pinecone"


def block_in_public_mode():
    """
    Dependency that blocks access to admin routes in public mode.
    Use on Railway to prevent unauthorized access to admin features.
    """
    if is_public_mode():
        raise HTTPException(
            status_code=404,
            detail="Not found"
        )
    return True


def get_admin_mode() -> str:
    """
    Get the admin mode based on VECTOR_DB_MODE.
    For backwards compatibility with code that checks admin mode.

    Returns:
        "global" if VECTOR_DB_MODE=global
        "local" otherwise
    """
    mode = get_vector_db_mode()
    return "global" if mode == "global" else "local"


def is_global_admin() -> bool:
    """Check if running in global admin mode (VECTOR_DB_MODE=global)"""
    return get_vector_db_mode() == "global"


def require_global_admin():
    """
    Dependency that blocks access unless VECTOR_DB_MODE=global.
    Use this to protect endpoints that write to the shared cloud.
    """
    if not is_global_admin():
        raise HTTPException(
            status_code=403,
            detail="This feature requires global admin access. Set VECTOR_DB_MODE=global to enable."
        )
    return True


# =============================================================================
# ROUTER SETUP
# =============================================================================

# Include sub-routers
from .cloud_upload import router as cloud_upload_router
from .routes.source_tools import router as source_tools_router
from .routes.packs import router as packs_router
from .routes.jobs import router as jobs_router
from .routes.visualise import router as visualise_router
from .routes.models import router as models_router
from .routes.job_builder import router as job_builder_router

# Create router with public mode check - blocks all routes when VECTOR_DB_MODE=pinecone
router = APIRouter(
    prefix="/useradmin",
    tags=["User Admin"],
    dependencies=[Depends(block_in_public_mode)]
)
router.include_router(cloud_upload_router)
router.include_router(source_tools_router)
router.include_router(packs_router)
router.include_router(jobs_router)
router.include_router(visualise_router)
router.include_router(models_router)
router.include_router(job_builder_router)


# =============================================================================
# ADMIN MODE API
# =============================================================================

@router.get("/api/admin-mode")
async def get_admin_mode_status():
    """
    Get current admin mode and available features.
    Frontend uses this to show/hide global-only UI elements.
    """
    raw_mode = os.getenv("VECTOR_DB_MODE", "NOT_SET")
    mode = get_admin_mode()
    print(f"[DEBUG admin-mode] VECTOR_DB_MODE raw='{raw_mode}', get_admin_mode()='{mode}'")
    return {
        "mode": mode,
        "is_global": mode == "global",
        "features": {
            "cloud_download": True,  # Always available (read-only)
            "cloud_upload": mode == "global",  # Write to backups/
            "pinecone_sync": mode == "global",  # Write to shared vectors
            "submissions": mode == "global",  # Approve community submissions
            "local_backup": True,  # Always available
            "local_index": True,  # Always available
        }
    }


# =============================================================================
# ZIM ANALYSIS HELPERS
# =============================================================================

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


def get_template_context(request, config=None):
    """
    Build common template context including admin_mode for CSS selection.

    Returns dict with:
        - request: FastAPI request object
        - config: local config dict (optional)
        - admin_mode: "local" or "global" based on VECTOR_DB_MODE env var
        - admin_css: CSS filename to load based on mode
    """
    mode = get_admin_mode()
    ctx = {
        "request": request,
        "admin_mode": mode,
        "admin_css": f"{mode}-admin.css"
    }
    if config is not None:
        ctx["config"] = config
    return ctx


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
    prompts: Optional[Dict[str, str]] = None  # {"online": "...", "offline": "..."}
    personal_cloud: Optional[Dict[str, Any]] = None  # Personal cloud storage config
    models: Optional[Dict[str, Any]] = None  # {"embedding_model": "all-mpnet-base-v2", ...}


# Routes

@router.get("/", response_class=HTMLResponse)
async def admin_home(request: Request):
    """Main admin panel page"""
    config = get_local_config()
    internet_available = check_internet_available()
    ctx = get_template_context(request, config.config)
    ctx["internet_available"] = internet_available
    return templates.TemplateResponse("settings.html", ctx)


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

    if updates.prompts is not None:
        for mode, prompt in updates.prompts.items():
            if mode in ["online", "offline", "system"]:
                config.set_prompt(mode, prompt)

    if updates.personal_cloud is not None:
        config.set_personal_cloud_config(**updates.personal_cloud)

    if updates.models is not None:
        # Handle model settings (embedding_model, llm_model, etc.)
        if "embedding_model" in updates.models:
            config.set_embedding_model(updates.models["embedding_model"])
        if "llm_model" in updates.models:
            config.set("models.llm_model", updates.models["llm_model"])

    if config.save():
        return {"status": "success", "settings": config.config}
    else:
        raise HTTPException(status_code=500, detail="Failed to save settings")


@router.post("/api/test-cloud-connection")
async def test_cloud_connection(config_data: Dict[str, Any]):
    """Test connection to personal cloud storage without saving"""
    try:
        from offline_tools.cloud.r2 import R2Config, R2Storage

        # Create temporary R2Config from provided settings
        temp_config = R2Config(
            access_key_id=config_data.get("access_key_id", ""),
            secret_access_key=config_data.get("secret_access_key", ""),
            endpoint_url=config_data.get("endpoint_url", ""),
            bucket_name=config_data.get("bucket_name", ""),
            token_expires=None
        )

        # Create temporary storage instance
        temp_storage = R2Storage(config=temp_config)

        # Test connection
        result = temp_storage.test_connection()

        return result

    except Exception as e:
        return {
            "configured": False,
            "connected": False,
            "bucket_exists": False,
            "error": str(e)
        }


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


@router.post("/api/reset-prompt/{mode}")
async def reset_prompt(mode: str):
    """Reset a prompt to its default value"""
    if mode not in ["online", "offline"]:
        raise HTTPException(status_code=400, detail="Mode must be 'online' or 'offline'")

    config = get_local_config()
    default_prompt = config.reset_prompt(mode)

    if config.save():
        return {"status": "success", "mode": mode, "prompt": default_prompt}
    else:
        raise HTTPException(status_code=500, detail="Failed to save settings")


@router.get("/api/prompts")
async def get_prompts():
    """Get current prompts"""
    config = get_local_config()
    return {
        "prompts": config.get_prompts(),
        "defaults": config.DEFAULT_CONFIG["prompts"]
    }


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
    """Get current system status including dashboard stats"""
    config = get_local_config()
    paths = config.get_backup_paths()
    backup_folder = config.get_backup_folder()

    # Count available backup files
    zim_count = len(scan_backup_folder(paths.get("zim_folder", ""), "zim"))
    html_count = len(scan_backup_folder(paths.get("html_folder", ""), "html"))
    pdf_count = len(scan_backup_folder(paths.get("pdf_folder", ""), "pdf"))

    # Count local sources and documents for dashboard
    # Use _master.json header for fast stats (avoids loading large files)
    local_sources = 0
    total_documents = 0
    storage_bytes = 0

    if backup_folder:
        backup_path = Path(backup_folder)
        master_file = backup_path / "_master.json"

        # Fast path: read header from _master.json
        if master_file.exists():
            from offline_tools.packager import read_json_header_only
            header = read_json_header_only(master_file)
            local_sources = header.get("source_count", 0)
            total_documents = header.get("total_documents", 0)
            storage_bytes = header.get("total_size_bytes", 0)

        # Fallback: scan folders if _master.json not available or empty
        # Skip system folders (chroma_db, models, etc.)
        if local_sources == 0 and backup_path.exists():
            from offline_tools.packager import read_json_header_only
            for source_folder in backup_path.iterdir():
                if source_folder.is_dir() and not source_folder.name.startswith("_"):
                    if source_folder.name.lower() in SYSTEM_FOLDERS:
                        continue
                    manifest_file = source_folder / get_manifest_file()
                    if manifest_file.exists():
                        local_sources += 1
                        # Fast header reading for metadata
                        metadata_file = source_folder / get_metadata_file()
                        if metadata_file.exists():
                            header = read_json_header_only(metadata_file)
                            total_documents += header.get("document_count", 0)
                            storage_bytes += header.get("total_chars", 0)

    storage_mb = round(storage_bytes / (1024 * 1024), 1) if storage_bytes > 0 else 0

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
        },
        # Dashboard stats
        "local_sources": local_sources,
        "total_documents": total_documents,
        "storage_mb": storage_mb
    }


@router.get("/api/chromadb-status")
async def get_chromadb_status():
    """Get local ChromaDB stats for dashboard"""
    try:
        from offline_tools.vectordb import get_vector_store
        # Use read_only=True to skip embedding model initialization
        store = get_vector_store(mode="local", read_only=True)
        stats = store.get_stats()

        return {
            "connected": True,
            "total_vectors": stats.get("total_documents", 0),
            "collection_name": stats.get("collection_name", "articles"),
            "sources": stats.get("sources", {}),
            "source_count": len(stats.get("sources", {}))
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
            "total_vectors": 0
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
    ctx = get_template_context(request, config.config)
    ctx["internet_available"] = internet_available
    return templates.TemplateResponse("packs.html", ctx)


# =============================================================================
# NOTE: Pack Management API endpoints moved to admin/routes/packs.py
# Includes: install-pack, installed-packs, cloud-sources, available-packs,
#           download-pack, reindex-pack
# =============================================================================

import json


# =============================================================================
# NOTE: Legacy indexing API endpoints have been removed and replaced with
# the new Source Tools API in admin/routes/source_tools.py:
# - /api/generate-metadata -> uses SourceManager.generate_metadata()
# - /api/index-source -> replaced by /api/create-index (background job)
# - /api/source-status/{source_id} -> removed (not used)
# - /api/export-index/{source_id} -> removed (not used)
# =============================================================================


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


# =============================================================================
# NOTE: Source Tools API endpoints moved to admin/routes/source_tools.py
# Includes: local-sources, update-source-config, auto-detect, create-source,
#           scan-backup, generate-metadata, create-index, validate-source,
#           cleanup-redundant-files, delete-source, rename-source
# =============================================================================


# =============================================================================
# NOTE: Job Management API endpoints moved to admin/routes/jobs.py
# Includes: jobs, jobs/active, jobs/{job_id}, jobs/{job_id}/cancel
# =============================================================================


# =============================================================================
# PAGE ROUTES - Admin UI Pages
# =============================================================================

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Dashboard - overview of system status"""
    config = get_local_config()
    return templates.TemplateResponse("dashboard.html", get_template_context(request, config.config))


@router.get("/sources", response_class=HTMLResponse)
async def sources_page(request: Request):
    """Sources - manage content sources"""
    config = get_local_config()
    return templates.TemplateResponse("sources.html", get_template_context(request, config.config))


@router.get("/sources/tools", response_class=HTMLResponse)
async def sources_tools_page(request: Request):
    """Source Tools - indexing, packaging, validation"""
    config = get_local_config()
    return templates.TemplateResponse("source_tools.html", get_template_context(request, config.config))


@router.get("/sources/create")
async def sources_create_page():
    """Redirect to Source Tools - create is now integrated there"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/useradmin/sources/tools", status_code=302)


@router.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    """Jobs - background job manager"""
    config = get_local_config()
    return templates.TemplateResponse("jobs.html", get_template_context(request, config.config))


@router.get("/job-builder", response_class=HTMLResponse)
async def job_builder_page(request: Request):
    """Job Builder - create custom job chains"""
    config = get_local_config()
    return templates.TemplateResponse("job_builder.html", get_template_context(request, config.config))


@router.get("/cloud", response_class=HTMLResponse)
async def cloud_page(request: Request):
    """Cloud - R2 and cloud storage management"""
    config = get_local_config()
    return templates.TemplateResponse("cloud_upload.html", get_template_context(request, config.config))


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings - app configuration"""
    config = get_local_config()
    internet_available = check_internet_available()
    ctx = get_template_context(request, config.config)
    ctx["internet_available"] = internet_available
    return templates.TemplateResponse("settings.html", ctx)


@router.get("/submissions", response_class=HTMLResponse)
async def submissions_page(request: Request):
    """Submissions - review incoming source submissions"""
    config = get_local_config()
    return templates.TemplateResponse("submissions.html", get_template_context(request, config.config))


@router.get("/pinecone", response_class=HTMLResponse)
async def pinecone_page(request: Request):
    """Pinecone - vector database settings and sync"""
    config = get_local_config()
    return templates.TemplateResponse("pinecone.html", get_template_context(request, config.config))


@router.get("/visualise", response_class=HTMLResponse)
async def visualise_page(request: Request):
    """Knowledge Map - 3D visualization of document embeddings"""
    config = get_local_config()
    ctx = get_template_context(request, config.config)
    ctx["is_admin"] = True
    return templates.TemplateResponse("visualise.html", ctx)
