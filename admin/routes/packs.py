"""
Pack Management API

Endpoints for browsing, installing, and downloading source packs.
Handles both the shared source-pack catalog and local pack state.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import os

from offline_tools.schemas import (
    get_manifest_file, get_metadata_file, get_index_file, get_vectors_file,
    get_vectors_768_file
)
from offline_tools.validation import is_system_folder

router = APIRouter(prefix="/api", tags=["Pack Management"])


# =============================================================================
# CACHE FOR R2 MASTER.JSON - Avoid downloading on every page load
# =============================================================================

_cloud_master_cache = {
    "data": None,
    "timestamp": 0,
    "ttl": 300  # 5 minutes
}


def _get_cached_cloud_master(storage) -> dict:
    """
    Get _master.json from R2 with caching.
    Returns cached data if still valid, otherwise downloads fresh copy.
    """
    import time
    import tempfile

    now = time.time()
    cache = _cloud_master_cache

    # Return cached data if still valid
    if cache["data"] is not None and (now - cache["timestamp"]) < cache["ttl"]:
        return cache["data"]

    # Download fresh copy
    tmp_path = Path(tempfile.gettempdir()) / "cloud_master.json"
    if storage.download_file("backups/_master.json", str(tmp_path)):
        try:
            with open(tmp_path, 'r', encoding='utf-8') as f:
                cache["data"] = json.load(f)
                cache["timestamp"] = now
                return cache["data"]
        except Exception as e:
            print(f"Failed to parse _master.json: {e}")

    return {}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_local_config():
    """Get local config - imported here to avoid circular imports"""
    from admin.local_config import get_local_config as _get_config
    return _get_config()


def _normalize_source_id(source_id: str) -> str:
    return source_id.strip().lower()


def _require_backup_path(config) -> Path:
    return Path(config.require_backup_folder())


def _list_installed_source_ids(config) -> List[str]:
    try:
        backup_path = _require_backup_path(config)
    except Exception:
        return []

    if not backup_path.exists():
        return []

    source_ids = []
    for source_folder in backup_path.iterdir():
        if source_folder.is_dir() and not source_folder.name.startswith("_") and not is_system_folder(source_folder.name):
            source_ids.append(source_folder.name)

    return sorted(source_ids, key=str.lower)


def _sync_installed_packs(config, installed_ids: List[str]) -> None:
    normalized = [_normalize_source_id(source_id) for source_id in installed_ids]
    if config.get("installed_packs", []) != normalized:
        config.set("installed_packs", normalized)
        config.save()


def _get_active_source_ids(config, installed_ids: List[str]) -> Tuple[List[str], str]:
    installed_set = {_normalize_source_id(source_id) for source_id in installed_ids}
    selected = [_normalize_source_id(source_id) for source_id in config.get_selected_sources()]

    if not selected:
        return sorted(installed_set), "all_installed"

    active = [source_id for source_id in selected if source_id in installed_set]
    if active != selected:
        config.set_selected_sources(active)
        config.save()

    return active, "selected"


def _read_live_catalog(config) -> Dict[str, Any]:
    if config.should_use_proxy():
        from admin.cloud_proxy import get_proxy_client

        proxy = get_proxy_client()
        result = proxy.get_catalog()
        if "error" in result and not result.get("connected", False):
            return {"catalog": {}, "connected": False, "error": result["error"], "via": "proxy"}

        catalog = result.get("catalog", {})
        return {
            "catalog": catalog if isinstance(catalog, dict) else {},
            "connected": True,
            "via": "proxy"
        }

    from dotenv import load_dotenv
    load_dotenv()
    from offline_tools.cloud.r2 import get_backups_storage

    storage = get_backups_storage()
    if not storage.is_configured():
        return {"catalog": {}, "connected": False, "error": "R2 not configured"}

    conn_status = storage.test_connection()
    if not conn_status["connected"]:
        return {"catalog": {}, "connected": False, "error": conn_status.get("error", "Connection failed")}

    raw = storage.download_file_content("published/catalog.json")
    if not raw:
        return {"catalog": {}, "connected": False, "error": "published/catalog.json not found"}

    try:
        catalog = json.loads(raw)
    except Exception as e:
        return {"catalog": {}, "connected": False, "error": f"Failed to parse published/catalog.json: {e}"}

    return {"catalog": catalog if isinstance(catalog, dict) else {}, "connected": True, "via": "r2"}


def load_catalog_packs_data() -> Dict[str, Any]:
    config = get_local_config()
    catalog_result = _read_live_catalog(config)
    if not catalog_result.get("connected"):
        return {"packs": [], "connected": False, "error": catalog_result.get("error", "Catalog unavailable")}

    catalog = catalog_result.get("catalog", {})
    catalog_sources = catalog.get("sources", [])
    installed_ids = _list_installed_source_ids(config)
    _sync_installed_packs(config, installed_ids)
    active_ids, selection_mode = _get_active_source_ids(config, installed_ids)
    installed_set = {_normalize_source_id(source_id) for source_id in installed_ids}
    active_set = {_normalize_source_id(source_id) for source_id in active_ids}

    packs = []
    for source in catalog_sources if isinstance(catalog_sources, list) else []:
        if not isinstance(source, dict):
            continue
        source_id = _normalize_source_id(str(source.get("source_id", "") or ""))
        if not source_id:
            continue

        size_bytes = int(source.get("size_bytes", 0) or 0)
        packs.append({
            "source_id": source_id,
            "name": source.get("name", source_id.replace("-", " ").replace("_", " ").title()),
            "description": source.get("description", ""),
            "license": source.get("license", "Unknown"),
            "license_verified": source.get("license_verified", False),
            "tags": source.get("tags", []),
            "topics": source.get("topics", []),
            "document_count": int(source.get("doc_count", 0) or 0),
            "size_bytes": size_bytes,
            "total_size_mb": round(size_bytes / (1024 * 1024), 1) if size_bytes else 0,
            "live_url": source.get("live_url", ""),
            "backup_url": source.get("backup_url", ""),
            "base_url": source.get("base_url", ""),
            "last_updated": source.get("last_updated", ""),
            "installed": source_id in installed_set,
            "active": source_id in active_set
        })

    return {
        "packs": packs,
        "connected": True,
        "total": len(packs),
        "selection_mode": selection_mode,
        "source": "published_catalog",
        "via": catalog_result.get("via", "r2")
    }


def load_installed_packs_data() -> Dict[str, Any]:
    config = get_local_config()
    installed_ids = _list_installed_source_ids(config)
    _sync_installed_packs(config, installed_ids)
    active_ids, selection_mode = _get_active_source_ids(config, installed_ids)
    active_set = {_normalize_source_id(source_id) for source_id in active_ids}

    try:
        backup_path = _require_backup_path(config)
    except Exception:
        return {"packs": [], "count": 0, "selection_mode": selection_mode}

    packs = []
    for source_id in installed_ids:
        source_config = {}
        manifest_file = backup_path / source_id / get_manifest_file()
        if manifest_file.exists():
            try:
                with open(manifest_file, 'r', encoding='utf-8-sig') as f:
                    source_config = json.load(f)
            except Exception:
                pass

        doc_count = 0
        metadata_path = backup_path / source_id / get_metadata_file()
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                doc_count = int(meta.get("document_count", 0) or 0)
            except Exception:
                pass

        packs.append({
            "source_id": source_id,
            "name": source_config.get("name", source_id),
            "license": source_config.get("license", "Unknown"),
            "description": source_config.get("description", ""),
            "document_count": doc_count,
            "installed": True,
            "active": source_id in active_set
        })

    return {"packs": packs, "count": len(packs), "selection_mode": selection_mode}


def load_active_packs_data() -> Dict[str, Any]:
    installed_result = load_installed_packs_data()
    active_packs = [pack for pack in installed_result.get("packs", []) if pack.get("active")]
    return {
        "packs": active_packs,
        "count": len(active_packs),
        "selection_mode": installed_result.get("selection_mode", "all_installed")
    }


def list_available_cloud_sources_data() -> Dict[str, Any]:
    catalog_result = load_catalog_packs_data()
    if not catalog_result.get("connected"):
        return {"sources": [], "connected": False, "error": catalog_result.get("error", "Catalog unavailable")}

    sources = []
    for pack in catalog_result.get("packs", []):
        sources.append({
            "source_id": pack["source_id"],
            "name": pack["name"],
            "description": pack.get("description", ""),
            "license": pack.get("license", "Unknown"),
            "document_count": pack.get("document_count", 0),
            "base_url": pack.get("base_url", ""),
            "installed": pack.get("installed", False),
            "active": pack.get("active", False)
        })

    return {
        "sources": sources,
        "connected": True,
        "total": len(sources),
        "selection_mode": catalog_result.get("selection_mode", "all_installed"),
        "source": catalog_result.get("source", "published_catalog"),
        "via": catalog_result.get("via", "r2")
    }


def list_uninstalled_cloud_sources_data() -> Dict[str, Any]:
    sources_result = list_available_cloud_sources_data()
    if not sources_result.get("connected"):
        return {"sources": [], "connected": False, "error": sources_result.get("error", "Catalog unavailable")}

    available_sources = [source for source in sources_result.get("sources", []) if not source.get("installed")]
    return {
        "sources": available_sources,
        "connected": True,
        "total_cloud": len(sources_result.get("sources", [])),
        "already_installed": len(sources_result.get("sources", [])) - len(available_sources)
    }


def _run_install_source_pack_job(source_id: str, include_backup: bool, sync_mode: str = "update", progress_callback=None, cancel_checker=None, job_id=None):
    from offline_tools.source_manager import install_source_from_cloud
    from offline_tools.vectordb.metadata import MetadataIndex
    from offline_tools.vectordb import get_vector_store

    deleted_count = 0
    source_id = _normalize_source_id(source_id)

    if sync_mode == "replace":
        if progress_callback:
            progress_callback(5, f"Deleting old vectors for {source_id}...")

        try:
            store = get_vector_store(mode="local", read_only=True)
            delete_result = store.delete_by_source(source_id)
            deleted_count = delete_result.get("deleted_count", 0)
        except Exception as e:
            print(f"[install] Warning: Failed to delete old vectors: {e}")

    def progress_wrapper(stage, current, total):
        if progress_callback:
            base = 10 if sync_mode == "replace" else 0
            percent = base + int((current / max(total, 1)) * (100 - base))
            progress_callback(percent, stage)

    result = install_source_from_cloud(
        source_id=source_id,
        include_backup=include_backup,
        progress_callback=progress_wrapper
    )

    if result.get("success"):
        result["sync_mode"] = sync_mode
        result["deleted_count"] = deleted_count

        try:
            metadata_index = MetadataIndex()
            metadata_index.get_stats()
        except Exception as e:
            print(f"[install] Warning: Failed to verify metadata index: {e}")

        config = get_local_config()
        installed_ids = _list_installed_source_ids(config)
        _sync_installed_packs(config, installed_ids)

    return result


def submit_install_source_pack(source_id: str, include_backup: bool = False, sync_mode: str = "update") -> Dict[str, Any]:
    from admin.job_manager import get_job_manager

    source_id = _normalize_source_id(source_id)
    manager = get_job_manager()
    job_id = manager.submit(
        "install_source",
        source_id,
        _run_install_source_pack_job,
        source_id,
        include_backup,
        sync_mode
    )

    mode_text = "Replacing" if sync_mode == "replace" else "Installing"
    return {
        "status": "submitted",
        "job_id": job_id,
        "source_id": source_id,
        "include_backup": include_backup,
        "sync_mode": sync_mode,
        "message": f"{mode_text} source pack '{source_id}' from the catalog"
    }


# =============================================================================
# REQUEST MODELS
# =============================================================================

class InstallPackRequest(BaseModel):
    source_id: str
    parent_url: str = ""


class ReindexPackRequest(BaseModel):
    source_id: str


class InstallSourceRequest(BaseModel):
    source_id: str
    include_backup: bool = False
    sync_mode: str = "update"  # "update" (add/merge) or "replace" (delete old vectors first)


class SetActivePacksRequest(BaseModel):
    source_ids: List[str] = []


# =============================================================================
# INSTALLED PACKS
# =============================================================================

@router.get("/installed-packs")
async def get_installed_packs():
    """Get list of locally installed source packs."""
    return load_installed_packs_data()


@router.get("/packs/installed")
async def get_installed_packs_canonical():
    """Canonical installed source-pack catalog for the local machine."""
    return load_installed_packs_data()


@router.get("/packs/active")
async def get_active_packs():
    """Get the active runtime source-pack catalog."""
    return load_active_packs_data()


@router.post("/packs/active")
async def set_active_packs(request: SetActivePacksRequest):
    """Set the active runtime source-pack catalog using selected_sources."""
    config = get_local_config()
    installed_ids = _list_installed_source_ids(config)
    installed_set = set(_normalize_source_id(source_id) for source_id in installed_ids)
    requested = []
    for source_id in request.source_ids:
        normalized = _normalize_source_id(source_id)
        if normalized and normalized not in requested:
            requested.append(normalized)

    invalid = [source_id for source_id in requested if source_id not in installed_set]
    if invalid:
        raise HTTPException(400, f"Cannot activate uninstalled source packs: {', '.join(invalid)}")

    config.set_selected_sources(requested)
    config.save()

    result = load_active_packs_data()
    result["message"] = "Active source-pack catalog updated"
    return result


@router.post("/install-pack")
async def install_pack(request: InstallPackRequest):
    """
    Install a source pack from the parent server.
    Downloads metadata and adds to local database.
    """
    import httpx

    config = get_local_config()
    parent_url = request.parent_url or os.getenv("PARENT_SERVER_URL", "")

    if not parent_url:
        parent_url = "http://localhost:8000"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            manifest_url = f"{parent_url}/api/v1/packs/{request.source_id}/manifest"
            response = await client.get(manifest_url)

            if response.status_code == 404:
                raise HTTPException(404, f"Pack not found: {request.source_id}")

            if response.status_code != 200:
                raise HTTPException(500, f"Failed to fetch pack: {response.status_code}")

            manifest = response.json()

            metadata_url = f"{parent_url}/api/v1/packs/{request.source_id}/metadata"
            meta_response = await client.get(metadata_url)

            if meta_response.status_code != 200:
                raise HTTPException(500, "Failed to fetch pack metadata")

            metadata = meta_response.json()

        try:
            backup_path = _require_backup_path(config)
        except Exception:
            raise HTTPException(400, "No BACKUP_PATH configured. Set it in Settings first.")

        metadata_dir = backup_path / request.source_id
        metadata_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = metadata_dir / get_metadata_file()
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        installed = config.get("installed_packs", [])
        if request.source_id not in installed:
            installed.append(request.source_id)
            config.set("installed_packs", installed)
            config.save()

        return {
            "status": "success",
            "source_id": request.source_id,
            "document_count": manifest.get("pack_info", {}).get("document_count", 0),
            "message": f"Installed source pack {request.source_id} successfully"
        }

    except httpx.RequestError as e:
        raise HTTPException(500, f"Network error: {str(e)}")


# =============================================================================
# CATALOG SOURCES (R2)
# =============================================================================

@router.get("/cloud-sources")
async def get_cloud_sources():
    """
    Get sources available from the shared catalog in R2.
    Returns sources in a format compatible with the sources page filtering.

    Uses Railway proxy if R2 keys aren't configured locally.
    """
    return list_available_cloud_sources_data()


@router.get("/available-packs")
async def get_available_packs():
    """
    Get source packs available from the shared catalog in R2.
    """
    return load_catalog_packs_data()


@router.get("/packs/catalog")
async def get_catalog_packs():
    """Canonical live source-pack catalog."""
    return load_catalog_packs_data()


@router.post("/packs/install")
async def install_catalog_pack(request: InstallSourceRequest):
    """Canonical source-pack install endpoint."""
    try:
        return submit_install_source_pack(
            request.source_id,
            include_backup=request.include_backup,
            sync_mode=request.sync_mode
        )
    except ValueError as e:
        raise HTTPException(409, str(e))


# NOTE: Download pack functionality has been consolidated into source_tools.py
# Use /useradmin/api/install-source or /useradmin/api/download-pack instead


@router.post("/reindex-pack")
async def reindex_pack(request: ReindexPackRequest):
    """
    Re-index an already downloaded pack into ChromaDB.
    Prefers 768-dim vectors for offline use.
    """
    config = get_local_config()
    try:
        backup_path = _require_backup_path(config)
    except Exception:
        raise HTTPException(400, "No BACKUP_PATH configured. Set it in Settings first.")

    source_id = request.source_id
    source_folder = backup_path / source_id

    if not source_folder.exists():
        raise HTTPException(404, f"Source folder not found: {source_id}")

    vectors_768_path = source_folder / get_vectors_768_file()
    index_path = source_folder / get_index_file()

    # Check we have at least one vector source
    if not vectors_768_path.exists() and not index_path.exists():
        raise HTTPException(404, f"No vector files found. Need {get_vectors_768_file()} or {get_index_file()}")

    try:
        from offline_tools.vectordb import get_vector_store

        # Prefer 768-dim vectors for offline use
        if vectors_768_path.exists():
            with open(vectors_768_path, 'r', encoding='utf-8') as f:
                vectors_data = json.load(f)

            # Load content from _index.json for display text
            index_data = {}
            if index_path.exists():
                with open(index_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
            index_docs = index_data.get("documents", {})

            vectors_dict = vectors_data.get("vectors", {})
            if not vectors_dict:
                raise HTTPException(400, "Vectors file contains no vectors")

            # Use 768-dim ChromaDB
            store = get_vector_store(dimension=768)

            ids = []
            embeddings = []
            contents = []
            metadatas = []

            for doc_id, embedding in vectors_dict.items():
                ids.append(doc_id)
                embeddings.append(embedding)

                # Get content from index file if available
                doc_info = index_docs.get(doc_id, {})
                contents.append(doc_info.get("content", ""))

                metadata = {
                    "source": source_id,
                    "title": doc_info.get("title", ""),
                    "url": doc_info.get("url", ""),
                }
                metadatas.append(metadata)

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

            dimension_used = "768-dim"

        # Fallback: Use legacy _index.json embeddings
        else:
            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            documents = index_data.get("documents", [])
            if not documents:
                raise HTTPException(400, "Index file contains no documents")

            store = get_vector_store()

            ids = []
            embeddings = []
            contents = []
            metadatas = []

            for doc in documents:
                ids.append(doc["id"])
                embeddings.append(doc["embedding"])
                contents.append(doc.get("content", ""))
                metadatas.append(doc.get("metadata", {}))

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

            dimension_used = "1536-dim (legacy)"

        installed = config.get("installed_packs", [])
        if source_id not in installed:
            installed.append(source_id)
            config.set("installed_packs", installed)
            config.save()

        return {
            "status": "success",
            "source_id": source_id,
            "indexed_documents": len(ids),
            "dimension": dimension_used,
            "message": f"Re-indexed {len(ids)} documents for {source_id} ({dimension_used})"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Re-index failed: {str(e)}")
