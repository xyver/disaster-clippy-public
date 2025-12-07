"""
Pack Management API

Endpoints for browsing, installing, and downloading source packs.
Handles both local pack management and cloud (R2) downloads.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json
import os

from offline_tools.schemas import (
    get_manifest_file, get_metadata_file, get_index_file, get_vectors_file
)

router = APIRouter(prefix="/api", tags=["Pack Management"])


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_local_config():
    """Get local config - imported here to avoid circular imports"""
    from admin.local_config import get_local_config as _get_config
    return _get_config()


# =============================================================================
# REQUEST MODELS
# =============================================================================

class InstallPackRequest(BaseModel):
    source_id: str
    parent_url: str = ""


class DownloadPackRequest(BaseModel):
    source_id: str


class ReindexPackRequest(BaseModel):
    source_id: str


# =============================================================================
# INSTALLED PACKS
# =============================================================================

@router.get("/installed-packs")
async def get_installed_packs():
    """Get list of locally installed packs"""
    config = get_local_config()
    installed = config.get("installed_packs", [])

    backup_folder = config.get_backup_folder()
    packs = []
    for source_id in installed:
        source_config = {}
        if backup_folder:
            manifest_file = Path(backup_folder) / source_id / get_manifest_file()
            if manifest_file.exists():
                try:
                    with open(manifest_file, 'r', encoding='utf-8') as f:
                        source_config = json.load(f)
                except Exception:
                    pass

        doc_count = 0
        metadata_path = Path(backup_folder) / source_id / get_metadata_file() if backup_folder else None

        if metadata_path and metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    doc_count = meta.get("document_count", 0)
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

        backup_folder = config.get_backup_folder()
        if not backup_folder:
            raise HTTPException(400, "No backup folder configured. Set it in Settings first.")

        metadata_dir = Path(backup_folder) / request.source_id
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
            "message": f"Installed {request.source_id} pack successfully"
        }

    except httpx.RequestError as e:
        raise HTTPException(500, f"Network error: {str(e)}")


# =============================================================================
# CLOUD SOURCES (R2)
# =============================================================================

@router.get("/cloud-sources")
async def get_cloud_sources():
    """
    Get sources available from the Global Cloud (R2 backups/ folder).
    Returns sources in a format compatible with the sources page filtering.

    Uses Railway proxy if R2 keys aren't configured locally.
    """
    try:
        # Check if we should use proxy (no R2 keys, but proxy URL configured)
        config = get_local_config()
        if config.should_use_proxy():
            from admin.cloud_proxy import get_proxy_client
            proxy = get_proxy_client()
            result = proxy.get_sources()
            if "error" in result and not result.get("connected", False):
                return {"sources": [], "connected": False, "error": result["error"], "via": "proxy"}
            result["via"] = "proxy"
            return result

        # Direct R2 access
        from dotenv import load_dotenv
        load_dotenv()
        from offline_tools.cloud.r2 import get_r2_storage
        storage = get_r2_storage()

        if not storage.is_configured():
            # No R2 keys and no proxy - show helpful message
            proxy_url = config.get_railway_proxy_url()
            if not proxy_url:
                return {
                    "sources": [],
                    "connected": False,
                    "error": "R2 not configured. Set RAILWAY_PROXY_URL to use cloud through Railway proxy."
                }
            return {"sources": [], "connected": False, "error": "R2 not configured"}

        conn_status = storage.test_connection()
        if not conn_status["connected"]:
            return {"sources": [], "connected": False, "error": conn_status.get("error", "Connection failed")}

        # Try to load _master.json for source metadata
        master_data = {}
        import tempfile
        tmp_path = Path(tempfile.gettempdir()) / "cloud_master.json"
        if storage.download_file("backups/_master.json", str(tmp_path)):
            try:
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    master_data = json.load(f)
            except Exception as e:
                print(f"Failed to parse _master.json: {e}")

        master_sources = master_data.get("sources", {})
        if master_sources:
            sources = []
            for source_id, source_info in master_sources.items():
                sources.append({
                    "source_id": source_id,
                    "name": source_info.get("name", source_id.replace("-", " ").replace("_", " ").title()),
                    "description": source_info.get("description", ""),
                    "license": source_info.get("license", "Unknown"),
                    "document_count": source_info.get("count", source_info.get("document_count", 0)),
                    "backup_type": source_info.get("backup_type", "cloud"),
                    "base_url": source_info.get("base_url", "")
                })
            sources.sort(key=lambda x: x["name"].lower())
            return {
                "sources": sources,
                "connected": True,
                "total": len(sources),
                "source": "master_json"
            }

        # Fallback: List files in backups/ folder
        files = storage.list_files("backups/")
        root_files = storage.list_files("")
        root_prefixes = set()
        for f in root_files[:100]:
            key = f["key"]
            if "/" in key:
                root_prefixes.add(key.split("/")[0])

        skip_files = {"_master.json", "sources.json", "backups.json"}
        source_ids = set()
        sample_keys = []
        for f in files:
            key = f["key"]
            if len(sample_keys) < 10:
                sample_keys.append(key)
            parts = key.split("/")
            if len(parts) >= 2:
                source_id = parts[1]
                if source_id and source_id not in skip_files and not source_id.startswith("_"):
                    source_ids.add(source_id)

        sources = []
        for source_id in source_ids:
            sources.append({
                "source_id": source_id,
                "name": source_id.replace("-", " ").replace("_", " ").title(),
                "description": "",
                "license": "Unknown",
                "document_count": 0,
                "backup_type": "cloud",
                "base_url": ""
            })

        sources.sort(key=lambda x: x["name"].lower())

        return {
            "sources": sources,
            "connected": True,
            "total": len(sources),
            "source": "file_listing",
            "files_found": len(files),
            "root_prefixes": list(root_prefixes),
            "sample_keys": sample_keys,
            "debug": {
                "master_found": False,
                "backups_files_count": len(files),
                "root_files_count": len(root_files)
            }
        }

    except ImportError:
        return {"sources": [], "connected": False, "error": "Storage module not available"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"sources": [], "connected": False, "error": str(e)}


@router.get("/available-packs")
async def get_available_packs():
    """
    Get packs available from the Global Cloud Backup (R2 backups/ folder).
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from offline_tools.cloud.r2 import get_r2_storage
        storage = get_r2_storage()

        if not storage.is_configured():
            return {"packs": [], "connected": False, "error": "R2 not configured"}

        conn_status = storage.test_connection()
        if not conn_status["connected"]:
            return {"packs": [], "connected": False, "error": conn_status.get("error", "Connection failed")}

        files = storage.list_files("backups/")

        skip_files = {"_master.json", "sources.json", "backups.json"}
        packs = {}
        for f in files:
            key = f["key"]
            parts = key.split("/")
            if len(parts) >= 2:
                source_id = parts[1]
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

        pack_list = list(packs.values())

        return {
            "packs": pack_list,
            "connected": True,
            "total": len(pack_list)
        }

    except ImportError:
        return {"packs": [], "connected": False, "error": "Storage module not available"}
    except Exception as e:
        return {"packs": [], "connected": False, "error": str(e)}


# =============================================================================
# DOWNLOAD PACK (Background Job)
# =============================================================================

def _run_download_pack(source_id: str, progress_callback=None, cancel_checker=None):
    """
    Background job function to download a source pack from R2.
    Automatically uses Railway proxy if R2 keys aren't configured.
    """
    import zipfile
    from datetime import datetime
    from dotenv import load_dotenv
    load_dotenv()

    def update_progress(pct, msg):
        if progress_callback:
            progress_callback(pct, msg)

    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise Exception("No backup folder configured. Set it in Settings first.")

    source_folder = Path(backup_folder) / source_id
    source_folder.mkdir(parents=True, exist_ok=True)

    # Check if we should use proxy
    use_proxy = config.should_use_proxy()

    if use_proxy:
        # Use Railway proxy
        from admin.cloud_proxy import get_proxy_client
        update_progress(0, "Connecting via Railway proxy...")
        proxy = get_proxy_client()

        if not proxy.is_configured():
            raise Exception("Railway proxy URL not configured")

        update_progress(5, "Listing files via proxy...")
        file_info = proxy.get_source_files(source_id)

        if "error" in file_info:
            raise Exception(f"Failed to get file list: {file_info['error']}")

        files = file_info.get("files", [])
        if not files:
            raise Exception(f"No files found for source: {source_id}")

        downloaded_files = []
        total_size = 0
        total_files = len(files)

        for idx, f in enumerate(files):
            filename = f["filename"]
            if not filename:
                continue

            local_path = source_folder / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)

            pct = 10 + int((idx / total_files) * 60)
            update_progress(pct, f"Downloading {filename}...")

            success = proxy.download_file(source_id, filename, str(local_path))
            if success:
                downloaded_files.append(filename)
                total_size += f.get("size_mb", 0)

    else:
        # Direct R2 access
        from offline_tools.cloud.r2 import get_r2_storage
        update_progress(0, "Connecting to R2 storage...")
        storage = get_r2_storage()

        if not storage.is_configured():
            raise Exception("R2 storage not configured. Set RAILWAY_PROXY_URL to use proxy instead.")

        update_progress(5, "Listing files in R2...")
        prefix = f"backups/{source_id}/"
        files = storage.list_files(prefix)

        if not files:
            raise Exception(f"No files found for source: {source_id}")

        downloaded_files = []
        total_size = 0
        total_files = len(files)

        for idx, f in enumerate(files):
            key = f["key"]
            relative_path = key.replace(prefix, "")
            if not relative_path:
                continue

            local_path = source_folder / relative_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            pct = 10 + int((idx / total_files) * 60)
            update_progress(pct, f"Downloading {relative_path}...")

            success = storage.download_file(key, str(local_path))
            if success:
                downloaded_files.append(relative_path)
                total_size += f.get("size_mb", 0)

    # Extract HTML zip files if present
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

    # Import index data into ChromaDB
    update_progress(80, "Importing to ChromaDB...")
    index_path = source_folder / get_index_file()
    indexed_count = 0
    if index_path.exists():
        try:
            from offline_tools.vectordb import get_vector_store

            with open(index_path, 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            documents = index_data.get("documents", [])
            if documents:
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
                total_batches = (len(ids) + batch_size - 1) // batch_size
                for i, start in enumerate(range(0, len(ids), batch_size)):
                    batch_ids = ids[start:start+batch_size]
                    batch_embeddings = embeddings[start:start+batch_size]
                    batch_contents = contents[start:start+batch_size]
                    batch_metadatas = metadatas[start:start+batch_size]

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

    # Update installed packs
    update_progress(95, "Updating local configuration...")
    installed = config.get("installed_packs", [])
    if source_id not in installed:
        installed.append(source_id)
        config.set("installed_packs", installed)
        config.save()

    # Create _manifest.json if it doesn't exist
    manifest_file = source_folder / get_manifest_file()
    if not manifest_file.exists():
        source_data = {
            "schema_version": 3,
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
        with open(manifest_file, 'w', encoding='utf-8') as f:
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


@router.post("/download-pack")
async def download_pack_from_r2(request: DownloadPackRequest):
    """
    Download a source pack from R2 cloud storage.
    Submits as background job - returns job_id for tracking.

    Automatically uses Railway proxy if R2 keys aren't configured.
    """
    try:
        config = get_local_config()
        backup_folder = config.get_backup_folder()

        if not backup_folder:
            raise HTTPException(400, "No backup folder configured. Set it in Settings first.")

        # Check if we should use proxy
        use_proxy = config.should_use_proxy()

        if use_proxy:
            # Verify proxy is configured and source exists
            from admin.cloud_proxy import get_proxy_client
            proxy = get_proxy_client()

            if not proxy.is_configured():
                raise HTTPException(400, "Railway proxy URL not configured")

            file_info = proxy.get_source_files(request.source_id)
            if "error" in file_info:
                raise HTTPException(503, f"Proxy error: {file_info['error']}")
            if not file_info.get("files"):
                raise HTTPException(404, f"No files found for source: {request.source_id}")

        else:
            # Direct R2 access
            from dotenv import load_dotenv
            load_dotenv()
            from offline_tools.cloud.r2 import get_r2_storage
            storage = get_r2_storage()

            if not storage.is_configured():
                raise HTTPException(400, "R2 not configured. Set RAILWAY_PROXY_URL to use proxy instead.")

            prefix = f"backups/{request.source_id}/"
            files = storage.list_files(prefix)
            if not files:
                raise HTTPException(404, f"No files found for source: {request.source_id}")

        from admin.job_manager import get_job_manager
        manager = get_job_manager()

        try:
            job_id = manager.submit(
                "download",
                request.source_id,
                _run_download_pack,
                request.source_id
            )
        except ValueError as e:
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


@router.post("/reindex-pack")
async def reindex_pack(request: ReindexPackRequest):
    """
    Re-index an already downloaded pack into ChromaDB.
    """
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise HTTPException(400, "No backup folder configured. Set it in Settings first.")

    source_id = request.source_id
    source_folder = Path(backup_folder) / source_id

    if not source_folder.exists():
        raise HTTPException(404, f"Source folder not found: {source_id}")

    index_path = source_folder / get_index_file()
    if not index_path.exists():
        raise HTTPException(404, f"Index file not found: {get_index_file()}")

    try:
        from offline_tools.vectordb import get_vector_store

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
