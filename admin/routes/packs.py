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
    get_manifest_file, get_metadata_file, get_index_file, get_vectors_file,
    get_vectors_768_file
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


# NOTE: Download pack functionality has been consolidated into source_tools.py
# Use /useradmin/api/install-source or /useradmin/api/download-pack instead


@router.post("/reindex-pack")
async def reindex_pack(request: ReindexPackRequest):
    """
    Re-index an already downloaded pack into ChromaDB.
    Prefers 768-dim vectors for offline use.
    """
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise HTTPException(400, "No backup folder configured. Set it in Settings first.")

    source_id = request.source_id
    source_folder = Path(backup_folder) / source_id

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
