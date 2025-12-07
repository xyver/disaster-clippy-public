"""
Cloud Backup Upload Routes
API endpoints for uploading verified source backups to R2 cloud storage

SECURITY: Upload endpoints require VECTOR_DB_MODE=global to prevent
unauthorized writes to the shared cloud storage.
"""

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import os
import json

from offline_tools.schemas import (
    get_manifest_file, get_metadata_file, get_index_file,
    get_vectors_file, get_backup_manifest_file
)

router = APIRouter()


# Import admin mode check (will be available after app.py imports this)
def _require_global_admin():
    """Local import to avoid circular dependency"""
    from .app import require_global_admin
    return require_global_admin()

# Templates
templates_path = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_path))


@router.get("/cloud-upload", response_class=HTMLResponse)
async def cloud_upload_page(request: Request):
    """Cloud backup upload page"""
    return templates.TemplateResponse("cloud_upload.html", {
        "request": request
    })


def get_backup_paths():
    """Get backup paths from local config"""
    from .local_config import get_local_config
    config = get_local_config()
    return config.get_backup_paths()


@router.get("/api/cloud-storage-status")
async def get_cloud_storage_status():
    """Check if cloud storage (R2) is configured and working"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from offline_tools.cloud.r2 import get_r2_storage
        storage = get_r2_storage()
        return storage.test_connection()
    except ImportError:
        return {
            "configured": False,
            "connected": False,
            "error": "Storage module not available. Install boto3."
        }
    except Exception as e:
        return {
            "configured": False,
            "connected": False,
            "error": str(e)
        }


@router.get("/api/pinecone-status")
async def get_pinecone_status():
    """Check if Pinecone is configured and working"""
    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            return {
                "configured": False,
                "connected": False,
                "error": "PINECONE_API_KEY not set"
            }

        from offline_tools.vectordb import PineconeStore
        store = PineconeStore()
        stats = store.get_stats()

        return {
            "configured": True,
            "connected": True,
            "index_name": stats.get("index_name", "unknown"),
            "total_vectors": stats.get("total_documents", 0),
            "namespace": stats.get("namespace", "default"),
            "sources": stats.get("sources", {}),
            "last_updated": stats.get("last_updated")
        }
    except ImportError as e:
        return {
            "configured": False,
            "connected": False,
            "error": f"Pinecone module not available: {str(e)}"
        }
    except Exception as e:
        # Check if it's just not configured vs actual error
        error_str = str(e)
        if "PINECONE_API_KEY" in error_str or "not set" in error_str.lower():
            return {
                "configured": False,
                "connected": False,
                "error": "Pinecone API key not configured"
            }
        return {
            "configured": True,
            "connected": False,
            "error": str(e)
        }


@router.get("/api/sources-for-upload")
async def get_sources_for_upload():
    """
    Get list of local sources that could be uploaded to cloud.
    Discovers sources from backup folder (looks for _manifest.json files).
    Checks completeness: needs source config, metadata, and backup file.
    """
    from .local_config import get_local_config
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        return {"sources": [], "total": 0, "complete_count": 0}

    backup_path = Path(backup_folder)
    if not backup_path.exists():
        return {"sources": [], "total": 0, "complete_count": 0}

    # Discover sources from backup folder - look for _manifest.json files
    sources_config = {}
    for source_folder in backup_path.iterdir():
        if source_folder.is_dir():
            source_id = source_folder.name
            manifest_file = source_folder / get_manifest_file()
            if manifest_file.exists():
                try:
                    with open(manifest_file, 'r', encoding='utf-8') as f:
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
            # Also include folders with backup content
            elif (source_folder / "pages").exists() or list(source_folder.glob("*.html")):
                sources_config[source_id] = {"name": source_id}

    # Check each source for completeness
    uploadable_sources = []

    for source_id, config_data in sources_config.items():
        source_status = {
            "source_id": source_id,
            "name": config_data.get("name", source_id),
            "license": config_data.get("license", "Unknown"),
            "license_verified": config_data.get("license_verified", False),
            "has_config": True,
            "has_metadata": False,
            "has_backup": False,
            "backup_type": None,
            "backup_path": None,
            "backup_size_mb": 0,
            "is_complete": False,
            "missing": []
        }

        # Check for metadata file
        if backup_folder:
            metadata_path = Path(backup_folder) / source_id / get_metadata_file()
            if metadata_path.exists():
                source_status["has_metadata"] = True
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        source_status["document_count"] = meta.get("document_count", 0)
                except:
                    pass
            else:
                source_status["missing"].append("metadata")

        # Check for backup files using unified detection
        from offline_tools.packager import detect_backup_status
        backup_status = detect_backup_status(source_id, Path(backup_folder) if backup_folder else None)
        source_status["has_backup"] = backup_status["has_backup"]
        source_status["backup_type"] = backup_status["backup_type"]
        source_status["backup_path"] = backup_status["backup_path"]
        source_status["backup_size_mb"] = backup_status["backup_size_mb"]

        if not backup_status["has_backup"]:
            source_status["missing"].append("backup file")
        elif backup_status["backup_size_mb"] < 0.1:
            # Backup exists but is essentially empty (less than 100KB)
            source_status["has_backup"] = False
            source_status["missing"].append("backup file (empty or incomplete)")

        # Check license
        if not source_status["license_verified"]:
            source_status["missing"].append("verified license")

        # Determine if complete (ready for upload)
        source_status["is_complete"] = (
            source_status["has_config"] and
            source_status["has_metadata"] and
            source_status["has_backup"] and
            source_status["backup_size_mb"] >= 0.1 and  # Must have actual content
            source_status["license_verified"]
        )

        uploadable_sources.append(source_status)

    # Sort: complete first, then by name
    uploadable_sources.sort(key=lambda x: (not x["is_complete"], x["name"]))

    return {
        "sources": uploadable_sources,
        "total": len(uploadable_sources),
        "complete_count": sum(1 for s in uploadable_sources if s["is_complete"])
    }


class UploadBackupRequest(BaseModel):
    source_id: str


def _run_upload_backup(source_id: str, source_info: dict, progress_callback=None):
    """Background worker function for uploading backup to cloud."""
    from dotenv import load_dotenv
    load_dotenv()

    if progress_callback:
        progress_callback(5, "Connecting to cloud storage...")

    # Get R2 storage
    from offline_tools.cloud.r2 import get_r2_storage
    storage = get_r2_storage()

    if not storage.is_configured():
        raise Exception("R2 cloud storage not configured")

    # Test connection
    conn_status = storage.test_connection()
    if not conn_status["connected"]:
        raise Exception(f"R2 connection failed: {conn_status.get('error', 'Unknown error')}")

    if progress_callback:
        progress_callback(10, "Preparing upload...")

    # Upload based on backup type
    backup_path = source_info["backup_path"]
    backup_type = source_info["backup_type"]

    # Generate submission folder with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_folder = f"submissions/{timestamp}_{source_id}"

    if backup_type == "zim":
        if progress_callback:
            progress_callback(15, f"Uploading ZIM file ({source_info.get('backup_size_mb', 0)} MB)...")

        # Upload single ZIM file
        remote_key = f"{submission_folder}/{source_id}.zim"
        success = storage.upload_file(backup_path, remote_key)

        if not success:
            raise Exception(f"Upload failed: {storage.get_last_error()}")

        if progress_callback:
            progress_callback(80, "Uploading metadata files...")

        # Also upload all metadata and v2 schema files
        uploaded_files = _upload_submission_metadata_sync(storage, submission_folder, source_id, source_info)
        uploaded_files.insert(0, f"{source_id}.zim")

        # Calculate total size
        total_size = source_info["backup_size_mb"]
        from .local_config import get_local_config
        config = get_local_config()
        backup_folder = config.get_backup_folder()
        if backup_folder:
            source_folder = Path(backup_folder) / source_id
            for f in uploaded_files[1:]:  # Skip ZIM file already counted
                fpath = source_folder / f
                if fpath.exists():
                    total_size += fpath.stat().st_size / (1024*1024)

        if progress_callback:
            progress_callback(100, "Upload complete")

        return {
            "status": "success",
            "source_id": source_id,
            "remote_key": remote_key,
            "size_mb": round(total_size, 2),
            "file_count": len(uploaded_files),
            "files": uploaded_files,
            "message": f"Submitted {len(uploaded_files)} files for review"
        }

    elif backup_type == "html":
        # For HTML folders, create a zip and upload
        import tempfile
        import zipfile

        html_path = Path(backup_path)

        if progress_callback:
            progress_callback(15, "Creating zip archive...")

        # Create temporary zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create zip of HTML folder
            with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                all_files = list(html_path.rglob('*'))
                file_count = len([f for f in all_files if f.is_file()])
                for i, file_path in enumerate(all_files):
                    if file_path.is_file():
                        arcname = file_path.relative_to(html_path)
                        zf.write(file_path, arcname)
                        if progress_callback and file_count > 0:
                            # Scale from 20-70%
                            percent = 20 + int((i / file_count) * 50)
                            progress_callback(percent, f"Zipping files... ({i+1}/{file_count})")

            if progress_callback:
                progress_callback(70, "Uploading zip file...")

            # Upload zip
            remote_key = f"{submission_folder}/{source_id}-html.zip"
            success = storage.upload_file(tmp_path, remote_key)

            if not success:
                raise Exception(f"Upload failed: {storage.get_last_error()}")

            # Get actual upload size
            zip_size = os.path.getsize(tmp_path) / (1024*1024)

            if progress_callback:
                progress_callback(85, "Uploading metadata files...")

            # Also upload all metadata and v2 schema files
            uploaded_files = _upload_submission_metadata_sync(storage, submission_folder, source_id, source_info)
            uploaded_files.insert(0, f"{source_id}-html.zip")

            # Calculate total size including metadata files
            total_size = zip_size
            from .local_config import get_local_config
            config = get_local_config()
            backup_folder = config.get_backup_folder()
            if backup_folder:
                source_folder = Path(backup_folder) / source_id
                for f in uploaded_files[1:]:  # Skip zip already counted
                    fpath = source_folder / f
                    if fpath.exists():
                        total_size += fpath.stat().st_size / (1024*1024)

            if progress_callback:
                progress_callback(100, "Upload complete")

            return {
                "status": "success",
                "source_id": source_id,
                "remote_key": remote_key,
                "size_mb": round(total_size, 2),
                "file_count": len(uploaded_files),
                "files": uploaded_files,
                "message": f"Submitted {len(uploaded_files)} files for review"
            }

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    elif backup_type == "pdf":
        # For PDF collections, create a zip and upload (similar to HTML)
        import tempfile
        import zipfile

        pdf_path = Path(backup_path)

        if progress_callback:
            progress_callback(15, "Creating zip archive...")

        # Create temporary zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create zip of PDF folder
            with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                all_files = list(pdf_path.rglob('*'))
                file_count = len([f for f in all_files if f.is_file()])
                for i, file_path in enumerate(all_files):
                    if file_path.is_file():
                        arcname = file_path.relative_to(pdf_path)
                        zf.write(file_path, arcname)
                        if progress_callback and file_count > 0:
                            # Scale from 20-70%
                            percent = 20 + int((i / file_count) * 50)
                            progress_callback(percent, f"Zipping files... ({i+1}/{file_count})")

            if progress_callback:
                progress_callback(70, "Uploading zip file...")

            # Upload zip
            remote_key = f"{submission_folder}/{source_id}-pdf.zip"
            success = storage.upload_file(tmp_path, remote_key)

            if not success:
                raise Exception(f"Upload failed: {storage.get_last_error()}")

            # Get actual upload size
            zip_size = os.path.getsize(tmp_path) / (1024*1024)

            if progress_callback:
                progress_callback(85, "Uploading metadata files...")

            # Also upload all metadata and v2 schema files
            uploaded_files = _upload_submission_metadata_sync(storage, submission_folder, source_id, source_info)
            uploaded_files.insert(0, f"{source_id}-pdf.zip")

            # Calculate total size including metadata files
            total_size = zip_size
            from .local_config import get_local_config
            config = get_local_config()
            backup_folder = config.get_backup_folder()
            if backup_folder:
                source_folder = Path(backup_folder) / source_id
                for f in uploaded_files[1:]:  # Skip zip already counted
                    fpath = source_folder / f
                    if fpath.exists():
                        total_size += fpath.stat().st_size / (1024*1024)

            if progress_callback:
                progress_callback(100, "Upload complete")

            return {
                "status": "success",
                "source_id": source_id,
                "remote_key": remote_key,
                "size_mb": round(total_size, 2),
                "file_count": len(uploaded_files),
                "files": uploaded_files,
                "message": f"Submitted {len(uploaded_files)} files for review"
            }

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    else:
        raise Exception(f"Unknown backup type: {backup_type}")


def _upload_submission_metadata_sync(storage, submission_folder: str, source_id: str, source_info: dict):
    """Upload all source schema files alongside the backup submission - synchronous version"""
    import tempfile

    # Get backup folder
    from .local_config import get_local_config
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    # Load source config from _manifest.json
    source_config = {}
    if backup_folder:
        manifest_file = Path(backup_folder) / source_id / get_manifest_file()
        if manifest_file.exists():
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    source_config = json.load(f)
            except Exception:
                pass

    uploaded_files = []
    metadata = {}

    if backup_folder:
        source_folder = Path(backup_folder) / source_id

        # Upload schema files
        schema_files = [
            get_manifest_file(),
            get_metadata_file(),
            get_index_file(),
            get_vectors_file(),
            get_backup_manifest_file(),
        ]

        for filename in schema_files:
            file_path = source_folder / filename
            if file_path.exists():
                remote_key = f"{submission_folder}/{filename}"
                if storage.upload_file(str(file_path), remote_key):
                    uploaded_files.append(filename)

                # Load metadata for summary
                if filename == get_metadata_file():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

    # Create submission manifest (for tracking status, not the pack manifest)
    from datetime import datetime
    submission_manifest = {
        "source_id": source_id,
        "submitted_at": datetime.now().isoformat(),
        "status": "pending_review",
        "source_config": source_config,
        "metadata_summary": {
            "document_count": metadata.get("document_count", metadata.get("total_documents", 0)),
            "source_type": metadata.get("source_type", "unknown")
        },
        "backup_info": {
            "type": source_info.get("backup_type"),
            "size_mb": source_info.get("backup_size_mb", 0)
        },
        "uploaded_files": uploaded_files
    }

    # Write to temp file and upload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
        json.dump(submission_manifest, tmp, indent=2)
        tmp_path = tmp.name

    try:
        storage.upload_file(tmp_path, f"{submission_folder}/submission_manifest.json")
        uploaded_files.append("submission_manifest.json")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return uploaded_files


@router.post("/api/upload-backup")
async def upload_backup_to_cloud(
    request: UploadBackupRequest,
    _: bool = Depends(_require_global_admin)
):
    """
    Upload a source's backup file to R2 cloud storage.
    Only allows upload if source is complete (config + metadata + backup + verified license).
    Runs as a background job with progress tracking.

    REQUIRES: VECTOR_DB_MODE=global
    """
    # First verify the source is complete
    sources_response = await get_sources_for_upload()
    sources = sources_response.get("sources", [])

    source_info = None
    for s in sources:
        if s["source_id"] == request.source_id:
            source_info = s
            break

    if not source_info:
        raise HTTPException(404, f"Source not found: {request.source_id}")

    if not source_info["is_complete"]:
        missing = ", ".join(source_info["missing"])
        raise HTTPException(400, f"Source is incomplete. Missing: {missing}")

    # Verify R2 is configured before starting job
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from offline_tools.cloud.r2 import get_r2_storage
        storage = get_r2_storage()

        if not storage.is_configured():
            raise HTTPException(500, "R2 cloud storage not configured")

    except ImportError:
        raise HTTPException(500, "Storage module not available. Install boto3.")

    # Submit as background job
    from .job_manager import get_job_manager
    manager = get_job_manager()

    try:
        job_id = manager.submit(
            "upload",
            request.source_id,
            _run_upload_backup,
            request.source_id,
            source_info
        )
    except ValueError as e:
        # Job already running for this source
        raise HTTPException(409, str(e))

    return {
        "status": "submitted",
        "job_id": job_id,
        "source_id": request.source_id,
        "message": f"Upload job started for {request.source_id}"
    }


async def _upload_submission_metadata(storage, submission_folder: str, source_id: str, source_info: dict):
    """Upload all source schema files alongside the backup submission"""
    import tempfile

    # Get backup folder
    from .local_config import get_local_config
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    # Load source config from _manifest.json
    source_config = {}
    if backup_folder:
        manifest_file = Path(backup_folder) / source_id / get_manifest_file()
        if manifest_file.exists():
            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    source_config = json.load(f)
            except Exception:
                pass

    uploaded_files = []
    metadata = {}

    if backup_folder:
        source_folder = Path(backup_folder) / source_id

        # Upload schema files
        schema_files = [
            get_manifest_file(),
            get_metadata_file(),
            get_index_file(),
            get_vectors_file(),
            get_backup_manifest_file(),
        ]

        for filename in schema_files:
            file_path = source_folder / filename
            if file_path.exists():
                remote_key = f"{submission_folder}/{filename}"
                if storage.upload_file(str(file_path), remote_key):
                    uploaded_files.append(filename)

                # Load metadata for summary
                if filename == get_metadata_file():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

    # Create submission manifest (for tracking status, not the pack manifest)
    from datetime import datetime
    submission_manifest = {
        "source_id": source_id,
        "submitted_at": datetime.now().isoformat(),
        "status": "pending_review",
        "source_config": source_config,
        "metadata_summary": {
            "document_count": metadata.get("document_count", metadata.get("total_documents", 0)),
            "source_type": metadata.get("source_type", "unknown")
        },
        "backup_info": {
            "type": source_info.get("backup_type"),
            "size_mb": source_info.get("backup_size_mb", 0)
        },
        "uploaded_files": uploaded_files
    }

    # Write to temp file and upload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
        json.dump(submission_manifest, tmp, indent=2)
        tmp_path = tmp.name

    try:
        storage.upload_file(tmp_path, f"{submission_folder}/submission_manifest.json")
        uploaded_files.append("submission_manifest.json")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return uploaded_files


def _load_manifest_from_r2(storage, manifest_key: str) -> dict:
    """Download and parse a manifest.json from R2"""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        if storage.download_file(manifest_key, tmp_path):
            with open(tmp_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return {}


@router.get("/api/cloud-backups")
async def list_cloud_backups():
    """List user's submissions in cloud storage with status from manifest"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from offline_tools.cloud.r2 import get_r2_storage
        storage = get_r2_storage()

        if not storage.is_configured():
            return {"submissions": [], "error": "R2 not configured"}

        # List submissions folder (user uploads awaiting review)
        files = storage.list_files("submissions/")

        # Group by submission folder
        submissions = {}
        for f in files:
            key = f["key"]
            parts = key.split("/")
            if len(parts) >= 2:
                submission_id = parts[1]  # e.g., "20241201_123456_builditsolar"
                if submission_id not in submissions:
                    submissions[submission_id] = {
                        "files": [],
                        "total_size_mb": 0,
                        "manifest_key": None,
                        "status": "pending_review",
                        "admin_message": None
                    }
                submissions[submission_id]["files"].append(f)
                submissions[submission_id]["total_size_mb"] += f.get("size_mb", 0)

                # Track manifest file
                if key.endswith("manifest.json"):
                    submissions[submission_id]["manifest_key"] = key

        # Load manifest for each submission to get status
        for sub_id, sub_info in submissions.items():
            if sub_info.get("manifest_key"):
                manifest = _load_manifest_from_r2(storage, sub_info["manifest_key"])
                if manifest:
                    sub_info["status"] = manifest.get("status", "pending_review")
                    sub_info["admin_message"] = manifest.get("admin_message")
                    sub_info["submitted_at"] = manifest.get("submitted_at")
                    sub_info["denied_at"] = manifest.get("denied_at")

        return {
            "submissions": submissions,
            "total_submissions": len(submissions),
            "total_files": len(files),
            "total_size_mb": round(sum(f.get("size_mb", 0) for f in files), 2)
        }

    except ImportError:
        return {"submissions": [], "error": "Storage module not available"}
    except Exception as e:
        return {"submissions": [], "error": str(e)}


# ============================================================================
# Pinecone Sync Operations
# ============================================================================

class PineconeSyncRequest(BaseModel):
    """Request model for Pinecone sync operations"""
    source_id: Optional[str] = None  # Filter to specific source
    dry_run: bool = True  # Default to dry run for safety


@router.post("/api/pinecone-compare")
async def pinecone_compare(request: PineconeSyncRequest = None):
    """
    Compare local ChromaDB with Pinecone to see what needs syncing.
    Returns counts of documents to push, pull, update, and conflicts.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()

        # Check Pinecone is configured
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=400,
                content={"error": "PINECONE_API_KEY not configured"}
            )

        from offline_tools.vectordb import VectorStore, PineconeStore
        from offline_tools.vectordb.sync import SyncManager

        # Get local ChromaDB
        local = VectorStore()
        remote = PineconeStore()

        sync = SyncManager(local, remote)
        source_filter = request.source_id if request else None
        diff = sync.compare(source=source_filter)

        return {
            "status": "success",
            "source_filter": source_filter,
            "local_total": len(sync.local_metadata.get_ids()),
            "remote_total": len(sync.remote_metadata.get_ids()) if sync.remote_metadata else 0,
            "to_push": len(diff.to_push),
            "to_pull": len(diff.to_pull),
            "to_update": len(diff.to_update),
            "in_sync": len(diff.in_sync),
            "conflicts": len(diff.conflicts),
            "conflict_details": diff.conflicts[:5] if diff.conflicts else []  # First 5 conflicts
        }

    except ImportError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Missing dependency: {str(e)}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.post("/api/pinecone-push")
async def pinecone_push(
    request: PineconeSyncRequest = None,
    _: bool = Depends(_require_global_admin)
):
    """
    Push local ChromaDB changes to Pinecone.
    Uploads new and updated documents from local to remote.

    REQUIRES: VECTOR_DB_MODE=global
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=400,
                content={"error": "PINECONE_API_KEY not configured"}
            )

        dry_run = request.dry_run if request else True
        source_filter = request.source_id if request else None

        from offline_tools.vectordb import VectorStore, PineconeStore
        from offline_tools.vectordb.sync import SyncManager

        local = VectorStore()
        remote = PineconeStore()

        sync = SyncManager(local, remote)
        diff = sync.compare(source=source_filter)

        # Check for conflicts
        if diff.conflicts:
            return JSONResponse(
                status_code=409,
                content={
                    "error": f"{len(diff.conflicts)} conflicts detected. Remote has newer versions.",
                    "conflicts": diff.conflicts[:5]
                }
            )

        # Nothing to push?
        if not diff.to_push and not diff.to_update:
            return {
                "status": "success",
                "message": "Already in sync - nothing to push",
                "dry_run": dry_run,
                "pushed": 0,
                "updated": 0
            }

        # Do the push
        stats = sync.push(diff, dry_run=dry_run, update=True, force=False)

        return {
            "status": "success",
            "dry_run": dry_run,
            "pushed": stats.get("pushed", 0),
            "updated": stats.get("updated", 0),
            "skipped": stats.get("skipped", 0),
            "errors": stats.get("errors", 0),
            "message": "Dry run complete - no changes made" if dry_run else "Push complete"
        }

    except ImportError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Missing dependency: {str(e)}"}
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@router.delete("/api/pinecone-namespace")
async def pinecone_delete_namespace(_: bool = Depends(_require_global_admin)):
    """
    Delete all vectors in the current Pinecone namespace.
    DANGER: This cannot be undone!

    REQUIRES: VECTOR_DB_MODE=global
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            return JSONResponse(
                status_code=400,
                content={"error": "PINECONE_API_KEY not configured"}
            )

        from offline_tools.vectordb import PineconeStore

        store = PineconeStore()
        stats_before = store.get_stats()
        vector_count = stats_before.get("total_documents", 0)

        # Delete all vectors in namespace
        store.delete_all()

        return {
            "status": "success",
            "message": f"Deleted {vector_count} vectors from namespace '{store.namespace}'",
            "deleted_count": vector_count,
            "namespace": store.namespace
        }

    except ImportError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Missing dependency: {str(e)}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
