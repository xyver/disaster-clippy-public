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
from typing import Optional, Dict, Any
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
                    # Get modification time for sorting (most recent first)
                    mtime = manifest_file.stat().st_mtime
                    sources_config[source_id] = {
                        "name": source_data.get("name", source_id),
                        "description": source_data.get("description", ""),
                        "license": source_data.get("license", "Unknown"),
                        "base_url": source_data.get("base_url", ""),
                        "license_verified": source_data.get("license_verified", False),
                        "last_published": source_data.get("last_published"),
                        "last_modified": mtime,
                    }
                except Exception:
                    sources_config[source_id] = {"name": source_id, "last_modified": 0}
            # Also include folders with backup content
            elif (source_folder / "pages").exists() or list(source_folder.glob("*.html")):
                # Use folder mtime as fallback
                mtime = source_folder.stat().st_mtime
                sources_config[source_id] = {"name": source_id, "last_modified": mtime}

    # Check each source for completeness
    uploadable_sources = []

    for source_id, config_data in sources_config.items():
        source_status = {
            "source_id": source_id,
            "name": config_data.get("name", source_id),
            "license": config_data.get("license", "Unknown"),
            "license_verified": config_data.get("license_verified", False),
            "last_published": config_data.get("last_published"),
            "last_modified": config_data.get("last_modified", 0),
            "has_config": True,
            "has_metadata": False,
            "has_backup": False,
            "backup_type": None,
            "backup_path": None,
            "backup_size_mb": 0,
            "is_complete": False,
            "missing": []
        }

        # Check for metadata file - use fast header reading
        if backup_folder:
            metadata_path = Path(backup_folder) / source_id / get_metadata_file()
            if metadata_path.exists():
                source_status["has_metadata"] = True
                # Fast read - only get document_count from header
                from offline_tools.packager import read_json_header_only
                meta_header = read_json_header_only(metadata_path)
                source_status["document_count"] = meta_header.get("document_count", 0)
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

    # Sort: complete sources first, then by most recently modified within each group
    uploadable_sources.sort(key=lambda x: (not x["is_complete"], -x.get("last_modified", 0)))

    return {
        "sources": uploadable_sources,
        "total": len(uploadable_sources),
        "complete_count": sum(1 for s in uploadable_sources if s["is_complete"])
    }


class UploadBackupRequest(BaseModel):
    source_id: str
    publish_mode: bool = False  # True = global admin publish to production, False = submit for review
    sync_mode: str = "update"  # "update" (add/merge) or "replace" (delete old vectors first)


def _run_upload_backup(source_id: str, source_info: dict, progress_callback=None, cancel_checker=None):
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
        # Uses cached zip if available (for resume after failed upload)
        import zipfile
        from admin.job_manager import get_jobs_folder

        html_path = Path(backup_path)

        # Use cached zip location in _jobs folder (survives failed uploads)
        jobs_folder = get_jobs_folder()
        if jobs_folder:
            cached_zip_path = jobs_folder / f"{source_id}_upload.zip"
        else:
            # Fallback to source folder if no jobs folder
            cached_zip_path = html_path.parent / f"{source_id}_upload.zip"

        zip_path = str(cached_zip_path)
        used_cache = False

        # Check if we have a valid cached zip
        if cached_zip_path.exists():
            # Get newest file modification time in source folder
            source_files = list(html_path.rglob('*'))
            newest_source_mtime = max((f.stat().st_mtime for f in source_files if f.is_file()), default=0)
            cache_mtime = cached_zip_path.stat().st_mtime

            if cache_mtime > newest_source_mtime:
                # Cache is newer than all source files - reuse it
                if progress_callback:
                    cache_size_mb = cached_zip_path.stat().st_size / (1024*1024)
                    progress_callback(15, f"Using cached zip ({cache_size_mb:.1f} MB)...")
                used_cache = True
            else:
                # Source files changed since cache was created - delete old cache
                if progress_callback:
                    progress_callback(15, "Source changed, recreating zip...")
                cached_zip_path.unlink()

        if not used_cache:
            if progress_callback:
                progress_callback(15, "Creating zip archive...")

            # Create zip of HTML folder
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
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
                zip_size_mb = os.path.getsize(zip_path) / (1024*1024)
                progress_callback(70, f"Zip created ({zip_size_mb:.1f} MB), uploading...")

        try:
            if progress_callback:
                progress_callback(70, "Uploading zip file...")

            # Upload zip
            remote_key = f"{submission_folder}/{source_id}-html.zip"
            success = storage.upload_file(zip_path, remote_key)

            if not success:
                raise Exception(f"Upload failed: {storage.get_last_error()}")

            # Get actual upload size
            zip_size = os.path.getsize(zip_path) / (1024*1024)

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

            # Upload succeeded - delete cached zip
            if cached_zip_path.exists():
                cached_zip_path.unlink()

            if progress_callback:
                progress_callback(100, "Upload complete")

            msg = f"Submitted {len(uploaded_files)} files for review"
            if used_cache:
                msg += " (used cached zip)"

            return {
                "status": "success",
                "source_id": source_id,
                "remote_key": remote_key,
                "size_mb": round(total_size, 2),
                "file_count": len(uploaded_files),
                "files": uploaded_files,
                "used_cached_zip": used_cache,
                "message": msg
            }

        except Exception as e:
            # Upload failed - keep the cached zip for retry
            if progress_callback:
                progress_callback(0, f"Upload failed (zip cached for retry): {e}")
            raise

    elif backup_type == "pdf":
        # For PDF collections, create a zip and upload
        # Uses cached zip if available (for resume after failed upload)
        import zipfile
        from admin.job_manager import get_jobs_folder

        pdf_path = Path(backup_path)

        # Use cached zip location in _jobs folder (survives failed uploads)
        jobs_folder = get_jobs_folder()
        if jobs_folder:
            cached_zip_path = jobs_folder / f"{source_id}_upload.zip"
        else:
            cached_zip_path = pdf_path.parent / f"{source_id}_upload.zip"

        zip_path = str(cached_zip_path)
        used_cache = False

        # Check if we have a valid cached zip
        if cached_zip_path.exists():
            source_files = list(pdf_path.rglob('*'))
            newest_source_mtime = max((f.stat().st_mtime for f in source_files if f.is_file()), default=0)
            cache_mtime = cached_zip_path.stat().st_mtime

            if cache_mtime > newest_source_mtime:
                if progress_callback:
                    cache_size_mb = cached_zip_path.stat().st_size / (1024*1024)
                    progress_callback(15, f"Using cached zip ({cache_size_mb:.1f} MB)...")
                used_cache = True
            else:
                if progress_callback:
                    progress_callback(15, "Source changed, recreating zip...")
                cached_zip_path.unlink()

        if not used_cache:
            if progress_callback:
                progress_callback(15, "Creating zip archive...")

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                all_files = list(pdf_path.rglob('*'))
                file_count = len([f for f in all_files if f.is_file()])
                for i, file_path in enumerate(all_files):
                    if file_path.is_file():
                        arcname = file_path.relative_to(pdf_path)
                        zf.write(file_path, arcname)
                        if progress_callback and file_count > 0:
                            percent = 20 + int((i / file_count) * 50)
                            progress_callback(percent, f"Zipping files... ({i+1}/{file_count})")

            if progress_callback:
                zip_size_mb = os.path.getsize(zip_path) / (1024*1024)
                progress_callback(70, f"Zip created ({zip_size_mb:.1f} MB), uploading...")

        try:
            if progress_callback:
                progress_callback(70, "Uploading zip file...")

            remote_key = f"{submission_folder}/{source_id}-pdf.zip"
            success = storage.upload_file(zip_path, remote_key)

            if not success:
                raise Exception(f"Upload failed: {storage.get_last_error()}")

            zip_size = os.path.getsize(zip_path) / (1024*1024)

            if progress_callback:
                progress_callback(85, "Uploading metadata files...")

            uploaded_files = _upload_submission_metadata_sync(storage, submission_folder, source_id, source_info)
            uploaded_files.insert(0, f"{source_id}-pdf.zip")

            total_size = zip_size
            from .local_config import get_local_config
            config = get_local_config()
            backup_folder = config.get_backup_folder()
            if backup_folder:
                source_folder = Path(backup_folder) / source_id
                for f in uploaded_files[1:]:
                    fpath = source_folder / f
                    if fpath.exists():
                        total_size += fpath.stat().st_size / (1024*1024)

            # Upload succeeded - delete cached zip
            if cached_zip_path.exists():
                cached_zip_path.unlink()

            if progress_callback:
                progress_callback(100, "Upload complete")

            msg = f"Submitted {len(uploaded_files)} files for review"
            if used_cache:
                msg += " (used cached zip)"

            return {
                "status": "success",
                "source_id": source_id,
                "remote_key": remote_key,
                "size_mb": round(total_size, 2),
                "file_count": len(uploaded_files),
                "files": uploaded_files,
                "used_cached_zip": used_cache,
                "message": msg
            }

        except Exception as e:
            # Upload failed - keep the cached zip for retry
            if progress_callback:
                progress_callback(0, f"Upload failed (zip cached for retry): {e}")
            raise

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


# =============================================================================
# GLOBAL ADMIN: Publish to Production
# =============================================================================

def _run_publish_to_production(source_id: str, source_info: dict, sync_mode: str = "update", progress_callback=None, cancel_checker=None):
    """
    Background worker for global admin: publish source directly to production.

    This:
    1. Uploads source pack to backups R2 bucket (not submissions)
    2. Updates _master.json in R2 with new source
    3. Syncs vectors to Pinecone for cloud search

    Args:
        sync_mode: "update" (add/merge new vectors) or "replace" (delete old vectors first)
        progress_callback: Callback for progress updates (added by job manager)

    REQUIRES: VECTOR_DB_MODE=global
    """
    from dotenv import load_dotenv
    load_dotenv()

    if progress_callback:
        progress_callback(2, "Connecting to backups storage...")

    # Get backups storage (separate bucket from submissions)
    from offline_tools.cloud.r2 import get_backups_storage
    storage = get_backups_storage()

    if not storage.is_configured():
        raise Exception("R2 backups storage not configured")

    # Test connection
    conn_status = storage.test_connection()
    if not conn_status["connected"]:
        raise Exception(f"R2 connection failed: {conn_status.get('error', 'Unknown error')}")

    if progress_callback:
        progress_callback(5, "Preparing source files for upload...")

    # Upload based on backup type
    backup_path = source_info["backup_path"]
    backup_type = source_info["backup_type"]

    # Upload to backups/{source_id}/ folder (not submissions/)
    remote_folder = f"backups/{source_id}"
    uploaded_files = []
    total_size = 0

    if backup_type == "zim":
        if progress_callback:
            progress_callback(10, f"Uploading ZIM file ({source_info.get('backup_size_mb', 0)} MB)...")

        # Upload ZIM file
        remote_key = f"{remote_folder}/{source_id}.zim"
        success = storage.upload_file(backup_path, remote_key)

        if not success:
            raise Exception(f"Upload failed: {storage.get_last_error()}")

        uploaded_files.append(f"{source_id}.zim")
        total_size = source_info["backup_size_mb"]

    elif backup_type == "html":
        # Create zip of HTML folder and upload
        import tempfile
        import zipfile

        html_path = Path(backup_path)

        if progress_callback:
            progress_callback(10, "Creating zip archive...")

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                all_files = list(html_path.rglob('*'))
                file_count = len([f for f in all_files if f.is_file()])
                for i, file_path in enumerate(all_files):
                    if file_path.is_file():
                        arcname = file_path.relative_to(html_path)
                        zf.write(file_path, arcname)
                        if progress_callback and file_count > 0:
                            percent = 8 + int((i / file_count) * 10)  # 8-18%
                            progress_callback(percent, f"Zipping files... ({i+1}/{file_count})")

            if progress_callback:
                progress_callback(20, "Uploading zip file...")

            remote_key = f"{remote_folder}/{source_id}-html.zip"
            success = storage.upload_file(tmp_path, remote_key)

            if not success:
                raise Exception(f"Upload failed: {storage.get_last_error()}")

            uploaded_files.append(f"{source_id}-html.zip")
            total_size = os.path.getsize(tmp_path) / (1024*1024)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    elif backup_type == "pdf":
        # Create zip of PDF folder and upload
        import tempfile
        import zipfile

        pdf_path = Path(backup_path)

        if progress_callback:
            progress_callback(10, "Creating zip archive...")

        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                all_files = list(pdf_path.rglob('*'))
                file_count = len([f for f in all_files if f.is_file()])
                for i, file_path in enumerate(all_files):
                    if file_path.is_file():
                        arcname = file_path.relative_to(pdf_path)
                        zf.write(file_path, arcname)
                        if progress_callback and file_count > 0:
                            percent = 8 + int((i / file_count) * 10)  # 8-18%
                            progress_callback(percent, f"Zipping files... ({i+1}/{file_count})")

            if progress_callback:
                progress_callback(20, "Uploading zip file...")

            remote_key = f"{remote_folder}/{source_id}-pdf.zip"
            success = storage.upload_file(tmp_path, remote_key)

            if not success:
                raise Exception(f"Upload failed: {storage.get_last_error()}")

            uploaded_files.append(f"{source_id}-pdf.zip")
            total_size = os.path.getsize(tmp_path) / (1024*1024)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        raise Exception(f"Unknown backup type: {backup_type}")

    # Upload schema files (manifest, metadata, index, vectors)
    if progress_callback:
        progress_callback(25, "Uploading metadata files...")

    from .local_config import get_local_config
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if backup_folder:
        source_folder = Path(backup_folder) / source_id

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
                remote_key = f"{remote_folder}/{filename}"
                if storage.upload_file(str(file_path), remote_key):
                    uploaded_files.append(filename)
                    total_size += file_path.stat().st_size / (1024*1024)

    # Update _master.json in R2
    if progress_callback:
        progress_callback(30, "Updating master index in R2...")

    _update_r2_master_json(storage, source_id, source_info, backup_folder)

    # Sync to Pinecone
    if progress_callback:
        mode_text = "Replacing" if sync_mode == "replace" else "Syncing"
        progress_callback(35, f"{mode_text} vectors in Pinecone...")

    pinecone_result = _sync_source_to_pinecone(source_id, progress_callback, sync_mode=sync_mode)

    if progress_callback:
        progress_callback(100, "Publish complete")

    # Save last_published timestamp to manifest
    try:
        from datetime import datetime
        manifest_path = Path(backup_folder) / source_id / get_manifest_file()
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
            manifest_data["last_published"] = datetime.now().isoformat()
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, indent=2)
    except Exception as e:
        print(f"[publish] Warning: could not save last_published timestamp: {e}")

    return {
        "status": "success",
        "source_id": source_id,
        "mode": "publish",
        "bucket": "backups",
        "size_mb": round(total_size, 2),
        "file_count": len(uploaded_files),
        "files": uploaded_files,
        "pinecone": pinecone_result,
        "message": f"Published {source_id} to production ({len(uploaded_files)} files, Pinecone synced)"
    }


def _update_r2_master_json(storage, source_id: str, source_info: dict, backup_folder: str):
    """
    Download _master.json from R2, add/update source entry, re-upload.
    """
    import tempfile

    # Download existing master.json (or create new)
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name

    master_data = {"version": 2, "sources": {}, "total_documents": 0, "total_chars": 0}

    try:
        # Try to download existing master.json
        if storage.download_file("backups/_master.json", tmp_path):
            with open(tmp_path, 'r', encoding='utf-8') as f:
                master_data = json.load(f)
    except Exception:
        pass  # Use default empty master

    # Load source metadata for counts and manifest for license/name/etc
    doc_count = 0
    total_chars = 0
    manifest_data = {}

    if backup_folder:
        metadata_path = Path(backup_folder) / source_id / get_metadata_file()
        manifest_path = Path(backup_folder) / source_id / get_manifest_file()

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                doc_count = meta.get("document_count", len(meta.get("documents", {})))
                total_chars = meta.get("total_chars", 0)
            except Exception:
                pass

        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
            except Exception:
                pass

    # Update master with this source - include all important fields from manifest
    if "sources" not in master_data:
        master_data["sources"] = {}

    from datetime import datetime
    master_data["sources"][source_id] = {
        "name": manifest_data.get("name", source_id),
        "description": manifest_data.get("description", ""),
        "license": manifest_data.get("license", "Unknown"),
        "license_verified": manifest_data.get("license_verified", False),
        "tags": manifest_data.get("tags", []),
        "total_docs": doc_count,
        "count": doc_count,  # Alias for compatibility
        "total_chars": total_chars,
        "chars": total_chars,  # Alias for compatibility
        "version": manifest_data.get("version", "1.0.0"),
        "has_vectors": manifest_data.get("has_vectors", True),
        "has_backup": manifest_data.get("has_backup", True),
        "total_size_bytes": manifest_data.get("total_size_bytes", 0),
        "total_size_mb": manifest_data.get("total_size_bytes", 0) / (1024*1024) if manifest_data.get("total_size_bytes") else 0,
        "base_url": manifest_data.get("base_url", ""),
        "source_type": manifest_data.get("source_type", "unknown"),
        "published_at": datetime.now().isoformat()
    }

    # Recalculate totals
    master_data["total_documents"] = sum(s.get("count", 0) for s in master_data["sources"].values())
    master_data["total_chars"] = sum(s.get("chars", 0) for s in master_data["sources"].values())
    master_data["last_updated"] = datetime.now().isoformat()

    # Write updated master and upload
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(master_data, f, indent=2)

        if not storage.upload_file(tmp_path, "backups/_master.json"):
            raise Exception(f"Failed to upload _master.json: {storage.get_last_error()}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _sync_source_to_pinecone(source_id: str, progress_callback=None, sync_mode: str = "update") -> dict:
    """
    Sync a specific source's vectors to Pinecone.

    Args:
        source_id: The source to sync
        progress_callback: Optional callback for progress updates
        sync_mode: "update" (add/update only) or "replace" (delete old vectors first)

    Returns dict with sync stats.
    """
    try:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            return {"status": "skipped", "reason": "PINECONE_API_KEY not configured"}

        from offline_tools.vectordb import VectorStore, PineconeStore
        from offline_tools.vectordb.sync import SyncManager

        local = VectorStore()
        remote = PineconeStore()

        deleted_count = 0

        # If replace mode, delete existing vectors first
        if sync_mode == "replace":
            if progress_callback:
                progress_callback(40, f"Deleting old vectors for {source_id}...")

            delete_result = remote.delete_by_source(source_id)
            deleted_count = delete_result.get("deleted_count", 0)
            print(f"[sync] Deleted {deleted_count} old vectors for {source_id}")

        sync = SyncManager(local, remote)
        diff = sync.compare(source=source_id)

        total_vectors = len(diff.to_push) + len(diff.to_update)
        if progress_callback:
            progress_callback(45, f"Pushing {total_vectors} vectors to Pinecone...")

        # Create a wrapper callback to translate batch progress to overall job percent
        # Pinecone phase is 45-95% of overall job (this is the slow part)
        def batch_progress(batch_num, total_batches, message):
            if progress_callback and total_batches > 0:
                # Map batch progress (0-100%) to job progress (45-95%)
                batch_percent = (batch_num / total_batches) * 100
                job_percent = 45 + (batch_percent * 0.50)  # 45% + up to 50% = 95%
                progress_callback(int(job_percent), f"Uploading batch {batch_num}/{total_batches}")

        # Do the push (not dry run)
        stats = sync.push(diff, dry_run=False, update=True, force=False, progress_callback=batch_progress)

        return {
            "status": "success",
            "sync_mode": sync_mode,
            "deleted": deleted_count,
            "pushed": stats.get("pushed", 0),
            "updated": stats.get("updated", 0),
            "errors": stats.get("errors", 0)
        }

    except ImportError as e:
        return {"status": "skipped", "reason": f"Missing dependency: {str(e)}"}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


@router.post("/api/upload-backup")
async def upload_backup_to_cloud(
    request: UploadBackupRequest,
    _: bool = Depends(_require_global_admin)
):
    """
    Upload a source's backup file to R2 cloud storage.
    Only allows upload if source is complete (config + metadata + backup + verified license).
    Runs as a background job with progress tracking.

    Two modes based on publish_mode flag:
    - publish_mode=False (default): Submit to submissions bucket for review
    - publish_mode=True: Publish directly to backups bucket + update master.json + sync Pinecone

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

        if request.publish_mode:
            # Global admin: use backups bucket
            from offline_tools.cloud.r2 import get_backups_storage
            storage = get_backups_storage()
        else:
            # Local admin: use submissions bucket
            from offline_tools.cloud.r2 import get_submissions_storage
            storage = get_submissions_storage()

        if not storage.is_configured():
            bucket_type = "backups" if request.publish_mode else "submissions"
            raise HTTPException(500, f"R2 {bucket_type} storage not configured")

    except ImportError:
        raise HTTPException(500, "Storage module not available. Install boto3.")

    # Submit as background job
    from .job_manager import get_job_manager
    manager = get_job_manager()

    try:
        if request.publish_mode:
            # Global admin: publish to production
            # sync_mode: "update" (add/merge) or "replace" (delete old vectors first)
            job_id = manager.submit(
                "upload",
                request.source_id,
                _run_publish_to_production,
                request.source_id,
                source_info,
                request.sync_mode  # "update" or "replace"
            )
            mode_text = "Replacing" if request.sync_mode == "replace" else "Publishing"
            message = f"{mode_text} {request.source_id} in production..."
        else:
            # Submit for review
            job_id = manager.submit(
                "upload",
                request.source_id,
                _run_upload_backup,
                request.source_id,
                source_info
            )
            message = f"Submitting {request.source_id} for review..."

    except ValueError as e:
        # Job already running for this source
        raise HTTPException(409, str(e))

    return {
        "status": "submitted",
        "job_id": job_id,
        "source_id": request.source_id,
        "publish_mode": request.publish_mode,
        "message": message
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


@router.get("/api/discover-novel-tags")
async def discover_novel_tags(
    _: bool = Depends(_require_global_admin)
):
    """
    Discover tags used by sources that are not in TOPIC_KEYWORDS.

    REQUIRES: VECTOR_DB_MODE=global (Global Admin only)

    Scans all sources in the backup folder and returns a report of novel tags
    that users have chosen but which don't exist in the canonical TOPIC_KEYWORDS list.
    """
    try:
        from offline_tools.source_manager import SourceManager

        manager = SourceManager()
        result = manager.discover_novel_used_tags()

        return {
            "status": "success",
            "novel_tags": result["novel_tags"],
            "known_tags": result["known_tags"],
            "sources_scanned": result["sources_scanned"],
            "sources_with_tags": result["sources_with_tags"],
            "report": result["report"],
            "errors": result["errors"]
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "novel_tags": {},
            "known_tags": {},
            "sources_scanned": 0,
            "sources_with_tags": 0,
            "report": f"Error: {str(e)}",
            "errors": [str(e)]
        }


# ============================================================================
# Pinecone Sync Operations
# ============================================================================

class PineconeSyncRequest(BaseModel):
    """Request model for Pinecone sync operations"""
    source_id: Optional[str] = None  # Filter to specific source
    dry_run: bool = True  # Default to dry run for safety
    sync_mode: str = "update"  # "update" (add/update only) or "replace" (delete old first)


def pinecone_sync_job(dry_run: bool = True, source_filter: str = None,
                      progress_callback=None, cancel_checker=None):
    """
    Background job function for Pinecone sync operations.

    Args:
        dry_run: If True, only report what would be done without making changes
        source_filter: Optional source ID to filter sync to
        progress_callback: Function to report progress (current, total, message)
        cancel_checker: Function to check if job was cancelled

    Returns:
        Dict with sync results
    """
    import os
    from dotenv import load_dotenv
    load_dotenv()

    def update(current, total, msg):
        if progress_callback:
            progress_callback(current, total, msg)

    def is_cancelled():
        return cancel_checker() if cancel_checker else False

    update(0, 100, "Initializing...")

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        return {"success": False, "error": "PINECONE_API_KEY not configured"}

    if is_cancelled():
        return {"status": "cancelled"}

    try:
        from offline_tools.vectordb import VectorStore, PineconeStore
        from offline_tools.vectordb.sync import SyncManager

        update(5, 100, "Loading local vector store...")
        local = VectorStore()

        if is_cancelled():
            return {"status": "cancelled"}

        update(10, 100, "Connecting to Pinecone...")
        remote = PineconeStore()

        if is_cancelled():
            return {"status": "cancelled"}

        update(15, 100, "Comparing local and remote...")
        sync = SyncManager(local, remote)
        diff = sync.compare(source=source_filter)

        if is_cancelled():
            return {"status": "cancelled"}

        # Calculate totals for progress
        to_push_count = len(diff.to_push) if hasattr(diff, 'to_push') else 0
        to_update_count = len(diff.to_update) if hasattr(diff, 'to_update') else 0
        total_vectors = to_push_count + to_update_count

        update(40, 100, f"Found {to_push_count:,} new + {to_update_count:,} updates")

        # Check for conflicts
        if diff.conflicts:
            return {
                "success": False,
                "error": f"{len(diff.conflicts)} conflicts detected. Remote has newer versions.",
                "conflicts": diff.conflicts[:5]
            }

        # Nothing to push?
        if not diff.to_push and not diff.to_update:
            update(100, 100, "Already in sync - nothing to push")
            return {
                "success": True,
                "status": "success",
                "message": "Already in sync - nothing to push",
                "dry_run": dry_run,
                "pushed": 0,
                "updated": 0,
                "to_push": 0,
                "to_update": 0
            }

        # For dry run, we're done after comparison
        if dry_run:
            update(100, 100, f"Dry run: {to_push_count:,} to push, {to_update_count:,} to update")
            return {
                "success": True,
                "status": "success",
                "dry_run": True,
                "pushed": 0,
                "updated": 0,
                "to_push": to_push_count,
                "to_update": to_update_count,
                "message": f"Dry run: {to_push_count:,} to push, {to_update_count:,} to update"
            }

        if is_cancelled():
            return {"status": "cancelled"}

        # Do the actual push with progress callback
        update(50, total_vectors, f"Pushing {total_vectors:,} vectors to Pinecone...")

        # Create wrapper to show batch progress
        def batch_progress(batch_num, total_batches, message):
            if progress_callback and total_batches > 0:
                # Map batch progress to 50-90 range
                pct = 50 + int((batch_num / total_batches) * 40)
                update(pct, 100, f"Uploading batch {batch_num}/{total_batches} ({total_vectors:,} total)")

        stats = sync.push(diff, dry_run=False, update=True, force=False, progress_callback=batch_progress)

        pushed = stats.get("pushed", 0)
        updated = stats.get("updated", 0)
        update(90, 100, f"Pushed {pushed:,} + updated {updated:,} vectors")

        # Note: Visualization is NOT auto-triggered here.
        # Use cloud_publish_job for combined sync + visualization.
        final_msg = f"Complete: {pushed:,} pushed, {updated:,} updated"
        update(100, 100, final_msg)

        return {
            "success": True,
            "status": "success",
            "dry_run": False,
            "pushed": pushed,
            "updated": updated,
            "skipped": stats.get("skipped", 0),
            "errors": stats.get("errors", 0),
            "message": final_msg
        }

    except ImportError as e:
        return {"success": False, "error": f"Missing dependency: {str(e)}"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def cloud_publish_job(
    dry_run: bool = False,
    source_filter: str = None,
    progress_callback=None,
    cancel_checker=None
) -> Dict[str, Any]:
    """
    Combined job: Pinecone sync + Visualization generation.

    Uses the combined job framework to run both operations as a single job
    with unified progress tracking.

    Args:
        dry_run: If True, only check what would sync (no visualization)
        source_filter: Optional source ID to filter sync to
        progress_callback: Function for progress updates
        cancel_checker: Function to check for cancellation

    Returns:
        Dict with combined results from both phases
    """
    from .job_manager import JobPhase, run_combined_job
    from .routes.visualise import generate_and_publish_visualisation_job

    # For dry run, only run sync phase
    if dry_run:
        return pinecone_sync_job(
            dry_run=True,
            source_filter=source_filter,
            progress_callback=progress_callback,
            cancel_checker=cancel_checker
        )

    # Define phases: Sync (70%) + Visualization (30%)
    phases = [
        JobPhase(
            name="Sync to Pinecone",
            func=pinecone_sync_job,
            weight=70,
            kwargs={"dry_run": False, "source_filter": source_filter}
        ),
        JobPhase(
            name="Generate Visualization",
            func=generate_and_publish_visualisation_job,
            weight=30
        )
    ]

    return run_combined_job(
        phases=phases,
        progress_callback=progress_callback,
        cancel_checker=cancel_checker
    )


@router.get("/api/pinecone-check-source/{source_id}")
async def pinecone_check_source(source_id: str):
    """
    Check if a source already exists in Pinecone.

    Returns info about existing vectors for conflict detection before upload.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            return {"exists": False, "configured": False, "error": "Pinecone not configured"}

        from offline_tools.vectordb import PineconeStore
        store = PineconeStore()

        # Get count of vectors for this source
        vector_count = store.get_source_vector_count(source_id)

        if vector_count > 0:
            return {
                "exists": True,
                "configured": True,
                "source_id": source_id,
                "vector_count": vector_count,
                "message": f"Source '{source_id}' has {vector_count} vectors in Pinecone"
            }
        else:
            return {
                "exists": False,
                "configured": True,
                "source_id": source_id,
                "vector_count": 0,
                "message": f"Source '{source_id}' not found in Pinecone"
            }

    except Exception as e:
        return {"exists": False, "configured": True, "error": str(e)}


@router.delete("/api/pinecone-source/{source_id}")
async def pinecone_delete_source(
    source_id: str,
    _: bool = Depends(_require_global_admin)
):
    """
    Delete all vectors for a specific source from Pinecone.

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

        # Delete vectors for this source
        result = store.delete_by_source(source_id)

        return {
            "status": "success",
            "source_id": source_id,
            "deleted_count": result["deleted_count"],
            "batches": result["batches"],
            "message": f"Deleted {result['deleted_count']} vectors for source '{source_id}'"
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


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
    Push local ChromaDB changes to Pinecone (runs as background job).
    Uploads new and updated documents from local to remote.

    For actual push (not dry_run), uses cloud_publish_job which combines:
    - Pinecone sync (70% of progress)
    - Visualization regeneration (30% of progress)

    Returns a job_id to poll for progress.

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

        from .job_manager import get_job_manager

        manager = get_job_manager()

        # Check if a related job is already running
        active_jobs = manager.get_active_jobs()
        existing_job = next(
            (job for job in active_jobs if job.get("job_type") in ["pinecone_sync", "cloud_publish"]),
            None
        )
        if existing_job:
            return JSONResponse(
                status_code=409,
                content={
                    "error": f"A {existing_job.get('job_type')} job is already running",
                    "job_id": existing_job.get("id")
                }
            )

        # Submit the background job
        # Dry run: just check sync status
        # Actual push: combined job (sync + visualization)
        if dry_run:
            job_type = "pinecone_sync_dry"
            job_id = manager.submit(
                job_type=job_type,
                source_id="_pinecone",
                func=pinecone_sync_job,
                dry_run=True,
                source_filter=source_filter
            )
            message = "Dry run job started"
        else:
            job_type = "cloud_publish"
            job_id = manager.submit(
                job_type=job_type,
                source_id="_pinecone",
                func=cloud_publish_job,
                dry_run=False,
                source_filter=source_filter
            )
            message = "Cloud publish job started (sync + visualization)"

        return {
            "status": "started",
            "job_id": job_id,
            "dry_run": dry_run,
            "job_type": job_type,
            "message": message
        }

    except ValueError as e:
        # Job conflict
        return JSONResponse(
            status_code=409,
            content={"error": str(e)}
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
