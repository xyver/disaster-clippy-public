"""
Cloud Backup Upload Routes
API endpoints for uploading verified source backups to R2 cloud storage
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path
import os
import json

router = APIRouter()

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
        from storage.r2 import get_r2_storage
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


@router.get("/api/sources-for-upload")
async def get_sources_for_upload():
    """
    Get list of local sources that could be uploaded to cloud.
    Checks completeness: needs source config, metadata, and backup file.
    """
    backup_paths = get_backup_paths()

    # Load sources config
    sources_path = Path(__file__).parent.parent / "config" / "sources.json"
    if not sources_path.exists():
        return {"sources": [], "error": "No sources.json found"}

    with open(sources_path, 'r', encoding='utf-8') as f:
        sources_data = json.load(f)

    sources_config = sources_data.get("sources", {})

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

        # Check for metadata
        metadata_path = Path(__file__).parent.parent / "data" / "metadata" / f"{source_id}.json"
        if metadata_path.exists():
            source_status["has_metadata"] = True
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    source_status["document_count"] = meta.get("total_documents", 0)
            except:
                pass
        else:
            source_status["missing"].append("metadata")

        # Check for backup files (ZIM or HTML folder)
        backup_found = False

        # Check ZIM folder
        if backup_paths.get("zim_folder"):
            zim_path = Path(backup_paths["zim_folder"]) / f"{source_id}.zim"
            if zim_path.exists():
                backup_found = True
                source_status["has_backup"] = True
                source_status["backup_type"] = "zim"
                source_status["backup_path"] = str(zim_path)
                source_status["backup_size_mb"] = round(zim_path.stat().st_size / (1024*1024), 2)

        # Check HTML folder
        if not backup_found and backup_paths.get("html_folder"):
            html_path = Path(backup_paths["html_folder"]) / source_id
            if html_path.exists() and html_path.is_dir():
                backup_found = True
                source_status["has_backup"] = True
                source_status["backup_type"] = "html"
                source_status["backup_path"] = str(html_path)
                # Calculate folder size
                try:
                    total_size = sum(f.stat().st_size for f in html_path.rglob('*') if f.is_file())
                    source_status["backup_size_mb"] = round(total_size / (1024*1024), 2)
                except:
                    pass

        if not backup_found:
            source_status["missing"].append("backup file")
        elif source_status["backup_size_mb"] < 0.1:
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


@router.post("/api/upload-backup")
async def upload_backup_to_cloud(request: UploadBackupRequest):
    """
    Upload a source's backup file to R2 cloud storage.
    Only allows upload if source is complete (config + metadata + backup + verified license).
    """
    from dotenv import load_dotenv
    load_dotenv()

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

    # Get R2 storage
    try:
        from storage.r2 import get_r2_storage
        storage = get_r2_storage()

        if not storage.is_configured():
            raise HTTPException(500, "R2 cloud storage not configured")

        # Test connection
        conn_status = storage.test_connection()
        if not conn_status["connected"]:
            raise HTTPException(500, f"R2 connection failed: {conn_status.get('error', 'Unknown error')}")

    except ImportError:
        raise HTTPException(500, "Storage module not available. Install boto3.")

    # Upload based on backup type
    backup_path = source_info["backup_path"]
    backup_type = source_info["backup_type"]

    # Generate submission folder with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_folder = f"submissions/{timestamp}_{request.source_id}"

    if backup_type == "zim":
        # Upload single ZIM file
        remote_key = f"{submission_folder}/{request.source_id}.zim"
        success = storage.upload_file(backup_path, remote_key)

        if success:
            # Also upload metadata
            await _upload_submission_metadata(storage, submission_folder, request.source_id, source_info)
            return {
                "status": "success",
                "source_id": request.source_id,
                "remote_key": remote_key,
                "size_mb": source_info["backup_size_mb"],
                "message": f"Submitted {request.source_id}.zim for review"
            }
        else:
            raise HTTPException(500, f"Upload failed: {storage.get_last_error()}")

    elif backup_type == "html":
        # For HTML folders, create a zip and upload
        import tempfile
        import zipfile

        html_path = Path(backup_path)

        # Create temporary zip file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Create zip of HTML folder
            with zipfile.ZipFile(tmp_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in html_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(html_path)
                        zf.write(file_path, arcname)

            # Upload zip
            remote_key = f"{submission_folder}/{request.source_id}-html.zip"
            success = storage.upload_file(tmp_path, remote_key)

            if success:
                # Get actual upload size
                zip_size = round(os.path.getsize(tmp_path) / (1024*1024), 2)
                # Also upload metadata
                await _upload_submission_metadata(storage, submission_folder, request.source_id, source_info)
                return {
                    "status": "success",
                    "source_id": request.source_id,
                    "remote_key": remote_key,
                    "size_mb": zip_size,
                    "message": f"Submitted {request.source_id} HTML archive for review"
                }
            else:
                raise HTTPException(500, f"Upload failed: {storage.get_last_error()}")

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    else:
        raise HTTPException(400, f"Unknown backup type: {backup_type}")


async def _upload_submission_metadata(storage, submission_folder: str, source_id: str, source_info: dict):
    """Upload metadata JSON and manifest alongside the backup submission"""
    import tempfile

    # Load source config
    sources_path = Path(__file__).parent.parent / "config" / "sources.json"
    source_config = {}
    if sources_path.exists():
        with open(sources_path, 'r', encoding='utf-8') as f:
            sources_data = json.load(f)
            source_config = sources_data.get("sources", {}).get(source_id, {})

    # Load full metadata
    metadata_path = Path(__file__).parent.parent / "data" / "metadata" / f"{source_id}.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        # Upload the FULL metadata.json file
        storage.upload_file(str(metadata_path), f"{submission_folder}/metadata.json")

    # Create submission manifest (for tracking status, not the pack manifest)
    from datetime import datetime
    submission_manifest = {
        "source_id": source_id,
        "submitted_at": datetime.now().isoformat(),
        "status": "pending_review",
        "source_config": source_config,
        "metadata_summary": {
            "total_documents": metadata.get("total_documents", 0),
            "source_type": metadata.get("source_type", "unknown")
        },
        "backup_info": {
            "type": source_info.get("backup_type"),
            "size_mb": source_info.get("backup_size_mb", 0)
        }
    }

    # Write to temp file and upload
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
        json.dump(submission_manifest, tmp, indent=2)
        tmp_path = tmp.name

    try:
        storage.upload_file(tmp_path, f"{submission_folder}/manifest.json")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


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
        from storage.r2 import get_r2_storage
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
