"""
Source Tools API

Local admin endpoints for managing, indexing, and validating source packs.
These endpoints are used by the Source Tools wizard in the admin dashboard.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import json

from offline_tools.schemas import (
    get_manifest_file, get_metadata_file, get_index_file,
    get_vectors_file, get_backup_manifest_file
)

router = APIRouter(prefix="/api", tags=["Source Tools"])


# =============================================================================
# REQUEST MODELS
# =============================================================================

class UpdateSourceConfigRequest(BaseModel):
    source_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    license: Optional[str] = None
    base_url: Optional[str] = None
    tags: Optional[List[str]] = None
    license_verified: Optional[bool] = None


class CreateSourceRequest(BaseModel):
    source_id: str
    source_type: str  # html, zim, pdf, scrape
    url: Optional[str] = None
    import_path: Optional[str] = None


class SourceIdRequest(BaseModel):
    source_id: str


class GenerateMetadataRequest(BaseModel):
    source_id: str
    language_filter: Optional[str] = None  # ISO code like 'en', 'es' to filter ZIM articles
    resume: bool = False  # If True, resume from checkpoint if available


class CreateIndexRequest(BaseModel):
    source_id: str
    limit: int = 1000
    force_reindex: bool = False
    language_filter: Optional[str] = None  # ISO code like 'en', 'es' to filter ZIM articles
    resume: bool = False  # If True, resume from checkpoint if available


class ValidateSourceRequest(BaseModel):
    source_id: str
    require_v2: bool = False


class RenameSourceRequest(BaseModel):
    old_source_id: str
    new_source_id: str


class DeleteSourceRequest(BaseModel):
    source_id: str
    delete_files: bool = True


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_local_config():
    """Get local config - imported here to avoid circular imports"""
    from admin.local_config import get_local_config as _get_config
    return _get_config()


# =============================================================================
# SOURCE LISTING & STATUS
# =============================================================================

@router.get("/local-sources")
async def get_local_sources():
    """
    Get ALL local sources with completeness status.
    Source of truth is the backup folder - discovers all sources from
    _manifest.json files in each source subfolder.
    """
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    # Primary discovery: scan backup folder for all sources
    sources_config = {}

    if backup_folder:
        backup_path = Path(backup_folder)
        if backup_path.exists():
            for source_folder in backup_path.iterdir():
                if source_folder.is_dir():
                    source_id = source_folder.name

                    # Check for manifest file
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
                                "tags": source_data.get("tags", []),
                            }
                        except Exception:
                            sources_config[source_id] = {"name": source_id, "tags": []}

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
            "tags": config_data.get("tags", []),
            "has_config": True,
            "has_metadata": False,
            "has_backup": False,
            "has_embeddings": False,
            "backup_type": None,
            "backup_size_mb": 0,
            "document_count": 0,
            "is_complete": False,
            "production_ready": False,
            "missing": [],
            "schema_version": 1,
            "has_source_metadata": False,
            "has_documents_file": False,
            "has_embeddings_file": False,
        }

        # Check for metadata file
        metadata_path = Path(backup_folder) / source_id / get_metadata_file() if backup_folder else None

        if metadata_path and metadata_path.exists():
            source_status["has_metadata"] = True
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                source_status["document_count"] = meta.get("document_count", meta.get("total_documents", 0))
            except Exception:
                pass
        else:
            source_status["missing"].append("metadata")

        # For ZIM sources without metadata yet, get article count from ZIM header
        if source_status["document_count"] == 0 and backup_folder:
            source_folder = Path(backup_folder) / source_id
            zim_files = list(source_folder.glob("*.zim")) if source_folder.exists() else []
            if zim_files:
                try:
                    from offline_tools.zim_utils import get_zim_metadata
                    zim_meta = get_zim_metadata(str(zim_files[0]))
                    if "error" not in zim_meta and zim_meta.get("article_count"):
                        source_status["document_count"] = zim_meta["article_count"]
                        source_status["document_count_source"] = "zim_header"  # Indicates this is from ZIM, not metadata
                except Exception:
                    pass

        # Check for backup files using unified detection
        from offline_tools.packager import detect_backup_status
        backup_status = detect_backup_status(source_id, Path(backup_folder) if backup_folder else None)
        source_status["has_backup"] = backup_status["has_backup"]
        source_status["backup_type"] = backup_status["backup_type"]
        source_status["backup_size_mb"] = backup_status["backup_size_mb"]

        if not backup_status["has_backup"]:
            source_status["missing"].append("backup file")
        elif backup_status["backup_size_mb"] < 0.1:
            source_status["has_backup"] = False
            source_status["missing"].append("backup file (empty)")

        # Check for schema files
        if backup_folder:
            source_folder = Path(backup_folder) / source_id
            manifest_file = source_folder / get_manifest_file()
            metadata_file = source_folder / get_metadata_file()
            vectors_file = source_folder / get_vectors_file()

            source_status["has_manifest"] = manifest_file.exists()
            source_status["has_metadata_file"] = metadata_file.exists()
            source_status["has_vectors_file"] = vectors_file.exists()

            # Determine schema version and embeddings status
            if source_status["has_manifest"] and source_status["has_metadata_file"] and source_status["has_vectors_file"]:
                source_status["schema_version"] = 3
                source_status["has_embeddings"] = True
            else:
                # Check index file as fallback
                index_path = source_folder / get_index_file()
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

        # Try to get base_url from manifest if not in config
        if not source_status["base_url"] and backup_folder:
            source_folder = Path(backup_folder) / source_id
            manifest_file = source_folder / get_manifest_file()
            if manifest_file.exists():
                try:
                    with open(manifest_file, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)
                    if manifest.get("base_url"):
                        source_status["base_url"] = manifest["base_url"]
                except Exception:
                    pass
            if not source_status["base_url"]:
                backup_manifest = source_folder / get_backup_manifest_file()
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


# =============================================================================
# SOURCE CONFIGURATION
# =============================================================================

@router.post("/update-source-config")
async def update_source_config(request: UpdateSourceConfigRequest):
    """Update source configuration in _manifest.json"""
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise HTTPException(400, "No backup folder configured")

    manifest_file = Path(backup_folder) / request.source_id / get_manifest_file()

    # Load existing source config or create new one
    if manifest_file.exists():
        try:
            with open(manifest_file, 'r', encoding='utf-8') as f:
                source = json.load(f)
        except Exception:
            source = {"source_id": request.source_id}
    else:
        source_folder = Path(backup_folder) / request.source_id
        if not source_folder.exists():
            raise HTTPException(404, f"Source folder not found: {request.source_id}")
        source = {"schema_version": 3, "source_id": request.source_id}

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

    # Save to _manifest.json
    try:
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(source, f, indent=2)

        # Update _master.json if tags changed
        if request.tags is not None:
            try:
                from offline_tools.packager import sync_master_metadata
                sync_master_metadata()
            except Exception as e:
                print(f"Warning: Could not sync master after tag update: {e}")

        return {"status": "success", "source_id": request.source_id}
    except Exception as e:
        raise HTTPException(500, f"Failed to save source config: {e}")


@router.get("/auto-detect/{source_id}")
async def auto_detect(source_id: str):
    """Auto-detect license, description, tags, and URL from source content"""
    try:
        from offline_tools.source_manager import SourceManager

        config = get_local_config()
        backup_folder = config.get_backup_folder()

        result = {
            "source_id": source_id,
            "detected_license": None,
            "detected_name": None,
            "detected_description": None,
            "suggested_tags": [],
            "base_url": None,
            "source_type": None
        }

        # Check if this is a ZIM source
        if backup_folder:
            source_path = Path(backup_folder) / source_id
            zim_files = list(source_path.glob("*.zim")) if source_path.exists() else []

            if zim_files:
                result["source_type"] = "zim"
                # Use ZIM metadata extraction
                try:
                    from offline_tools.zim_utils import get_zim_metadata
                    zim_meta = get_zim_metadata(str(zim_files[0]))

                    if "error" not in zim_meta:
                        if zim_meta.get("title"):
                            result["detected_name"] = zim_meta["title"]
                        if zim_meta.get("description"):
                            result["detected_description"] = zim_meta["description"]
                        if zim_meta.get("license"):
                            result["detected_license"] = zim_meta["license"]
                        if zim_meta.get("source_url"):
                            result["base_url"] = zim_meta["source_url"]
                        if zim_meta.get("tags"):
                            result["suggested_tags"] = zim_meta["tags"]

                        # Include raw ZIM metadata for reference
                        result["zim_metadata"] = zim_meta
                except ImportError:
                    pass  # zimply-core not installed

        # Fallback to SourceManager detection for non-ZIM or if ZIM detection failed
        manager = SourceManager()

        if not result["detected_license"]:
            detected_license = manager.detect_license(source_id)
            if detected_license:
                result["detected_license"] = detected_license

        if not result["suggested_tags"]:
            suggested_tags = manager.suggest_tags(source_id)
            if suggested_tags:
                result["suggested_tags"] = suggested_tags

        return result

    except Exception as e:
        raise HTTPException(500, f"Auto-detect failed: {e}")


# =============================================================================
# SOURCE CREATION
# =============================================================================

@router.post("/create-source")
async def create_source(request: CreateSourceRequest):
    """
    Create a new source with initial configuration.
    """
    import re

    source_id = request.source_id.strip().lower()

    if not re.match(r'^[a-z0-9][a-z0-9_-]*$', source_id):
        raise HTTPException(400, "Invalid source ID format. Use lowercase letters, numbers, underscores, hyphens.")

    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise HTTPException(400, "Backup folder not configured. Please set it in Settings first.")

    source_path = Path(backup_folder) / source_id

    if source_path.exists():
        raise HTTPException(409, f"Source '{source_id}' already exists")

    try:
        source_path.mkdir(parents=True, exist_ok=True)

        backup_type = request.source_type
        if backup_type == "scrape":
            backup_type = "html"

        source_config = {
            "source_id": source_id,
            "name": source_id.replace("_", " ").replace("-", " ").title(),
            "description": "",
            "license": "Unknown",
            "license_verified": False,
            "base_url": request.url or "",
            "backup_type": backup_type,
            "created_at": datetime.now().isoformat()
        }

        manifest_file = source_path / get_manifest_file()
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(source_config, f, indent=2)

        import_info = None
        zim_moved = False

        # Handle ZIM file import - move/copy into source folder
        if request.source_type == "zim" and request.import_path:
            import shutil

            # Handle both absolute and relative paths
            import_path = Path(request.import_path)
            if import_path.is_absolute():
                import_full_path = import_path
            else:
                import_full_path = Path(backup_folder) / request.import_path

            # Find the ZIM file
            zim_source = None
            if import_full_path.exists():
                if import_full_path.suffix.lower() == ".zim":
                    # Direct path to ZIM file
                    zim_source = import_full_path
                elif import_full_path.is_dir():
                    # Folder containing ZIM file
                    zim_files = list(import_full_path.glob("*.zim"))
                    if zim_files:
                        zim_source = zim_files[0]

            if zim_source and zim_source.exists():
                # Move ZIM file into source folder
                zim_dest = source_path / f"{source_id}.zim"
                try:
                    shutil.move(str(zim_source), str(zim_dest))
                    zim_moved = True
                    import_info = {
                        "path": str(zim_dest),
                        "original": str(zim_source),
                        "moved": True,
                        "size_mb": round(zim_dest.stat().st_size / (1024*1024), 2)
                    }

                    # Auto-extract ZIM metadata and update manifest
                    try:
                        from offline_tools.zim_utils import get_zim_metadata
                        zim_meta = get_zim_metadata(str(zim_dest))
                        if "error" not in zim_meta:
                            # Update source config with ZIM metadata
                            if zim_meta.get("title"):
                                source_config["name"] = zim_meta["title"]
                            if zim_meta.get("description"):
                                source_config["description"] = zim_meta["description"]
                            if zim_meta.get("license"):
                                source_config["license"] = zim_meta["license"]
                            if zim_meta.get("source_url"):
                                source_config["base_url"] = zim_meta["source_url"]
                            # Save updated manifest
                            with open(manifest_file, 'w', encoding='utf-8') as f:
                                json.dump(source_config, f, indent=2)
                            import_info["metadata_extracted"] = True
                    except ImportError:
                        pass  # zimply-core not installed

                except Exception as move_err:
                    import_info = {
                        "path": str(zim_source),
                        "exists": True,
                        "move_error": str(move_err)
                    }
            else:
                import_info = {
                    "path": str(import_full_path),
                    "exists": False,
                    "error": "ZIM file not found at specified path"
                }

        elif request.import_path:
            # Non-ZIM import path handling (existing behavior)
            import_full_path = Path(backup_folder) / request.import_path
            if import_full_path.exists():
                import_info = {"path": str(import_full_path), "exists": True}
                if import_full_path != source_path:
                    import_info["note"] = f"Files exist at {request.import_path}. Use Scan Backup to import."

        # Sync master to add new source
        try:
            from offline_tools.packager import sync_master_metadata
            sync_master_metadata()
        except Exception as e:
            print(f"Warning: Could not sync master after source creation: {e}")

        return {
            "status": "success",
            "source_id": source_id,
            "source_type": request.source_type,
            "backup_type": backup_type,
            "path": str(source_path),
            "import_info": import_info,
            "zim_moved": zim_moved,
            "message": f"Source '{source_id}' created successfully" + (" (ZIM file imported)" if zim_moved else "")
        }

    except Exception as e:
        if source_path.exists():
            import shutil
            shutil.rmtree(source_path, ignore_errors=True)
        raise HTTPException(500, f"Failed to create source: {e}")


# =============================================================================
# INDEXING TOOLS
# =============================================================================

@router.post("/scan-backup")
async def scan_backup(request: SourceIdRequest):
    """Scan the backup folder and create a manifest"""
    try:
        from offline_tools.source_manager import SourceManager

        manager = SourceManager()
        result = manager.scan_backup(request.source_id)

        if not result.get("success", True):
            raise HTTPException(400, result.get("error", "Scan failed"))

        return {
            "status": "success",
            "source_id": request.source_id,
            "file_count": result.get("file_count", 0),
            "backup_type": result.get("backup_type"),
            "message": f"Scanned {result.get('file_count', 0)} files"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Scan failed: {e}")


@router.post("/generate-metadata")
async def generate_metadata(request: GenerateMetadataRequest):
    """
    Generate metadata from backup files.
    Runs as a background job since large sources can take a while.

    For ZIM files, you can optionally filter by language (ISO code like 'en', 'es').
    If resume=True and a checkpoint exists, continues from where it left off.
    """
    from admin.job_manager import get_job_manager

    def _run_generate_metadata(source_id: str, language_filter: str = None,
                               resume: bool = False, progress_callback=None,
                               cancel_checker=None):
        from offline_tools.source_manager import SourceManager
        manager = SourceManager()
        result = manager.generate_metadata(
            source_id,
            progress_callback=progress_callback,
            language_filter=language_filter,
            resume=resume,
            cancel_checker=cancel_checker
        )

        # Handle cancellation
        if result.get("cancelled"):
            return {"status": "cancelled", "error": "Job cancelled by user", "document_count": result.get("document_count", 0)}

        if not result.get("success", True):
            return {"status": "error", "error": result.get("error", "Metadata generation failed")}

        lang_info = ""
        if result.get("language_filtered_count", 0) > 0:
            lang_info = f" ({result['language_filtered_count']} filtered by language)"

        resumed_info = " (resumed)" if result.get("resumed") else ""

        return {
            "status": "success",
            "source_id": source_id,
            "document_count": result.get("document_count", 0),
            "language_filter": result.get("language_filter"),
            "language_filtered_count": result.get("language_filtered_count", 0),
            "resumed": result.get("resumed", False),
            "message": f"Generated metadata for {result.get('document_count', 0)} documents{lang_info}{resumed_info}"
        }

    manager = get_job_manager()

    try:
        job_id = manager.submit(
            "metadata",
            request.source_id,
            _run_generate_metadata,
            request.source_id,
            request.language_filter,
            request.resume
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    lang_msg = f" (language: {request.language_filter})" if request.language_filter else ""
    return {
        "status": "submitted",
        "job_id": job_id,
        "source_id": request.source_id,
        "language_filter": request.language_filter,
        "message": f"Metadata generation started{lang_msg}"
    }


@router.post("/create-index")
async def create_index(request: CreateIndexRequest):
    """Create vector embeddings for a source (runs as background job)

    Supports checkpoint-based resumption for ZIM files. If resume=True
    and a checkpoint exists, continues from where it left off.
    """
    from admin.job_manager import get_job_manager

    def _run_create_index(source_id: str, limit: int, force: bool,
                          language_filter: str = None, resume: bool = False,
                          progress_callback=None, cancel_checker=None):
        from offline_tools.source_manager import SourceManager
        manager = SourceManager()
        result = manager.create_index(
            source_id,
            limit=limit,
            skip_existing=not force,  # force=True means skip_existing=False
            progress_callback=progress_callback,
            language_filter=language_filter,
            resume=resume,
            cancel_checker=cancel_checker
        )

        # Handle cancelled jobs
        if hasattr(result, 'error') and result.error == "Cancelled by user":
            return {"status": "cancelled", "error": "Job cancelled by user", "indexed_count": result.indexed_count}

        return result

    manager = get_job_manager()

    try:
        job_id = manager.submit(
            "index",
            request.source_id,
            _run_create_index,
            request.source_id,
            request.limit,
            request.force_reindex,
            request.language_filter,
            request.resume
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    lang_msg = f" (language: {request.language_filter})" if request.language_filter else ""
    resume_msg = " (resuming)" if request.resume else ""
    return {
        "status": "submitted",
        "job_id": job_id,
        "source_id": request.source_id,
        "limit": request.limit,
        "force_reindex": request.force_reindex,
        "language_filter": request.language_filter,
        "resume": request.resume,
        "message": f"Index creation started for {request.source_id}{lang_msg}{resume_msg}"
    }


# =============================================================================
# VALIDATION & CLEANUP
# =============================================================================

@router.post("/validate-source")
async def validate_source(request: ValidateSourceRequest):
    """
    Validate source completeness and readiness for distribution.

    Returns validation results including actionable_issues - a list of issues
    with hints about which wizard step/tool can fix them.
    """
    try:
        from offline_tools.source_manager import SourceManager

        manager = SourceManager()
        result = manager.validate_source(request.source_id)

        # Convert actionable_issues to dicts for JSON serialization
        actionable = [issue.to_dict() for issue in result.actionable_issues] if result.actionable_issues else []

        return {
            "status": "success",
            "source_id": request.source_id,
            "is_valid": result.is_valid,
            "production_ready": result.production_ready,
            "has_backup": result.has_backup,
            "has_metadata": result.has_metadata,
            "has_manifest": result.has_manifest,
            "has_metadata_file": result.has_metadata_file,
            "has_vectors_file": result.has_vectors_file,
            "has_embeddings": result.has_embeddings,
            "has_license": result.has_license,
            "has_tags": result.has_tags,
            "license_verified": result.license_verified,
            "schema_version": result.schema_version,
            "issues": result.issues,
            "warnings": result.warnings,
            "detected_license": result.detected_license,
            "suggested_tags": result.suggested_tags,
            # New actionable issues with fix hints
            "actionable_issues": actionable
        }

    except Exception as e:
        raise HTTPException(500, f"Validation failed: {e}")


# =============================================================================
# CHECKPOINT SYSTEM
# =============================================================================

@router.get("/check-checkpoint/{source_id}/{job_type}")
async def check_checkpoint(source_id: str, job_type: str):
    """
    Check if a checkpoint exists for a source/job_type combination.

    Used by frontend to show "Resume or Restart?" modal.
    """
    try:
        from admin.job_manager import load_checkpoint

        checkpoint = load_checkpoint(source_id, job_type)

        if checkpoint:
            return {
                "has_checkpoint": True,
                "checkpoint": {
                    "source_id": checkpoint.source_id,
                    "job_type": checkpoint.job_type,
                    "progress": checkpoint.progress,
                    "last_saved": checkpoint.last_saved,
                    "created_at": checkpoint.created_at,
                    "last_article_index": checkpoint.last_article_index,
                    "documents_processed": checkpoint.documents_processed
                }
            }
        else:
            return {"has_checkpoint": False}

    except Exception as e:
        # If checkpoint system not available, just say no checkpoint
        return {"has_checkpoint": False, "error": str(e)}


@router.get("/interrupted-jobs")
async def get_interrupted_jobs():
    """
    Get all interrupted jobs (checkpoint files that can be resumed).

    Used by Jobs page to show "Interrupted Jobs" section.
    """
    try:
        from admin.job_manager import get_interrupted_jobs

        jobs = get_interrupted_jobs()
        return {"jobs": jobs}

    except Exception as e:
        return {"jobs": [], "error": str(e)}


@router.post("/discard-checkpoint/{source_id}/{job_type}")
async def discard_checkpoint(source_id: str, job_type: str):
    """
    Discard a checkpoint and its partial work file.

    Used when user chooses "Restart" instead of "Resume".
    """
    try:
        from admin.job_manager import delete_checkpoint

        deleted = delete_checkpoint(source_id, job_type)
        return {
            "status": "success" if deleted else "not_found",
            "deleted": deleted
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to discard checkpoint: {e}")


@router.post("/cleanup-stale-checkpoints")
async def cleanup_stale_checkpoints():
    """
    Delete checkpoint files older than 7 days.

    Manual cleanup option from Jobs page.
    """
    try:
        from admin.job_manager import cleanup_stale_checkpoints

        result = cleanup_stale_checkpoints(max_age_days=7)
        return {
            "status": "success",
            "deleted": result["deleted"],
            "errors": result["errors"]
        }

    except Exception as e:
        raise HTTPException(500, f"Cleanup failed: {e}")


@router.post("/cleanup-redundant-files")
async def cleanup_redundant_files(request: SourceIdRequest):
    """Delete redundant legacy files that have v2 equivalents"""
    try:
        from offline_tools.source_manager import SourceManager

        manager = SourceManager()
        result = manager.cleanup_redundant_files(request.source_id)

        if result["errors"]:
            return {
                "status": "partial",
                "source_id": request.source_id,
                "deleted": result["deleted"],
                "errors": result["errors"],
                "freed_mb": result.get("freed_mb", 0)
            }

        return {
            "status": "success",
            "source_id": request.source_id,
            "deleted": result["deleted"],
            "freed_mb": result.get("freed_mb", 0),
            "message": f"Cleaned up {len(result['deleted'])} file(s), freed {result.get('freed_mb', 0)} MB"
        }

    except Exception as e:
        raise HTTPException(500, f"Cleanup failed: {e}")


# =============================================================================
# RENAME & DELETE
# =============================================================================

@router.post("/rename-source")
async def rename_source(request: RenameSourceRequest):
    """Rename a source ID and all associated files"""
    try:
        from offline_tools.source_manager import SourceManager

        manager = SourceManager()
        result = manager.rename_source(request.old_source_id, request.new_source_id)

        if not result.get("success"):
            raise HTTPException(400, result.get("error", "Rename failed"))

        return {
            "status": "success",
            "old_source_id": result["old_source_id"],
            "new_source_id": result["new_source_id"],
            "renamed_files": result.get("renamed_files", []),
            "updated_json": result.get("updated_json", []),
            "master_updated": result.get("master_updated", False),
            "chromadb_updated": result.get("chromadb_updated", False),
            "chromadb_documents": result.get("chromadb_documents", 0),
            "errors": result.get("errors", []),
            "message": result.get("message", f"Renamed {request.old_source_id} to {request.new_source_id}")
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Rename failed: {e}")


@router.post("/delete-source")
async def delete_source(request: DeleteSourceRequest):
    """Completely remove a source from the system"""
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
                with open(master_file, 'r', encoding='utf-8') as f:
                    master_data = json.load(f)

                if source_id in master_data.get("sources", {}):
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
        from offline_tools.vectordb import get_vector_store
        store = get_vector_store()
        # Note: metadata field is "source", not "source_id"
        result = store.delete_by_source(source_id)
        deleted_count = result.get("deleted_count", 0)
        if deleted_count > 0:
            deleted_items.append(f"ChromaDB entries ({deleted_count} docs)")
    except Exception as e:
        errors.append(f"Failed to remove from ChromaDB: {e}")

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
# ZIM FILE TOOLS
# =============================================================================

class ZIMInspectRequest(BaseModel):
    """Request model for ZIM inspection"""
    zim_path: str
    scan_limit: int = 5000
    min_text_length: int = 50


class ZIMIndexRequest(BaseModel):
    """Request model for ZIM indexing"""
    zim_path: str
    source_id: str
    limit: int = 1000


@router.get("/zim/list")
async def list_zim_files():
    """
    List all ZIM files found in the backup folder.
    Used to populate the ZIM file selector in the UI.
    """
    from pathlib import Path as PathLib

    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        return {"zim_files": [], "error": "No backup folder configured"}

    # Debug info
    folder_path = PathLib(backup_folder)
    folder_exists = folder_path.exists()
    folder_is_dir = folder_path.is_dir() if folder_exists else False

    try:
        from offline_tools.zim_utils import find_zim_files
        zim_files = find_zim_files(backup_folder)
        return {
            "zim_files": zim_files,
            "backup_folder": backup_folder,
            "debug": {
                "folder_exists": folder_exists,
                "folder_is_dir": folder_is_dir,
                "folder_path_resolved": str(folder_path.resolve()) if folder_exists else None
            }
        }
    except Exception as e:
        import traceback
        return {"zim_files": [], "error": str(e), "traceback": traceback.format_exc()}


@router.post("/zim/inspect")
async def inspect_zim_file(request: ZIMInspectRequest):
    """
    Inspect a ZIM file and return detailed analysis.
    Runs as a background job since large ZIM files can take several minutes.

    Returns immediately with a job_id to poll for results.
    """
    from admin.job_manager import get_job_manager
    from pathlib import Path as PathLib

    # Validate ZIM file exists
    zim_path = PathLib(request.zim_path)
    if not zim_path.exists():
        raise HTTPException(404, f"ZIM file not found: {request.zim_path}")

    # Extract source_id from path for job tracking
    source_id = zim_path.stem

    def _run_zim_inspect(zim_path: str, scan_limit: int, min_text_length: int, progress_callback=None, cancel_checker=None):
        """Background job function for ZIM inspection"""
        from offline_tools.zim_utils import inspect_zim_file as do_inspect

        result = do_inspect(
            zim_path=zim_path,
            scan_limit=scan_limit,
            min_text_length=min_text_length,
            progress_callback=progress_callback
        )

        if result.error:
            return {"status": "error", "error": result.error}

        return {
            "status": "success",
            **result.to_dict()
        }

    manager = get_job_manager()

    try:
        job_id = manager.submit(
            "zim_inspect",
            source_id,
            _run_zim_inspect,
            str(zim_path),
            request.scan_limit,
            request.min_text_length
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    return {
        "status": "submitted",
        "job_id": job_id,
        "zim_path": str(zim_path),
        "scan_limit": request.scan_limit,
        "message": f"ZIM inspection started (scanning up to {request.scan_limit} articles)"
    }


@router.get("/zim/metadata/{source_id}")
async def get_zim_metadata(source_id: str):
    """
    Get just the metadata from a ZIM file (quick operation).
    Useful for auto-populating source configuration.
    """
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise HTTPException(400, "No backup folder configured")

    # Find the ZIM file for this source
    source_folder = Path(backup_folder) / source_id
    zim_path = None

    if source_folder.exists():
        zim_files = list(source_folder.glob("*.zim"))
        if zim_files:
            zim_path = zim_files[0]

    if not zim_path:
        # Try root level
        zim_path = Path(backup_folder) / f"{source_id}.zim"

    if not zim_path or not zim_path.exists():
        raise HTTPException(404, f"No ZIM file found for source: {source_id}")

    try:
        from offline_tools.zim_utils import get_zim_metadata as do_get_metadata

        metadata = do_get_metadata(str(zim_path))

        if "error" in metadata:
            raise HTTPException(400, metadata["error"])

        return {
            "status": "success",
            "source_id": source_id,
            "zim_path": str(zim_path),
            **metadata
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to get metadata: {e}")


@router.get("/zim/samples/{source_id}")
async def get_zim_samples(source_id: str, count: int = 10):
    """
    Get sample articles directly from ZIM file (no metadata required).

    Quick scan for URL pattern detection before metadata generation.
    Returns article titles and paths for base URL configuration.
    """
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise HTTPException(400, "No backup folder configured")

    # Find the ZIM file
    source_folder = Path(backup_folder) / source_id
    zim_path = None

    if source_folder.exists():
        zim_files = list(source_folder.glob("*.zim"))
        if zim_files:
            zim_path = zim_files[0]

    if not zim_path or not zim_path.exists():
        raise HTTPException(404, f"No ZIM file found for source: {source_id}")

    try:
        from zimply_core.zim_core import ZIMFile
        from offline_tools.indexer import extract_text_from_html
    except ImportError:
        raise HTTPException(400, "zimply-core not installed")

    samples = []
    try:
        zim = ZIMFile(str(zim_path), 'utf-8')
        article_count = zim.header_fields.get('articleCount', 0)

        # Sample articles spread across the ZIM (skip first few which are often index pages)
        step = max(1, article_count // (count * 2))
        start = min(100, article_count // 10)  # Start ~10% in to skip index pages

        checked = 0
        for i in range(start, min(article_count, start + step * count * 3), step):
            if len(samples) >= count:
                break
            checked += 1
            if checked > count * 5:  # Safety limit
                break

            try:
                article = zim.get_article_by_id(i)
                if article is None:
                    continue

                url = getattr(article, 'url', '') or ''
                title = getattr(article, 'title', '') or ''
                mimetype = str(getattr(article, 'mimetype', ''))

                # Only include HTML articles with real titles
                if 'text/html' not in mimetype:
                    continue
                if not title or len(title) < 3:
                    continue
                if title.lower() in ['index', 'home', 'main page', 'contents']:
                    continue

                # Extract article name (last path segment)
                article_name = url.split('/')[-1] if '/' in url else url

                samples.append({
                    "title": title[:100],
                    "zim_path": url,
                    "article_name": article_name,
                })
            except Exception:
                continue

        zim.close()

        return {
            "status": "success",
            "source_id": source_id,
            "article_count": article_count,
            "sample_count": len(samples),
            "samples": samples,
            "message": f"Found {len(samples)} sample articles from {article_count:,} total"
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to sample ZIM: {e}")


@router.post("/zim/index")
async def index_zim_file(request: ZIMIndexRequest):
    """
    Index a ZIM file as a new source.
    Runs as a background job with progress tracking.
    """
    from admin.job_manager import get_job_manager

    # Validate source_id format
    import re
    source_id = request.source_id.strip().lower()
    if not re.match(r'^[a-z0-9][a-z0-9_-]*$', source_id):
        raise HTTPException(400, "Invalid source ID format")

    # Check ZIM file exists
    zim_path = Path(request.zim_path)
    if not zim_path.exists():
        raise HTTPException(404, f"ZIM file not found: {request.zim_path}")

    def _run_zim_index(zim_path: str, source_id: str, limit: int, progress_callback=None, cancel_checker=None):
        """Background job function for ZIM indexing"""
        from offline_tools.indexer import ZIMIndexer

        config = get_local_config()
        backup_folder = config.get_backup_folder()

        indexer = ZIMIndexer(
            zim_path=zim_path,
            source_id=source_id,
            backup_folder=backup_folder
        )

        return indexer.index(limit=limit, progress_callback=progress_callback)

    manager = get_job_manager()

    try:
        job_id = manager.submit(
            "zim_index",
            source_id,
            _run_zim_index,
            str(zim_path),
            source_id,
            request.limit
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    return {
        "status": "submitted",
        "job_id": job_id,
        "source_id": source_id,
        "zim_path": str(zim_path),
        "limit": request.limit,
        "message": f"ZIM indexing started for {source_id}"
    }


# =============================================================================
# CLOUD SOURCE MANAGEMENT - List and install sources from R2
# =============================================================================

@router.get("/cloud-sources")
async def list_cloud_sources():
    """
    List available sources from R2 cloud storage.

    Returns sources that can be installed locally.
    """
    from offline_tools.source_manager import list_cloud_sources as _list_cloud_sources

    result = _list_cloud_sources()

    if result.get("error"):
        raise HTTPException(500, result["error"])

    return result


class InstallSourceRequest(BaseModel):
    source_id: str
    include_backup: bool = False
    sync_mode: str = "update"  # "update" (add/merge) or "replace" (delete old vectors first)


def _run_install_job(source_id: str, include_backup: bool, sync_mode: str = "update", progress_callback=None, cancel_checker=None):
    """Background job function for source installation"""
    from offline_tools.source_manager import install_source_from_cloud
    from offline_tools.vectordb.metadata import MetadataIndex
    from offline_tools.vectordb import get_vector_store

    deleted_count = 0

    # If replace mode, delete existing vectors first
    if sync_mode == "replace":
        if progress_callback:
            progress_callback(5, f"Deleting old vectors for {source_id}...")

        try:
            store = get_vector_store(mode="local")
            delete_result = store.delete_by_source(source_id)
            deleted_count = delete_result.get("deleted_count", 0)
            print(f"[install] Deleted {deleted_count} old vectors for {source_id}")
        except Exception as e:
            print(f"[install] Warning: Failed to delete old vectors: {e}")

    def progress_wrapper(stage, current, total):
        if progress_callback:
            # Job manager expects (percent, message) not (current, total)
            # Offset by 10% if we did a delete first
            base = 10 if sync_mode == "replace" else 0
            percent = base + int((current / max(total, 1)) * (100 - base))
            progress_callback(percent, stage)
            print(f"[install] {stage}: {current}/{total} ({percent}%)")

    result = install_source_from_cloud(
        source_id=source_id,
        include_backup=include_backup,
        progress_callback=progress_wrapper
    )

    # Add deletion stats to result
    if result.get("success"):
        result["sync_mode"] = sync_mode
        result["deleted_count"] = deleted_count

    # Verify metadata index can read the new source
    if result.get("success"):
        try:
            # Creating new MetadataIndex reads fresh _master.json from disk
            metadata_index = MetadataIndex()
            stats = metadata_index.get_stats()
            print(f"[install] Metadata index updated: {stats.get('total_documents', 0)} total documents")
        except Exception as e:
            print(f"[install] Warning: Failed to verify metadata index: {e}")

    return result


@router.post("/install-source")
async def install_source_from_cloud(request: InstallSourceRequest):
    """
    Download and install a source from R2 cloud storage.

    Downloads the source pack files and imports vectors into local ChromaDB.
    Runs as a background job.

    Args:
        sync_mode: "update" (add/merge) or "replace" (delete old vectors first)
    """
    from admin.job_manager import get_job_manager

    source_id = request.source_id.strip().lower()
    manager = get_job_manager()

    try:
        job_id = manager.submit(
            "install_source",
            source_id,
            _run_install_job,
            source_id,
            request.include_backup,
            request.sync_mode  # "update" or "replace"
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    mode_text = "Replacing" if request.sync_mode == "replace" else "Installing"
    return {
        "status": "submitted",
        "job_id": job_id,
        "source_id": source_id,
        "include_backup": request.include_backup,
        "sync_mode": request.sync_mode,
        "message": f"{mode_text} source '{source_id}' from cloud"
    }


# Alias for sources.html compatibility
@router.post("/download-pack")
async def download_pack(request: InstallSourceRequest):
    """
    Alias for /install-source - used by sources.html UI.
    Downloads source pack from R2 and installs to local ChromaDB.
    """
    return await install_source_from_cloud(request)


@router.post("/install-source-sync")
async def install_source_sync(request: InstallSourceRequest):
    """
    Synchronously install a source from R2 (for small sources).

    Use /install-source for large sources that need background processing.
    """
    from offline_tools.source_manager import install_source_from_cloud

    source_id = request.source_id.strip().lower()

    result = install_source_from_cloud(
        source_id=source_id,
        include_backup=request.include_backup
    )

    if not result["success"]:
        raise HTTPException(500, result.get("error", "Installation failed"))

    return {
        "status": "success",
        "source_id": source_id,
        "documents_added": result.get("install_result", {}).get("documents_added", 0),
        "files_downloaded": result.get("download_result", {}).get("files_downloaded", [])
    }


@router.get("/local-source-check/{source_id}")
async def check_local_source(source_id: str):
    """
    Check if a source already exists in local ChromaDB.

    Returns info about existing vectors for conflict detection before install.
    """
    from offline_tools.vectordb import get_vector_store

    source_id = source_id.strip().lower()

    try:
        store = get_vector_store(mode="local")
        vector_count = store.get_source_vector_count(source_id)

        if vector_count > 0:
            return {
                "exists": True,
                "source_id": source_id,
                "vector_count": vector_count,
                "message": f"Source '{source_id}' has {vector_count} vectors in local ChromaDB"
            }
        else:
            return {
                "exists": False,
                "source_id": source_id,
                "vector_count": 0,
                "message": f"Source '{source_id}' not found in local ChromaDB"
            }
    except Exception as e:
        return {"exists": False, "error": str(e)}


@router.get("/test-links/{source_id}")
async def test_source_links(source_id: str, test_base_url: Optional[str] = None):
    """
    Get sample links from a source to test URL construction.

    Returns sample articles with both local and online URLs so user can
    verify base_url is configured correctly before publishing to Pinecone.

    Args:
        source_id: The source ID to test
        test_base_url: Optional base URL to test with (simulates URL construction)
    """
    from admin.local_config import get_local_config
    import re

    source_id = source_id.strip().lower()
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        raise HTTPException(status_code=500, detail="Backup folder not configured")

    backup_path = Path(backup_folder)
    source_path = backup_path / source_id

    if not source_path.exists():
        raise HTTPException(status_code=404, detail=f"Source '{source_id}' not found")

    # Load manifest for base_url
    manifest_path = source_path / get_manifest_file()
    stored_base_url = ""
    source_type = "unknown"
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            stored_base_url = manifest.get("base_url", "")
            source_type = manifest.get("source_type", "unknown")
        except Exception:
            pass

    # Use test_base_url if provided, otherwise use stored
    base_url = test_base_url if test_base_url else stored_base_url

    # Load metadata for sample documents
    metadata_path = source_path / get_metadata_file()
    sample_links = []

    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Get up to 5 sample documents
            docs = metadata.get("documents", {})
            for i, (doc_id, doc) in enumerate(docs.items()):
                if i >= 5:
                    break

                local_url = doc.get("local_url", doc.get("url", ""))
                stored_url = doc.get("url", "")
                zim_url = doc.get("zim_url", "")  # Original ZIM path

                # Extract article path from local_url
                article_path = ""
                if local_url:
                    # Remove /zim/ or /backup/ prefixes
                    if local_url.startswith("/zim/"):
                        article_path = local_url[5:]  # Remove /zim/
                        # Remove source_id prefix if present
                        if "/" in article_path:
                            article_path = article_path.split("/", 1)[1]
                    elif local_url.startswith("/backup/"):
                        # /backup/source_id/path -> get just path
                        parts = local_url.split("/", 3)
                        if len(parts) > 3:
                            article_path = parts[3]
                    else:
                        article_path = local_url.lstrip("/")

                # Extract just the article name (last path segment)
                article_name = article_path.split("/")[-1] if "/" in article_path else article_path

                # Check if article_path contains full domain (www.example.com/path)
                # and extract just the path portion
                domain_pattern = r'^(www\.|[a-z0-9-]+\.(gov|com|org|net|edu|io|co|info|wiki))[/.]'
                clean_path = article_path
                if re.match(domain_pattern, article_path, re.IGNORECASE):
                    # Extract path after domain
                    path_match = re.match(r'^[^/]+/(.*)$', article_path)
                    if path_match:
                        clean_path = path_match.group(1)
                        article_name = clean_path.split("/")[-1] if "/" in clean_path else clean_path

                # If testing a new base_url, reconstruct the URL
                if test_base_url:
                    # Construct test URL using article name
                    if article_name:
                        test_url = test_base_url.rstrip("/") + "/" + article_name
                    else:
                        test_url = stored_url  # Fallback to stored
                    constructed_url = test_url
                else:
                    constructed_url = stored_url

                # Determine if URL looks valid
                is_online_url = constructed_url.startswith("http://") or constructed_url.startswith("https://")
                is_local_url = constructed_url.startswith("/zim/") or constructed_url.startswith("/backup/")

                sample_links.append({
                    "title": doc.get("title", "Unknown"),
                    "stored_url": stored_url,
                    "constructed_url": constructed_url,
                    "local_url": local_url,
                    "zim_path": article_path,  # Full path from ZIM
                    "article_name": article_name,  # Just the article identifier
                    "is_online_url": is_online_url,
                    "is_local_url": is_local_url,
                    "url_type": "online" if is_online_url else ("local" if is_local_url else "unknown")
                })
        except Exception as e:
            return {
                "source_id": source_id,
                "error": f"Failed to read metadata: {e}",
                "sample_links": []
            }

    return {
        "source_id": source_id,
        "source_type": source_type,
        "base_url": base_url,
        "stored_base_url": stored_base_url,
        "testing_custom_url": bool(test_base_url),
        "sample_count": len(sample_links),
        "sample_links": sample_links,
        "warning": None if all(s["is_online_url"] for s in sample_links) else
                   "Some URLs are local paths - online users won't be able to access them. Check base_url configuration."
    }


@router.delete("/local-source/{source_id}")
async def delete_local_source(source_id: str, delete_files: bool = False):
    """
    Delete a source from local ChromaDB.

    Args:
        source_id: Source to delete
        delete_files: If True, also delete the source folder from disk
    """
    from offline_tools.vectordb import get_vector_store
    from admin.local_config import get_local_config

    source_id = source_id.strip().lower()

    # Get local ChromaDB store
    store = get_vector_store(mode="local")

    # Get documents for this source
    try:
        # ChromaDB doesn't have a direct "delete by metadata" so we query first
        collection = store.collection
        results = collection.get(
            where={"source": source_id},
            include=[]
        )

        if results["ids"]:
            collection.delete(ids=results["ids"])
            deleted_count = len(results["ids"])
        else:
            deleted_count = 0

    except Exception as e:
        raise HTTPException(500, f"Failed to delete from ChromaDB: {e}")

    # Optionally delete files
    files_deleted = False
    if delete_files:
        config = get_local_config()
        backup_folder = config.get_backup_folder()
        if backup_folder:
            source_path = Path(backup_folder) / source_id
            if source_path.exists():
                import shutil
                shutil.rmtree(source_path)
                files_deleted = True

    return {
        "status": "success",
        "source_id": source_id,
        "documents_deleted": deleted_count,
        "files_deleted": files_deleted
    }
