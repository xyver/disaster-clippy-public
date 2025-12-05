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


class CreateIndexRequest(BaseModel):
    source_id: str
    limit: int = 1000
    force_reindex: bool = False


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
        return {"status": "success", "source_id": request.source_id}
    except Exception as e:
        raise HTTPException(500, f"Failed to save source config: {e}")


@router.get("/auto-detect/{source_id}")
async def auto_detect(source_id: str):
    """Auto-detect license, description, tags, and URL from source content"""
    try:
        from offline_tools.source_manager import SourceManager

        result = {
            "source_id": source_id,
            "detected_license": None,
            "suggested_tags": [],
            "base_url": None
        }

        manager = SourceManager()

        # Detect license
        detected_license = manager.detect_license(source_id)
        if detected_license:
            result["detected_license"] = detected_license

        # Suggest tags
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
        if request.import_path:
            import_full_path = Path(backup_folder) / request.import_path
            if import_full_path.exists():
                import_info = {"path": str(import_full_path), "exists": True}
                if import_full_path != source_path:
                    import_info["note"] = f"Files exist at {request.import_path}. Use Scan Backup to import."

        return {
            "status": "success",
            "source_id": source_id,
            "source_type": request.source_type,
            "backup_type": backup_type,
            "path": str(source_path),
            "import_info": import_info,
            "message": f"Source '{source_id}' created successfully"
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
async def generate_metadata(request: SourceIdRequest):
    """Generate metadata from backup files"""
    try:
        from offline_tools.source_manager import SourceManager

        manager = SourceManager()
        result = manager.generate_metadata(request.source_id)

        if not result.get("success", True):
            raise HTTPException(400, result.get("error", "Metadata generation failed"))

        return {
            "status": "success",
            "source_id": request.source_id,
            "document_count": result.get("document_count", 0),
            "message": f"Generated metadata for {result.get('document_count', 0)} documents"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Metadata generation failed: {e}")


@router.post("/create-index")
async def create_index(request: CreateIndexRequest):
    """Create vector embeddings for a source (runs as background job)"""
    from admin.job_manager import get_job_manager

    def _run_create_index(source_id: str, limit: int, force: bool, progress_callback=None):
        from offline_tools.source_manager import SourceManager
        manager = SourceManager()
        return manager.create_index(
            source_id,
            limit=limit,
            force_reindex=force,
            progress_callback=progress_callback
        )

    manager = get_job_manager()

    try:
        job_id = manager.submit(
            "index",
            request.source_id,
            _run_create_index,
            request.source_id,
            request.limit,
            request.force_reindex
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    return {
        "status": "submitted",
        "job_id": job_id,
        "source_id": request.source_id,
        "limit": request.limit,
        "force_reindex": request.force_reindex,
        "message": f"Index creation started for {request.source_id}"
    }


# =============================================================================
# VALIDATION & CLEANUP
# =============================================================================

@router.post("/validate-source")
async def validate_source(request: ValidateSourceRequest):
    """Validate source completeness and readiness for distribution"""
    try:
        from offline_tools.source_manager import SourceManager

        manager = SourceManager()
        result = manager.validate_source(request.source_id)

        return {
            "status": "success",
            "source_id": request.source_id,
            "is_valid": result.is_valid,
            "production_ready": result.production_ready,
            "has_backup": result.has_backup,
            "has_metadata": result.has_metadata,
            "has_embeddings": result.has_embeddings,
            "has_license": result.has_license,
            "license_verified": result.license_verified,
            "schema_version": result.schema_version,
            "issues": result.issues,
            "warnings": result.warnings,
            "detected_license": result.detected_license,
            "suggested_tags": result.suggested_tags
        }

    except Exception as e:
        raise HTTPException(500, f"Validation failed: {e}")


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
        results = store.collection.get(where={"source_id": source_id})
        if results and results.get("ids"):
            store.collection.delete(ids=results["ids"])
            deleted_items.append(f"ChromaDB entries ({len(results['ids'])} docs)")
    except Exception:
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
