"""
Source Localization API

Endpoints for creating and managing localized source variants.
Part of Phase 4 implementation for pre-translated sources.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import shutil

router = APIRouter(prefix="/api", tags=["Source Localization"])


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

class LocalizeSourceRequest(BaseModel):
    source_id: str
    target_language: str
    batch_size: int = 32
    resume: bool = True
    embedding_model: str = "all-mpnet-base-v2"
    force_overwrite: bool = False


class DeleteLocalizationRequest(BaseModel):
    source_id: str
    target_language: str


class DeleteSourceRequest(BaseModel):
    source_id: str
    confirm: bool = False  # Safety check


# =============================================================================
# LOCALIZATION REQUIREMENTS
# =============================================================================

@router.get("/localization-requirements/{source_id}/{target_lang}")
async def get_localization_requirements(source_id: str, target_lang: str):
    """
    Check if a source can be localized to the target language.

    Returns requirements status and any issues that would prevent localization.
    """
    try:
        from offline_tools.source_localizer import validate_localization_requirements

        result = validate_localization_requirements(source_id, target_lang)

        # Add estimated size
        config = get_local_config()
        backup_folder = config.get_backup_folder()

        if backup_folder:
            from pathlib import Path
            source_path = Path(backup_folder) / source_id
            if source_path.exists():
                # Estimate localized size (roughly doubles)
                size_bytes = sum(f.stat().st_size for f in source_path.rglob('*') if f.is_file())
                size_mb = size_bytes / (1024 * 1024)
                result["source_size_mb"] = round(size_mb, 1)
                result["estimated_localized_size_mb"] = round(size_mb * 1.5, 1)  # Conservative estimate

        return result

    except Exception as e:
        raise HTTPException(500, f"Failed to check requirements: {str(e)}")


# =============================================================================
# LIST LOCALIZATIONS
# =============================================================================

@router.get("/localizations/{source_id}")
async def get_localizations(source_id: str):
    """
    Get list of available localizations for a source.

    Returns language variants that have been created for the specified source.
    """
    try:
        from offline_tools.source_localizer import get_localized_sources

        localizations = get_localized_sources(source_id)

        return {
            "source_id": source_id,
            "localizations": localizations,
            "count": len(localizations)
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to get localizations: {str(e)}")


@router.get("/all-localized-sources")
async def get_all_localized_sources():
    """
    Get all localized sources across all parents.

    Scans backup folder for any sources marked as localizations.
    """
    try:
        from pathlib import Path
        import json

        config = get_local_config()
        backup_folder = config.get_backup_folder()

        if not backup_folder:
            return {"sources": [], "count": 0}

        backup_path = Path(backup_folder)
        localized_sources = []

        for item in backup_path.iterdir():
            if not item.is_dir():
                continue

            manifest_path = item / "_manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)

                    if manifest.get("is_localization"):
                        # Get folder size
                        size_bytes = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                        size_mb = size_bytes / (1024 * 1024)

                        localized_sources.append({
                            "source_id": item.name,
                            "parent_source": manifest.get("parent_source", ""),
                            "language": manifest.get("language", ""),
                            "language_name": manifest.get("name", item.name),
                            "document_count": manifest.get("total_docs", 0),
                            "created_at": manifest.get("localized_at", manifest.get("created_at", "")),
                            "size_mb": round(size_mb, 1),
                            "has_vectors": manifest.get("has_vectors", False)
                        })
                except Exception:
                    pass

        return {
            "sources": localized_sources,
            "count": len(localized_sources)
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to get localized sources: {str(e)}")


# =============================================================================
# START LOCALIZATION JOB
# =============================================================================

def _localize_source_job(source_id: str, target_lang: str,
                         progress_callback=None, cancel_checker=None, job_id=None,
                         batch_size: int = 32, resume: bool = True,
                         embedding_model: str = "all-mpnet-base-v2",
                         force_overwrite: bool = False):
    """
    Background job to localize a source.

    Translates all HTML files, regenerates metadata, creates embeddings,
    and populates language-specific ChromaDB.
    """
    from offline_tools.source_localizer import localize_source

    def update_progress(current, total, message):
        if progress_callback:
            # Convert to percentage format expected by job manager
            pct = int((current / total) * 100) if total > 0 else current
            progress_callback(pct, message)

    result = localize_source(
        source_id=source_id,
        target_lang=target_lang,
        progress_callback=update_progress,
        cancel_checker=cancel_checker,
        job_id=job_id,
        batch_size=batch_size,
        resume=resume,
        embedding_model=embedding_model,
        force_overwrite=force_overwrite
    )

    if not result.success:
        raise Exception(result.error or "Localization failed")

    return {
        "status": "success",
        "source_id": result.source_id,
        "target_lang": result.target_lang,
        "localized_source_id": result.localized_source_id,
        "documents_translated": result.documents_translated,
        "html_files_translated": result.html_files_translated,
        "embeddings_generated": result.embeddings_generated,
        "resumed": result.resumed,
        "warnings": result.warnings,
        "message": f"Successfully localized {source_id} to {target_lang}"
    }


@router.post("/localize-source")
async def localize_source_endpoint(request: LocalizeSourceRequest):
    """
    Start a source localization job.

    This is a long-running operation that runs in the background.
    Returns a job_id for tracking progress.
    """
    from offline_tools.source_localizer import validate_localization_requirements

    source_id = request.source_id
    target_lang = request.target_language

    # Validate requirements first (pass force_overwrite to skip "already exists" check)
    requirements = validate_localization_requirements(source_id, target_lang, force_overwrite=request.force_overwrite)

    if not requirements["can_localize"]:
        issues = ", ".join(requirements["issues"]) if requirements["issues"] else "Unknown issue"
        raise HTTPException(400, f"Cannot localize source: {issues}")

    try:
        config = get_local_config()
        backup_folder = config.get_backup_folder()

        if not backup_folder:
            raise HTTPException(400, "No backup folder configured. Set it in Settings first.")

        from admin.job_manager import get_job_manager
        manager = get_job_manager()

        # Submit as background job
        job_key = f"{source_id}_{target_lang}"

        try:
            job_id = manager.submit(
                "localize",
                job_key,
                _localize_source_job,
                source_id,
                target_lang,
                batch_size=request.batch_size,
                resume=request.resume,
                embedding_model=request.embedding_model,
                force_overwrite=request.force_overwrite
            )
        except ValueError as e:
            raise HTTPException(409, str(e))

        return {
            "status": "submitted",
            "job_id": job_id,
            "source_id": source_id,
            "target_language": target_lang,
            "localized_source_id": f"{source_id}_{target_lang}",
            "message": f"Localization job started for {source_id} -> {target_lang}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to start localization: {str(e)}")


# =============================================================================
# DELETE LOCALIZATION
# =============================================================================

@router.delete("/localizations/{source_id}/{target_lang}")
async def delete_localization(source_id: str, target_lang: str):
    """
    Delete a localized source variant.

    Removes all files for the localized source and cleans up ChromaDB entries.
    """
    try:
        from offline_tools.source_localizer import delete_localized_source

        result = delete_localized_source(source_id, target_lang)

        if not result["success"]:
            raise HTTPException(400, result.get("error", "Failed to delete localization"))

        return {
            "status": "success",
            "deleted_source": result.get("deleted_source"),
            "file_count": len(result.get("deleted_files", [])),
            "message": f"Deleted localization: {source_id}_{target_lang}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to delete localization: {str(e)}")


# =============================================================================
# DELETE SOURCE (General cleanup)
# =============================================================================

@router.delete("/sources/{source_id}")
async def delete_source(source_id: str, confirm: bool = False):
    """
    Delete a source entirely.

    This is for cleanup after localization - user can delete the original
    English source after successfully localizing to their target language.

    Requires confirm=true as a safety check.
    """
    if not confirm:
        raise HTTPException(400, "Deletion requires confirm=true parameter")

    try:
        from offline_tools.source_localizer import delete_source as do_delete_source

        result = do_delete_source(source_id)

        if not result["success"]:
            raise HTTPException(400, result.get("error", "Failed to delete source"))

        return {
            "status": "success",
            "deleted_source": result.get("deleted_source"),
            "file_count": result.get("file_count", 0),
            "message": f"Deleted source: {source_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to delete source: {str(e)}")


# =============================================================================
# LOCALIZATION STATUS / PROGRESS
# =============================================================================

@router.get("/localization-status/{source_id}/{target_lang}")
async def get_localization_status(source_id: str, target_lang: str):
    """
    Get status of a localization job or existing localization.

    Checks both active jobs and completed localizations.
    """
    try:
        localized_source_id = f"{source_id}_{target_lang}"

        # First check if there's an active job
        from admin.job_manager import get_job_manager
        manager = get_job_manager()

        job_key = f"{source_id}_{target_lang}"
        active_jobs = manager.get_active_jobs()

        for job in active_jobs:
            if job.get("source_id") == job_key and job.get("job_type") == "localize":
                return {
                    "status": "in_progress",
                    "job_id": job.get("job_id"),
                    "progress": job.get("progress", 0),
                    "message": job.get("message", "Localizing..."),
                    "started_at": job.get("started_at")
                }

        # Check if localization exists
        from pathlib import Path
        import json

        config = get_local_config()
        backup_folder = config.get_backup_folder()

        if backup_folder:
            localized_path = Path(backup_folder) / localized_source_id
            manifest_path = localized_path / "_manifest.json"

            if manifest_path.exists():
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)

                # Get folder size
                size_bytes = sum(f.stat().st_size for f in localized_path.rglob('*') if f.is_file())

                return {
                    "status": "completed",
                    "localized_source_id": localized_source_id,
                    "language": manifest.get("language"),
                    "document_count": manifest.get("total_docs", 0),
                    "has_vectors": manifest.get("has_vectors", False),
                    "created_at": manifest.get("localized_at", manifest.get("created_at")),
                    "size_mb": round(size_bytes / (1024 * 1024), 1)
                }

        # Check for checkpoint (resumable job)
        from offline_tools.source_localizer import _load_checkpoint
        checkpoint = _load_checkpoint(source_id, target_lang)

        if checkpoint:
            return {
                "status": "resumable",
                "phase": checkpoint.phase,
                "progress": checkpoint.progress,
                "last_saved": checkpoint.last_saved,
                "html_files_processed": checkpoint.html_files_processed,
                "message": f"Checkpoint available at {checkpoint.progress}% - can resume"
            }

        return {
            "status": "not_started",
            "message": f"No localization exists for {source_id} -> {target_lang}"
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to get status: {str(e)}")


# =============================================================================
# AVAILABLE LANGUAGES FOR LOCALIZATION
# =============================================================================

@router.get("/available-localization-languages")
async def get_available_localization_languages():
    """
    Get list of languages available for localization.

    Returns only languages where translation models are installed.
    """
    try:
        from offline_tools.language_registry import get_language_registry

        registry = get_language_registry()
        installed = registry.get_installed_packs()

        languages = []
        for lang_code in installed:
            pack_info = registry.get_pack_info(lang_code)
            if pack_info:
                languages.append({
                    "code": lang_code,
                    "name": pack_info.get("display_name", lang_code),
                    "native_name": pack_info.get("native_name", ""),
                    "installed": True
                })

        return {
            "languages": languages,
            "count": len(languages),
            "message": "Languages with installed translation models"
        }

    except Exception as e:
        raise HTTPException(500, f"Failed to get languages: {str(e)}")
