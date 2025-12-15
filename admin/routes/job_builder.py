"""
Job Builder API

Routes for the visual job chain builder.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/job-builder", tags=["Job Builder"])


# =============================================================================
# REQUEST MODELS
# =============================================================================

class ValidateChainRequest(BaseModel):
    job_types: List[str]
    source_id: Optional[str] = None  # For source-aware validation


class RunChainRequest(BaseModel):
    source_id: Optional[str] = None
    source_from_job: bool = False  # If True, first job determines source_id
    source_name: Optional[str] = None  # Custom display name for new source (manifest.name)
    phases: List[Dict[str, Any]]  # [{job_type, params}, ...]
    resumable: bool = True
    chain_name: Optional[str] = None  # Optional name for the combined job


class SaveTemplateRequest(BaseModel):
    name: str
    description: str = ""
    phases: List[Dict[str, Any]]


# =============================================================================
# API ENDPOINTS
# =============================================================================

@router.get("/schemas")
async def get_schemas():
    """Get all job schemas for the job builder UI"""
    from admin.job_schemas import get_all_schemas_dict
    return get_all_schemas_dict()


@router.get("/dimension-status")
async def get_dimension_status():
    """
    Get current ChromaDB dimension usage status.

    Returns which dimensions are currently being used by running jobs,
    useful for opportunistic scheduling and monitoring.

    Example response:
    {
        "384": {"busy": false, "job_id": null, "operation": null, "source_id": null},
        "768": {"busy": true, "job_id": "abc123", "operation": "index", "source_id": "wiki"},
        "1024": {"busy": false, "job_id": null, "operation": null, "source_id": null},
        "1536": {"busy": true, "job_id": "def456", "operation": "delete", "source_id": "bitcoin"}
    }
    """
    from admin.job_manager import get_job_manager
    manager = get_job_manager()
    status = manager.get_dimension_status()
    # Convert int keys to strings for JSON serialization
    return {str(k): v for k, v in status.items()}


@router.post("/validate")
async def validate_chain(request: ValidateChainRequest):
    """
    Validate a job chain for ordering issues, conflicts, and source state.

    If source_id is provided, also checks the source's current state
    (has metadata, has vectors, etc.) and returns relevant warnings.
    """
    from admin.job_schemas import validate_job_chain

    # Basic chain validation (ordering, conflicts)
    result = validate_job_chain(request.job_types)

    # If source_id provided, add source-aware validation
    if request.source_id:
        source_warnings = check_source_state(request.source_id, request.job_types)
        result["warnings"] = result.get("warnings", []) + source_warnings

    return result


def check_source_state(source_id: str, job_types: List[str]) -> List[str]:
    """
    Check source state and return relevant warnings for the job chain.

    Uses unified validation module to check source state and provide
    actionable warnings about what's needed for submission/publishing.

    Returns warnings like:
    - "Source already has online index (1536-dim) - will regenerate"
    - "Source has no metadata - run Metadata first"
    - "Submission requires: verified license, verified links, English language"
    """
    from pathlib import Path
    from admin.local_config import get_local_config
    from offline_tools.schemas import get_metadata_file, get_index_file

    warnings = []
    config = get_local_config()
    backup_folder = config.get_backup_folder()

    if not backup_folder:
        return warnings

    source_path = Path(backup_folder) / source_id
    if not source_path.exists():
        # New source, no state to check
        return warnings

    # Use unified validation for comprehensive checks
    try:
        from offline_tools.validation import validate_light
        validation = validate_light(str(source_path), source_id, use_cache=True)

        # Check for existing metadata
        has_metadata = validation.has_metadata

        # Check for existing vectors
        vector_counts = {}
        if validation.vector_count_1536 > 0:
            vector_counts[1536] = validation.vector_count_1536
        if validation.vector_count_768 > 0:
            vector_counts[768] = validation.vector_count_768

        # Add validation status warnings
        if not validation.can_submit:
            missing_for_submit = []
            if not validation.license_in_allowlist:
                missing_for_submit.append("license in allowlist")
            if not validation.license_verified:
                missing_for_submit.append("verified license")
            if not validation.links_verified_offline:
                missing_for_submit.append("verified offline links")
            if not validation.links_verified_online:
                missing_for_submit.append("verified online links")
            if not validation.language_is_english:
                missing_for_submit.append("English language")
            if not (validation.has_vectors_1536 or validation.has_vectors_768):
                missing_for_submit.append("at least one index (768 or 1536)")

            if missing_for_submit:
                warnings.append(f"Submission requires: {', '.join(missing_for_submit)}")

    except Exception as e:
        # Fallback to simple file checks
        print(f"Validation error in job builder: {e}")
        metadata_path = source_path / get_metadata_file()
        has_metadata = metadata_path.exists()
        vector_counts = {}

        # Try to get vector counts from ChromaDB
        try:
            from offline_tools.vectordb import get_vector_store
            try:
                store_768 = get_vector_store(mode="local", dimension=768, read_only=True)
                count_768 = store_768.get_source_vector_count(source_id)
                if count_768 > 0:
                    vector_counts[768] = count_768
            except Exception:
                pass
            try:
                store_1536 = get_vector_store(mode="local", dimension=1536, read_only=True)
                count_1536 = store_1536.get_source_vector_count(source_id)
                if count_1536 > 0:
                    vector_counts[1536] = count_1536
            except Exception:
                pass
        except Exception:
            pass

    # Generate warnings based on job chain and source state
    for job_type in job_types:
        if job_type == "metadata":
            if has_metadata:
                warnings.append(f"Source '{source_id}' already has metadata - will regenerate")

        elif job_type == "index_online":
            if not has_metadata and "metadata" not in job_types:
                warnings.append(f"Source '{source_id}' has no metadata - add Metadata job first")
            if 1536 in vector_counts:
                warnings.append(f"Source already has {vector_counts[1536]:,} online vectors (1536-dim) - will regenerate")

        elif job_type == "index_offline":
            if not has_metadata and "metadata" not in job_types:
                warnings.append(f"Source '{source_id}' has no metadata - add Metadata job first")
            if 768 in vector_counts:
                warnings.append(f"Source already has {vector_counts[768]:,} offline vectors (768-dim) - will regenerate")

        elif job_type == "pinecone_sync":
            if 1536 not in vector_counts and "index_online" not in job_types:
                warnings.append(f"Source has no online index - add Index (Online) first")

        elif job_type == "translate_source":
            if not has_metadata and "metadata" not in job_types:
                warnings.append(f"Source '{source_id}' has no metadata - add Metadata job first")

        elif job_type == "detect_language":
            # New job type for language detection
            if not has_metadata and "metadata" not in job_types:
                warnings.append(f"Source '{source_id}' has no metadata - add Metadata job first")

        elif job_type == "detect_base_url":
            # Check if pages folder exists
            pages_path = source_path / "pages"
            if not pages_path.exists():
                warnings.append(f"Source '{source_id}' has no pages folder - extract content first (ZIM or scrape)")
            # Check if base_url already set (check both manifest file locations)
            manifest_path = source_path / "_manifest.json"
            if not manifest_path.exists():
                manifest_path = source_path / "manifest.json"
            if manifest_path.exists():
                try:
                    import json
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    if manifest.get("base_url"):
                        warnings.append(f"Source already has base_url: {manifest.get('base_url')} - will be overwritten if auto-apply is on")
                except Exception:
                    pass

    return warnings


# Jobs that create/establish a source (can be first when source_from_job=True)
CREATION_JOBS = {'zim_import', 'zim_extract', 'install_source', 'scrape', 'scrape_sitemap'}


@router.post("/run")
async def run_chain(request: RunChainRequest):
    """
    Run a custom job chain as a combined job.

    Creates JobPhase objects from the request and runs them using run_combined_job.

    If source_from_job=True, the first job must be a creation job (ZIM inspect,
    install, scrape) that establishes the source_id for subsequent jobs.
    """
    from admin.job_manager import get_job_manager, JobPhase, run_combined_job
    from admin.job_schemas import get_job_schema

    if not request.phases:
        raise HTTPException(400, "No phases provided")

    # Determine the source_id based on mode
    effective_source_id = request.source_id

    if request.source_from_job:
        # First job must be a creation job that determines the source
        first_job = request.phases[0].get("job_type")
        if first_job not in CREATION_JOBS:
            raise HTTPException(400, f"When using 'New Source', first job must be a creation job. Got: {first_job}")

        # Extract source_id from the first job's params
        first_params = request.phases[0].get("params", {})

        if first_job == "install_source":
            effective_source_id = first_params.get("cloud_source_id", "").strip().lower()
            if not effective_source_id:
                raise HTTPException(400, "install_source requires selecting a cloud source")

        elif first_job in ("zim_import", "zim_extract"):
            # Source ID comes from user-provided name, or ZIM filename
            zim_path = first_params.get("zim_path", "")
            if not zim_path:
                raise HTTPException(400, f"{first_job} requires selecting a ZIM file")

            from pathlib import Path
            from admin.local_config import get_local_config
            zim_file = Path(zim_path)

            # Helper to sanitize source_id
            def sanitize_source_id(name):
                safe = re.sub(r'[<>:"/\\|?*]', '_', name)
                safe = re.sub(r'\s+', '-', safe)
                safe = safe.strip('._-')
                return safe.lower()[:100]

            # Priority: user-provided name > ZIM filename (always)
            # We no longer try to derive from folder names - that was causing issues
            if request.source_name:
                effective_source_id = sanitize_source_id(request.source_name)
            else:
                # Always use ZIM filename as source_id when no name provided
                effective_source_id = sanitize_source_id(zim_file.stem)
                logger.info(f"[run_chain] Derived source_id from ZIM filename: {effective_source_id}")

        elif first_job in ("scrape", "scrape_sitemap"):
            # Source ID should be provided in params for scrape jobs
            effective_source_id = first_params.get("source_id", "").strip().lower()
            if not effective_source_id:
                raise HTTPException(400, "Scrape jobs require a source_id in params")

        if not effective_source_id:
            raise HTTPException(400, f"Could not determine source_id from {first_job} job")

        # Inject source_name into first job params for creation jobs
        if request.source_name and first_job in ("zim_import", "zim_extract"):
            request.phases[0]["params"]["source_name"] = request.source_name

    elif not request.source_id:
        # No source and not from job - check if any job needs a source
        for phase in request.phases:
            job_type = phase.get("job_type")
            schema = get_job_schema(job_type)
            # install_source uses cloud_source_id from params, not the main source_id
            if job_type == "install_source":
                if not phase.get("params", {}).get("cloud_source_id"):
                    raise HTTPException(400, "install_source requires selecting a cloud source")
                continue
            # zim_import uses zim_path from params, not the main source_id
            if job_type == "zim_import":
                if not phase.get("params", {}).get("zim_path"):
                    raise HTTPException(400, f"{job_type} requires selecting a ZIM file")
                continue
            if schema and schema.requires_source:
                raise HTTPException(400, f"Source ID required for job: {job_type}")

    # Build JobPhase objects
    phases = []
    for i, phase_data in enumerate(request.phases):
        job_type = phase_data.get("job_type")
        params = phase_data.get("params", {})

        schema = get_job_schema(job_type)
        if not schema:
            raise HTTPException(400, f"Unknown job type: {job_type}")

        # Get the actual job function based on job type
        func = get_job_function(job_type, effective_source_id, params)
        if not func:
            raise HTTPException(400, f"No function found for job type: {job_type}")

        phases.append(JobPhase(
            name=schema.label,
            func=func,
            weight=schema.weight,
            args=(),
            kwargs={}
        ))

    # Determine combined job type name
    job_type_name = request.chain_name or "custom_chain"

    # Store phase definitions for checkpoint resume
    phase_defs = [{"job_type": p.get("job_type"), "params": p.get("params", {})} for p in request.phases]

    # Capture source_name for use in wrapper
    custom_source_name = request.source_name

    # Create wrapper function that runs the combined job
    def _run_custom_chain(progress_callback=None, cancel_checker=None):
        print(f"[job_builder] Starting custom chain: {job_type_name} for {effective_source_id}")
        print(f"[job_builder] Phases: {[p.name for p in phases]}")
        try:
            result = run_combined_job(
                phases,
                progress_callback=progress_callback,
                cancel_checker=cancel_checker,
                source_id=effective_source_id,
                job_type=job_type_name,
                resume=request.resumable,
                phase_definitions=phase_defs
            )
            print(f"[job_builder] Chain completed: {result.get('success', 'unknown')}")
            if not result.get('success'):
                print(f"[job_builder] Chain failed: {result.get('error', result.get('message', 'unknown'))}")

            # Apply custom source name to manifest if provided
            if custom_source_name and result.get('success'):
                try:
                    _apply_source_name(effective_source_id, custom_source_name)
                    print(f"[job_builder] Applied custom name: {custom_source_name}")
                except Exception as e:
                    print(f"[job_builder] Warning: Could not apply source name: {e}")

            return result
        except Exception as e:
            print(f"[job_builder] Chain error: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    # Submit to job manager
    manager = get_job_manager()

    print(f"[job_builder] Submitting job: {job_type_name} for {effective_source_id} with {len(phases)} phases")

    try:
        job_id = manager.submit(
            job_type_name,
            effective_source_id,
            _run_custom_chain
        )
        print(f"[job_builder] Job submitted successfully: {job_id}")
    except ValueError as e:
        print(f"[job_builder] Job submission failed: {e}")
        raise HTTPException(409, str(e))

    return {
        "status": "submitted",
        "job_id": job_id,
        "source_id": effective_source_id,
        "source_from_job": request.source_from_job,
        "phases_count": len(phases),
        "resumable": request.resumable,
        "message": f"Custom chain started with {len(phases)} phases"
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _apply_source_name(source_id: str, name: str):
    """
    Update the manifest.json with a custom source name.
    Called after job chain completes to apply user-provided name.
    """
    from pathlib import Path
    from admin.local_config import get_local_config
    from offline_tools.schemas import get_manifest_file
    import json

    config = get_local_config()
    backup_folder = config.get_backup_folder()
    if not backup_folder:
        return

    manifest_path = Path(backup_folder) / source_id / get_manifest_file()
    if not manifest_path.exists():
        return

    # Read, update, write
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["name"] = name

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def get_job_function(job_type: str, source_id: str, params: Dict[str, Any]):
    """
    Get the actual function to run for a job type.

    Returns a callable that accepts (progress_callback, cancel_checker).
    """

    if job_type == "metadata":
        def run_metadata(progress_callback=None, cancel_checker=None):
            from offline_tools.source_manager import SourceManager
            manager = SourceManager()

            if progress_callback:
                progress_callback(0, 100, "Generating metadata...")

            # Language filter: subtract articles not in this language
            lang_filter = params.get("language_filter") or None

            result = manager.generate_metadata(
                source_id,
                language_filter=lang_filter,
                progress_callback=lambda curr, total, msg: progress_callback(
                    int((curr / max(total, 1)) * 100), 100, msg
                ) if progress_callback else None
            )

            if progress_callback:
                progress_callback(100, 100, "Metadata complete")

            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "result": result}

        return run_metadata

    elif job_type == "index_online":
        def run_index_online(progress_callback=None, cancel_checker=None, job_id=None, **kwargs):
            from offline_tools.source_manager import SourceManager
            from admin.job_manager import get_job_manager

            force = params.get("force_reindex", False)
            resume = params.get("resume", False)

            # Track dimension usage for opportunistic scheduling
            job_manager = get_job_manager()
            if job_id:
                job_manager.update_job_dimension(job_id, 1536, "index")

            try:
                manager = SourceManager()
                result = manager.create_index(
                    source_id,
                    limit=None,  # No limit - index all documents
                    skip_existing=not force,  # force=True means skip_existing=False
                    dimension=1536,
                    resume=resume,
                    progress_callback=progress_callback,
                    cancel_checker=cancel_checker
                )
                return result
            finally:
                # Clear dimension tracking when done
                if job_id:
                    job_manager.update_job_dimension(job_id, None, None)

        return run_index_online

    elif job_type == "index_offline":
        def run_index_offline(progress_callback=None, cancel_checker=None, job_id=None, **kwargs):
            from offline_tools.source_manager import SourceManager
            from admin.job_manager import get_job_manager

            force = params.get("force_reindex", False)
            resume = params.get("resume", False)
            dimension = int(params.get("dimension", 768))

            # Track dimension usage for opportunistic scheduling
            job_manager = get_job_manager()
            if job_id:
                job_manager.update_job_dimension(job_id, dimension, "index")

            try:
                manager = SourceManager()
                result = manager.create_index(
                    source_id,
                    limit=None,  # No limit - index all documents
                    skip_existing=not force,  # force=True means skip_existing=False
                    dimension=dimension,
                    resume=resume,
                    progress_callback=progress_callback,
                    cancel_checker=cancel_checker
                )
                return result
            finally:
                # Clear dimension tracking when done
                if job_id:
                    job_manager.update_job_dimension(job_id, None, None)

        return run_index_offline

    elif job_type == "clear_vectors":
        def run_clear_vectors(progress_callback=None, cancel_checker=None, job_id=None, **kwargs):
            from offline_tools.vectordb import get_vector_store
            from admin.job_manager import get_job_manager

            dimension = int(params.get("dimension", 768))

            # Track dimension usage for opportunistic scheduling (delete has priority)
            job_manager = get_job_manager()
            if job_id:
                job_manager.update_job_dimension(job_id, dimension, "delete")

            try:
                if progress_callback:
                    progress_callback(0, 100, f"Clearing {dimension}-dim vectors...")

                store = get_vector_store(mode="local", dimension=dimension, read_only=True)
                result = store.delete_by_source(source_id)
                deleted = result.get("deleted_count", 0) if isinstance(result, dict) else 0

                if progress_callback:
                    progress_callback(100, 100, f"Cleared {deleted} vectors")

                return {"success": True, "deleted_count": deleted}
            finally:
                # Clear dimension tracking when done
                if job_id:
                    job_manager.update_job_dimension(job_id, None, None)

        return run_clear_vectors

    elif job_type == "suggest_tags":
        def run_suggest_tags(progress_callback=None, cancel_checker=None):
            from offline_tools.source_manager import SourceManager

            manager = SourceManager()

            if progress_callback:
                progress_callback(0, 100, "Analyzing content...")

            suggested = manager.suggest_tags(source_id)

            if progress_callback:
                progress_callback(100, 100, f"Found {len(suggested)} suggested tags")

            return {"success": True, "tags": suggested, "count": len(suggested)}

        return run_suggest_tags

    elif job_type == "pinecone_sync":
        def run_pinecone_sync(progress_callback=None, cancel_checker=None):
            from admin.cloud_upload import push_to_pinecone

            dry_run = params.get("dry_run", False)

            result = push_to_pinecone(
                source_id,
                dry_run=dry_run,
                progress_callback=progress_callback,
                cancel_checker=cancel_checker
            )
            return result

        return run_pinecone_sync

    elif job_type == "visualisation":
        def run_visualisation(progress_callback=None, cancel_checker=None):
            from admin.cloud_upload import generate_knowledge_map

            result = generate_knowledge_map(
                source_ids=[source_id] if source_id else None,
                progress_callback=progress_callback,
                cancel_checker=cancel_checker
            )
            return result

        return run_visualisation

    elif job_type == "install_source":
        def run_install_source(progress_callback=None, cancel_checker=None):
            from admin.routes.source_tools import _run_install_job

            # Use cloud_source_id from params (not the main source_id)
            cloud_source_id = params.get("cloud_source_id", "").strip().lower()
            if not cloud_source_id:
                return {"success": False, "error": "No cloud source selected"}

            include_backup = params.get("include_backup", False)
            sync_mode = params.get("sync_mode", "update")

            result = _run_install_job(
                cloud_source_id,
                include_backup,
                sync_mode,
                progress_callback=progress_callback,
                cancel_checker=cancel_checker
            )
            return result

        return run_install_source

    elif job_type == "translate_source":
        def run_translate_source(progress_callback=None, cancel_checker=None):
            from admin.routes.source_tools import _run_translate_source_job

            language = params.get("language", "")
            if not language:
                return {"success": False, "error": "No target language selected"}

            batch_size = int(params.get("batch_size", 10))
            skip_cached = params.get("skip_cached", True)

            result = _run_translate_source_job(
                source_id,
                language,
                batch_size=batch_size,
                skip_cached=skip_cached,
                progress_callback=progress_callback,
                cancel_checker=cancel_checker
            )
            return result

        return run_translate_source

    elif job_type == "zim_import":
        def run_zim_import(progress_callback=None, cancel_checker=None):
            """
            Combined ZIM Import job that:
            1. Opens and analyzes ZIM file
            2. Creates a dedicated source folder
            3. Moves ZIM file into that folder
            4. Extracts indexable HTML pages to pages/ folder
            5. Creates manifest with all metadata
            """
            from pathlib import Path
            from admin.local_config import get_local_config
            import json
            import shutil
            import re

            zim_path = params.get("zim_path", "")
            min_text_length = int(params.get("min_text_length", 100))
            custom_source_name = params.get("source_name", "")  # User-provided source name

            if not zim_path:
                return {"success": False, "error": "No ZIM file selected"}

            zim_file = Path(zim_path)
            if not zim_file.exists():
                return {"success": False, "error": f"ZIM file not found: {zim_path}"}

            # Get backup folder
            config = get_local_config()
            backup_folder = config.get_backup_folder()
            if not backup_folder:
                return {"success": False, "error": "No backup folder configured"}

            backup_path = Path(backup_folder).resolve()
            zim_parent = zim_file.parent.resolve()

            # Determine source_id:
            # 1. If user provided a custom name, use that (sanitized)
            # 2. Otherwise, always use the ZIM filename (sanitized)
            # Note: We no longer try to derive from folder names - that was causing issues
            def sanitize_source_id(name):
                """Sanitize a name for use as source_id (filesystem-safe)"""
                # Remove/replace problematic characters
                safe = re.sub(r'[<>:"/\\|?*]', '_', name)
                safe = re.sub(r'\s+', '-', safe)  # spaces to hyphens
                safe = safe.strip('._-')  # remove leading/trailing dots, underscores, hyphens
                return safe.lower()[:100]  # lowercase, max 100 chars

            if custom_source_name:
                # User provided a name - use it as source_id
                derived_source_id = sanitize_source_id(custom_source_name)
                logger.info(f"[zim_import] Using user-provided source name: {derived_source_id}")
            else:
                # Always use ZIM filename as source_id
                derived_source_id = sanitize_source_id(zim_file.stem)
                logger.info(f"[zim_import] Derived source_id from ZIM filename: {derived_source_id}")

            source_path = backup_path / derived_source_id
            pages_path = source_path / "pages"

            # Check if ZIM is already in the target source folder
            # Use os.path.normcase for case-insensitive comparison on Windows
            import os
            zim_already_in_place = os.path.normcase(str(zim_parent)) == os.path.normcase(str(source_path))

            if progress_callback:
                progress_callback(0, 100, "Opening ZIM file...")

            # Import zimply-core
            try:
                from zimply_core.zim_core import ZIMFile
            except ImportError:
                return {"success": False, "error": "zimply-core not installed. Run: pip install zimply-core"}

            # Import text extractor
            try:
                from offline_tools.indexer import extract_text_from_html
            except ImportError:
                import re
                def extract_text_from_html(html):
                    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
                    text = re.sub(r'<[^>]+>', ' ', text)
                    text = re.sub(r'\s+', ' ', text)
                    return text.strip()

            try:
                zim = ZIMFile(str(zim_file), 'utf-8')
            except Exception as e:
                return {"success": False, "error": f"Failed to open ZIM file: {e}"}

            # Extract header metadata
            header = zim.header_fields
            article_count = header.get('articleCount', 0)

            if progress_callback:
                progress_callback(5, 100, f"ZIM has {article_count} entries, scanning for HTML content...")

            # Create source folder structure
            source_path.mkdir(parents=True, exist_ok=True)
            pages_path.mkdir(parents=True, exist_ok=True)

            # Build manifest from ZIM metadata
            # Use custom name if provided, otherwise fall back to ZIM metadata
            display_name = custom_source_name or header.get('Title') or header.get('Name') or derived_source_id
            manifest = {
                "source_id": derived_source_id,
                "name": display_name,
                "description": header.get('Description') or "",
                "base_url": "",  # Will be set by detect_base_url or manually
                "license": header.get('License') or "",
                "language": header.get('Language') or "",
                "zim_metadata": {
                    "creator": header.get('Creator') or "",
                    "publisher": header.get('Publisher') or "",
                    "source_url": header.get('Source') or "",
                    "date": header.get('Date') or "",
                    "original_article_count": article_count,
                },
                "created_from": "zim_import",
                "zim_file": zim_file.name,
                "zim_path": str(zim_file),  # Full path for samples endpoint
            }

            # Smart license detection for common ZIM sources
            if not manifest["license"]:
                creator_lower = (manifest["zim_metadata"]["creator"] or '').lower()
                publisher_lower = (manifest["zim_metadata"]["publisher"] or '').lower()
                title_lower = (manifest["name"] or '').lower()

                if any(x in creator_lower or x in publisher_lower or x in title_lower
                       for x in ['wikipedia', 'wikimedia', 'wikibooks', 'wikivoyage', 'wiktionary']):
                    manifest["license"] = "CC BY-SA 3.0"
                elif 'stackexchange' in creator_lower or 'stack overflow' in creator_lower:
                    manifest["license"] = "CC BY-SA 4.0"

            # Scan and extract HTML content
            extracted_count = 0
            skipped_count = 0
            total_text_length = 0

            for i in range(article_count):
                if cancel_checker and cancel_checker():
                    # Clean up on cancel
                    try:
                        zim.close()
                    except Exception:
                        pass
                    return {"success": False, "error": "Job cancelled", "cancelled": True}

                if progress_callback and i % 100 == 0:
                    pct = 5 + int((i / max(article_count, 1)) * 90)
                    progress_callback(pct, 100, f"Processing {i}/{article_count} entries ({extracted_count} pages extracted)...")

                try:
                    article = zim.get_article_by_id(i)
                    if article is None:
                        continue

                    url = getattr(article, 'url', '') or ''
                    mimetype = str(getattr(article, 'mimetype', ''))

                    # Only process HTML content
                    if 'text/html' not in mimetype:
                        continue

                    content = article.data
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='ignore')

                    # Check text length
                    text = extract_text_from_html(content)
                    text_len = len(text)

                    if text_len < min_text_length:
                        skipped_count += 1
                        continue

                    # Extract to pages folder
                    # Clean up URL for filesystem
                    safe_url = url.replace('/', '_').replace('\\', '_').replace(':', '_')
                    if not safe_url.endswith('.html'):
                        safe_url += '.html'

                    page_file = pages_path / safe_url
                    with open(page_file, 'w', encoding='utf-8') as f:
                        f.write(content)

                    extracted_count += 1
                    total_text_length += text_len

                except Exception:
                    continue

            # Close ZIM file
            try:
                zim.close()
            except Exception:
                pass

            # Move ZIM file into source folder if not already there
            zim_moved = False
            final_zim_path = zim_file  # Default: keep original path

            if not zim_already_in_place:
                if progress_callback:
                    progress_callback(96, 100, "Moving ZIM file into source folder...")

                try:
                    # Keep original filename (don't rename to source.zim - preserves version info)
                    new_zim_path = source_path / zim_file.name

                    # Check if destination already exists
                    if new_zim_path.exists() and new_zim_path != zim_file:
                        # ZIM with same name already in folder - skip move
                        final_zim_path = new_zim_path
                    else:
                        # Move the file
                        shutil.move(str(zim_file), str(new_zim_path))
                        final_zim_path = new_zim_path
                        zim_moved = True
                except Exception as e:
                    # Move failed - continue with original path but warn
                    logger.warning(f"Could not move ZIM file: {e}")
            else:
                final_zim_path = zim_file

            # Update manifest with extraction stats and final ZIM path
            manifest["extraction_stats"] = {
                "extracted_pages": extracted_count,
                "skipped_pages": skipped_count,
                "min_text_length": min_text_length,
                "avg_text_length": round(total_text_length / max(extracted_count, 1)),
            }
            manifest["zim_file"] = final_zim_path.name
            manifest["zim_path"] = str(final_zim_path)

            # Save manifest (use _manifest.json standard)
            # IMPORTANT: Preserve user-edited fields from existing manifest
            manifest_path = source_path / "_manifest.json"
            existing_manifest = {}
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r', encoding='utf-8') as f:
                        existing_manifest = json.load(f)
                except Exception:
                    pass

            # Fields to preserve from existing manifest (user edits take precedence)
            preserved_fields = ["name", "description", "base_url", "license",
                               "license_verified", "attribution", "tags", "language",
                               "version", "created_at"]
            for field in preserved_fields:
                if field in existing_manifest and existing_manifest[field]:
                    manifest[field] = existing_manifest[field]

            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            if progress_callback:
                progress_callback(100, 100, f"Import complete: {extracted_count} pages extracted")

            result_msg = f"Imported {extracted_count} pages from ZIM ({skipped_count} skipped as too short)"
            if zim_moved:
                result_msg += f". ZIM moved to {source_path.name}/"

            return {
                "success": True,
                "source_id": derived_source_id,
                "source_folder": str(source_path),
                "extracted_pages": extracted_count,
                "skipped_pages": skipped_count,
                "total_zim_entries": article_count,
                "manifest_created": True,
                "zim_moved": zim_moved,
                "zim_path": str(final_zim_path),
                "message": result_msg
            }

        return run_zim_import

    elif job_type == "detect_base_url":
        def run_detect_base_url(progress_callback=None, cancel_checker=None):
            """
            Sample URLs from source and auto-detect base URL pattern.

            This job:
            1. Scans the pages/ folder for HTML files
            2. Samples random articles
            3. Analyzes file paths to detect URL patterns
            4. Suggests a base_url for the manifest
            5. Optionally auto-applies if confident
            """
            from pathlib import Path
            from admin.local_config import get_local_config
            import random
            import json
            import re

            sample_count = int(params.get("sample_count", 10))
            force_override = params.get("force_override", False)

            if progress_callback:
                progress_callback(0, 100, "Scanning source folder...")

            config = get_local_config()
            backup_folder = config.get_backup_folder()

            if not backup_folder:
                return {"success": False, "error": "No backup folder configured"}

            source_path = Path(backup_folder) / source_id
            pages_path = source_path / "pages"

            # Check manifest for existing hints first (can work without pages)
            # Try _manifest.json first (standard), then manifest.json (legacy)
            manifest_path = source_path / "_manifest.json"
            if not manifest_path.exists():
                manifest_path = source_path / "manifest.json"
            existing_manifest = {}
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        existing_manifest = json.load(f)
                except Exception:
                    pass
            else:
                # If neither exists, use the standard path for saving
                manifest_path = source_path / "_manifest.json"

            # Check for existing base_url - don't override if user has set a meaningful value
            existing_base_url = existing_manifest.get("base_url", "")
            # Consider these as "not set" - placeholder values that should be overwritten
            placeholder_urls = ["", "https://example.org/", "https://example.com/"]
            has_user_set_url = existing_base_url and existing_base_url not in placeholder_urls
            # Allow override if force_override is set
            if force_override:
                has_user_set_url = False

            # Try ZIM metadata detection first (works without pages)
            zim_metadata = existing_manifest.get("zim_metadata", {})
            detected_base_url = None
            confidence = "low"
            detection_method = "unknown"

            if zim_metadata:
                if progress_callback:
                    progress_callback(20, 100, "Checking ZIM metadata...")

                # ZIM files often have the original URL in metadata
                zim_name = zim_metadata.get("Name", "")
                if "wikipedia" in zim_name.lower():
                    # Wikipedia ZIMs typically use wiki.domain.org/wiki/
                    lang_match = re.search(r'wikipedia_(\w+)', zim_name.lower())
                    lang = lang_match.group(1) if lang_match else "en"
                    detected_base_url = f"https://{lang}.wikipedia.org/wiki/"
                    confidence = "high"
                    detection_method = "wikipedia_zim"
                elif "wikimedia" in zim_name.lower():
                    detected_base_url = "https://commons.wikimedia.org/wiki/"
                    confidence = "medium"
                    detection_method = "wikimedia_zim"
                elif "wikibooks" in zim_name.lower():
                    lang_match = re.search(r'wikibooks_(\w+)', zim_name.lower())
                    lang = lang_match.group(1) if lang_match else "en"
                    detected_base_url = f"https://{lang}.wikibooks.org/wiki/"
                    confidence = "high"
                    detection_method = "wikibooks_zim"
                elif "wikivoyage" in zim_name.lower():
                    lang_match = re.search(r'wikivoyage_(\w+)', zim_name.lower())
                    lang = lang_match.group(1) if lang_match else "en"
                    detected_base_url = f"https://{lang}.wikivoyage.org/wiki/"
                    confidence = "high"
                    detection_method = "wikivoyage_zim"
                elif "wiktionary" in zim_name.lower():
                    lang_match = re.search(r'wiktionary_(\w+)', zim_name.lower())
                    lang = lang_match.group(1) if lang_match else "en"
                    detected_base_url = f"https://{lang}.wiktionary.org/wiki/"
                    confidence = "high"
                    detection_method = "wiktionary_zim"

            # If detected from ZIM, we can return early (no pages needed)
            if detected_base_url and confidence in ("high", "medium"):
                result = {
                    "success": True,
                    "source_id": source_id,
                    "total_files": 0,
                    "samples_analyzed": 0,
                    "sample_paths": [],
                    "detected_base_url": detected_base_url,
                    "existing_base_url": existing_base_url,
                    "confidence": confidence,
                    "detection_method": detection_method,
                    "auto_applied": False
                }

                # Apply if no user-set URL exists (or force_override is on)
                if not has_user_set_url:
                    if progress_callback:
                        progress_callback(85, 100, "Updating manifest...")
                    try:
                        existing_manifest["base_url"] = detected_base_url
                        with open(manifest_path, "w", encoding="utf-8") as f:
                            json.dump(existing_manifest, f, indent=2)
                        result["auto_applied"] = True
                        result["message"] = f"Base URL set to: {detected_base_url} (from ZIM metadata)"
                    except Exception as e:
                        result["warning"] = f"Could not update manifest: {str(e)}"
                else:
                    result["message"] = f"Keeping existing base URL: {existing_base_url} (check 'Override Existing' to replace)"

                if progress_callback:
                    progress_callback(100, 100, "Detection complete")

                return result

            # If no ZIM detection and no pages folder, skip gracefully with warning
            if not pages_path.exists():
                return {
                    "success": True,  # Don't fail the chain, just warn
                    "skipped": True,
                    "warning": "No pages folder found and could not detect from ZIM metadata. Base URL not set - please configure manually before indexing.",
                    "detected_base_url": None,
                    "message": "Skipped: No pages folder. Set base_url manually in source editor."
                }

            # Collect all HTML files
            if progress_callback:
                progress_callback(10, 100, "Finding HTML files...")

            html_files = list(pages_path.rglob("*.html"))

            if not html_files:
                return {"success": False, "error": "No HTML files found in pages/"}

            # Sample random files
            sample_size = min(sample_count, len(html_files))
            sampled = random.sample(html_files, sample_size)

            if progress_callback:
                progress_callback(30, 100, f"Analyzing {sample_size} sample URLs...")

            # Analyze file paths to detect URL patterns
            # The file path relative to pages/ represents the URL path
            sample_paths = []
            for html_file in sampled:
                rel_path = html_file.relative_to(pages_path)
                # Convert Windows path separators and remove .html extension
                url_path = str(rel_path).replace("\\", "/")
                if url_path.endswith(".html"):
                    url_path = url_path[:-5]  # Remove .html
                sample_paths.append(url_path)

            if progress_callback:
                progress_callback(50, 100, "Detecting base URL pattern...")

            # At this point, ZIM metadata detection already happened above and returned early
            # if successful. Now we analyze page paths for scraped sites.

            # Try analyzing the paths to detect URL patterns
            if sample_paths:
                # Look for common patterns in the sample paths
                # Check if paths have a common prefix (like wiki/, A/, etc.)
                first_segments = [p.split("/")[0] if "/" in p else "" for p in sample_paths]
                common_prefix = None
                if first_segments:
                    # See if all samples have the same first segment
                    first_seg_counts = {}
                    for seg in first_segments:
                        if seg:
                            first_seg_counts[seg] = first_seg_counts.get(seg, 0) + 1

                    if first_seg_counts:
                        most_common = max(first_seg_counts.items(), key=lambda x: x[1])
                        if most_common[1] >= len(sample_paths) * 0.8:  # 80% have this prefix
                            common_prefix = most_common[0]

                # Check original_url in manifest (for scraped sites)
                original_url = existing_manifest.get("original_url", "")
                if original_url:
                    # Use the original scrape URL as base
                    detected_base_url = original_url.rstrip("/") + "/"
                    confidence = "high"
                    detection_method = "original_url"
                elif common_prefix:
                    # Just note the common prefix for user reference
                    detected_base_url = f"https://example.org/{common_prefix}/"
                    confidence = "low"
                    detection_method = "common_prefix"

            if progress_callback:
                progress_callback(70, 100, "Preparing results...")

            # Prepare result
            result = {
                "success": True,
                "source_id": source_id,
                "total_files": len(html_files),
                "samples_analyzed": sample_size,
                "sample_paths": sample_paths[:5],  # Show first 5 samples
                "detected_base_url": detected_base_url,
                "existing_base_url": existing_base_url,
                "confidence": confidence,
                "detection_method": detection_method,
                "auto_applied": False
            }

            # Apply if confident and no user-set URL (or force_override is on)
            if detected_base_url and confidence in ("high", "medium") and not has_user_set_url:
                if progress_callback:
                    progress_callback(85, 100, "Updating manifest...")

                try:
                    existing_manifest["base_url"] = detected_base_url
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        json.dump(existing_manifest, f, indent=2)
                    result["auto_applied"] = True
                    result["message"] = f"Base URL set to: {detected_base_url}"
                except Exception as e:
                    result["warning"] = f"Could not update manifest: {str(e)}"
            elif has_user_set_url:
                result["message"] = f"Keeping existing base URL: {existing_base_url} (check 'Override Existing' to replace)"
            elif detected_base_url:
                result["message"] = f"Detected base URL: {detected_base_url} (confidence: {confidence}) - review and set manually if needed"
            else:
                result["message"] = "Could not auto-detect base URL. Please set manually in source editor."

            # For ZIM sources, pre-build URL index for fast offline content serving
            # This avoids delay when user first tests offline links
            if existing_manifest.get("created_from") == "zim_import":
                zim_path = existing_manifest.get("zim_path")
                if zim_path and Path(zim_path).exists():
                    if progress_callback:
                        progress_callback(90, 100, "Building ZIM URL index for offline preview...")

                    try:
                        from admin.routes.search_test import get_cached_zim, build_url_index
                        zim = get_cached_zim(zim_path)
                        if zim:
                            url_index = build_url_index(zim_path, zim)
                            result["zim_index_entries"] = len(url_index)
                            result["message"] = (result.get("message", "") +
                                f" | ZIM index built: {len(url_index)} entries for offline preview")
                    except Exception as e:
                        # Non-fatal - index will be built on first preview request
                        print(f"[detect_base_url] Could not pre-build ZIM index: {e}")

            if progress_callback:
                progress_callback(100, 100, "Detection complete")

            return result

        return run_detect_base_url

    # Add more job types as needed...

    return None


# =============================================================================
# RESUME ENDPOINT FOR CUSTOM CHAINS
# =============================================================================

class ResumeChainRequest(BaseModel):
    source_id: str
    job_type: str = "custom_chain"


@router.post("/resume")
async def resume_chain(request: ResumeChainRequest):
    """
    Resume an interrupted custom chain job from checkpoint.

    Loads the checkpoint, recreates phases from stored definitions, and continues.
    """
    from admin.job_manager import get_job_manager, JobPhase, run_combined_job, load_checkpoint
    from admin.job_schemas import get_job_schema

    # Load checkpoint
    checkpoint = load_checkpoint(request.source_id, request.job_type)
    if not checkpoint:
        raise HTTPException(404, f"No checkpoint found for {request.source_id}/{request.job_type}")

    if not checkpoint.phase_definitions:
        raise HTTPException(400, "Checkpoint has no phase definitions - cannot resume. Please discard and restart.")

    print(f"[job_builder] Resuming {request.job_type} for {request.source_id}")
    print(f"[job_builder] Phases completed: {checkpoint.phases_completed}")
    print(f"[job_builder] Phase definitions: {[p.get('job_type') for p in checkpoint.phase_definitions]}")

    # Recreate phases from checkpoint
    phases = []
    phase_defs = checkpoint.phase_definitions

    for i, phase_data in enumerate(phase_defs):
        job_type = phase_data.get("job_type")
        params = phase_data.get("params", {})

        schema = get_job_schema(job_type)
        if not schema:
            raise HTTPException(400, f"Unknown job type in checkpoint: {job_type}")

        func = get_job_function(job_type, request.source_id, params)
        if not func:
            raise HTTPException(400, f"No function found for job type: {job_type}")

        phases.append(JobPhase(
            name=schema.label,
            func=func,
            weight=schema.weight,
            args=(),
            kwargs={}
        ))

    # Create wrapper that resumes
    def _resume_custom_chain(progress_callback=None, cancel_checker=None):
        print(f"[job_builder] Resuming custom chain: {request.job_type} for {request.source_id}")
        try:
            result = run_combined_job(
                phases,
                progress_callback=progress_callback,
                cancel_checker=cancel_checker,
                source_id=request.source_id,
                job_type=request.job_type,
                resume=True,
                phase_definitions=phase_defs
            )
            print(f"[job_builder] Resume completed: {result.get('success', 'unknown')}")
            return result
        except Exception as e:
            print(f"[job_builder] Resume error: {e}")
            import traceback
            traceback.print_exc()
            return {"success": False, "error": str(e)}

    # Submit to job manager
    manager = get_job_manager()

    try:
        job_id = manager.submit(
            request.job_type,
            request.source_id,
            _resume_custom_chain
        )
        print(f"[job_builder] Resume job submitted: {job_id}")
    except ValueError as e:
        raise HTTPException(409, str(e))

    return {
        "status": "resumed",
        "job_id": job_id,
        "source_id": request.source_id,
        "phases_completed": len(checkpoint.phases_completed),
        "phases_total": len(phase_defs),
        "message": f"Resuming from phase {len(checkpoint.phases_completed) + 1} of {len(phase_defs)}"
    }
