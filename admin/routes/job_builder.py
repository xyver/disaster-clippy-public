"""
Job Builder API

Routes for the visual job chain builder.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter(prefix="/api/job-builder", tags=["Job Builder"])


# =============================================================================
# REQUEST MODELS
# =============================================================================

class ValidateChainRequest(BaseModel):
    job_types: List[str]


class RunChainRequest(BaseModel):
    source_id: str
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


@router.post("/validate")
async def validate_chain(request: ValidateChainRequest):
    """Validate a job chain for ordering issues and conflicts"""
    from admin.job_schemas import validate_job_chain
    return validate_job_chain(request.job_types)


@router.post("/run")
async def run_chain(request: RunChainRequest):
    """
    Run a custom job chain as a combined job.

    Creates JobPhase objects from the request and runs them using run_combined_job.
    """
    from admin.job_manager import get_job_manager, JobPhase, run_combined_job
    from admin.job_schemas import get_job_schema

    if not request.phases:
        raise HTTPException(400, "No phases provided")

    if not request.source_id:
        # Check if any job needs a source
        for phase in request.phases:
            schema = get_job_schema(phase.get("job_type"))
            if schema and schema.requires_source:
                raise HTTPException(400, f"Source ID required for job: {phase.get('job_type')}")

    # Build JobPhase objects
    phases = []
    for i, phase_data in enumerate(request.phases):
        job_type = phase_data.get("job_type")
        params = phase_data.get("params", {})

        schema = get_job_schema(job_type)
        if not schema:
            raise HTTPException(400, f"Unknown job type: {job_type}")

        # Get the actual job function based on job type
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

    # Determine combined job type name
    job_type_name = request.chain_name or "custom_chain"

    # Create wrapper function that runs the combined job
    def _run_custom_chain(progress_callback=None, cancel_checker=None):
        return run_combined_job(
            phases,
            progress_callback=progress_callback,
            cancel_checker=cancel_checker,
            source_id=request.source_id,
            job_type=job_type_name,
            resume=request.resumable
        )

    # Submit to job manager
    manager = get_job_manager()

    try:
        job_id = manager.submit(
            job_type_name,
            request.source_id,
            _run_custom_chain
        )
    except ValueError as e:
        raise HTTPException(409, str(e))

    return {
        "status": "submitted",
        "job_id": job_id,
        "source_id": request.source_id,
        "phases_count": len(phases),
        "resumable": request.resumable,
        "message": f"Custom chain started with {len(phases)} phases"
    }


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

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

            result = manager.generate_metadata(
                source_id,
                language_filter=params.get("language_filter"),
                progress_callback=lambda curr, total, msg: progress_callback(
                    int((curr / max(total, 1)) * 100), 100, msg
                ) if progress_callback else None
            )

            if progress_callback:
                progress_callback(100, 100, "Metadata complete")

            return {"success": True, **result} if isinstance(result, dict) else {"success": True, "result": result}

        return run_metadata

    elif job_type == "index_online":
        def run_index_online(progress_callback=None, cancel_checker=None):
            from offline_tools.vectordb import create_source_index

            limit = int(params.get("limit", 1000))
            force = params.get("force_reindex", False)
            lang = params.get("language_filter")

            result = create_source_index(
                source_id,
                limit=limit,
                force_reindex=force,
                dimension=1536,
                progress_callback=progress_callback,
                cancel_checker=cancel_checker
            )
            return result

        return run_index_online

    elif job_type == "index_offline":
        def run_index_offline(progress_callback=None, cancel_checker=None):
            from offline_tools.vectordb import create_source_index

            limit = int(params.get("limit", 1000))
            force = params.get("force_reindex", False)
            dimension = int(params.get("dimension", 768))

            result = create_source_index(
                source_id,
                limit=limit,
                force_reindex=force,
                dimension=dimension,
                progress_callback=progress_callback,
                cancel_checker=cancel_checker
            )
            return result

        return run_index_offline

    elif job_type == "clear_vectors":
        def run_clear_vectors(progress_callback=None, cancel_checker=None):
            from offline_tools.vectordb import get_vector_store

            dimension = int(params.get("dimension", 768))

            if progress_callback:
                progress_callback(0, 100, f"Clearing {dimension}-dim vectors...")

            store = get_vector_store(mode="local", dimension=dimension, read_only=True)
            result = store.delete_by_source(source_id)
            deleted = result.get("deleted_count", 0) if isinstance(result, dict) else 0

            if progress_callback:
                progress_callback(100, 100, f"Cleared {deleted} vectors")

            return {"success": True, "deleted_count": deleted}

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

    # Add more job types as needed...

    return None
