"""
Job Management API

Endpoints for monitoring and managing background jobs.
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/jobs", tags=["Job Management"])


def get_job_manager():
    """Get job manager - imported here to avoid circular imports"""
    from admin.job_manager import get_job_manager as _get_manager
    return _get_manager()


@router.get("")
async def get_all_jobs(limit: int = 20):
    """Get all jobs (recent history)."""
    manager = get_job_manager()
    return {
        "jobs": manager.get_all_jobs(limit=limit)
    }


@router.get("/active")
async def get_active_jobs(source_id: str = None):
    """Get currently active (pending/running) jobs."""
    manager = get_job_manager()
    return {
        "jobs": manager.get_active_jobs(source_id=source_id)
    }


@router.get("/history")
async def get_job_history(limit: int = 50, days: int = 7):
    """
    Get completed job history from disk.

    Args:
        limit: Maximum number of jobs to return (default 50)
        days: Maximum age of jobs in days (default 7)
    """
    manager = get_job_manager()
    return {
        "jobs": manager.get_job_history(limit=limit, max_age_days=days)
    }


@router.get("/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job."""
    manager = get_job_manager()
    status = manager.get_status(job_id)
    if not status:
        raise HTTPException(404, f"Job {job_id} not found")
    return status


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Request cancellation of a job."""
    manager = get_job_manager()
    if manager.cancel(job_id):
        return {"status": "cancelled", "job_id": job_id}
    raise HTTPException(400, f"Could not cancel job {job_id}")
