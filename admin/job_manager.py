"""
Background Job Manager for long-running tasks.

Provides:
- Job queue with unique IDs
- Background execution (thread-based)
- Progress tracking and status updates
- Persistence across page navigation
- Conflict prevention (one job per source at a time)
"""

import threading
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import traceback


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    job_type: str  # e.g., "index", "download", "upload", "metadata"
    source_id: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0  # 0-100
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_type": self.job_type,
            "source_id": self.source_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self._get_duration()
        }

    def _get_duration(self) -> Optional[float]:
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds()


class JobManager:
    """
    Singleton job manager for background tasks.

    Usage:
        manager = JobManager()
        job_id = manager.submit("index", "builditsolar", index_function, arg1, arg2)
        status = manager.get_status(job_id)
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._jobs: Dict[str, Job] = {}
        self._job_lock = threading.Lock()
        self._max_history = 50  # Keep last N completed jobs
        self._initialized = True

    def submit(self, job_type: str, source_id: str,
               func: Callable, *args, **kwargs) -> str:
        """
        Submit a job for background execution.

        Args:
            job_type: Type of job (index, download, upload, metadata)
            source_id: Source being operated on
            func: Function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Job ID

        Raises:
            ValueError: If a job is already running for this source
        """
        # Check for conflicting jobs
        with self._job_lock:
            for job in self._jobs.values():
                if (job.source_id == source_id and
                    job.status in [JobStatus.PENDING, JobStatus.RUNNING]):
                    raise ValueError(
                        f"A {job.job_type} job is already running for {source_id}"
                    )

        # Create job
        job_id = str(uuid.uuid4())[:8]
        job = Job(
            id=job_id,
            job_type=job_type,
            source_id=source_id,
            message=f"Starting {job_type}..."
        )

        with self._job_lock:
            self._jobs[job_id] = job

        # Start background thread
        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, func, args, kwargs),
            daemon=True
        )
        thread.start()

        return job_id

    def _run_job(self, job_id: str, func: Callable,
                 args: tuple, kwargs: dict):
        """Execute job in background thread."""
        job = self._jobs.get(job_id)
        if not job:
            return

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()

        try:
            # Create a progress callback for the function to use
            # Supports both formats:
            #   (current, total, message) - used by indexers
            #   (progress_percent, message) - simple percentage
            def update_progress(*args):
                if len(args) >= 3:
                    # Format: (current, total, message)
                    current, total, message = args[0], args[1], args[2]
                    if total > 0:
                        job.progress = min(100, max(0, int((current / total) * 100)))
                    if message:
                        job.message = str(message)
                elif len(args) >= 2:
                    # Format: (progress_percent, message)
                    progress, message = args[0], args[1]
                    job.progress = min(100, max(0, int(progress)))
                    if message:
                        job.message = str(message)
                elif len(args) == 1:
                    # Just progress percentage
                    job.progress = min(100, max(0, int(args[0])))

            # Add progress callback to kwargs if function accepts it
            kwargs['progress_callback'] = update_progress

            # Run the function
            result = func(*args, **kwargs)

            # Check if result indicates failure (some functions return success: False)
            if isinstance(result, dict) and result.get("success") is False:
                job.status = JobStatus.FAILED
                job.error = result.get("error", "Operation failed")
                job.message = f"Failed: {result.get('error', 'Unknown error')}"
                job.result = result
            else:
                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.result = result if isinstance(result, dict) else {"result": result}
                job.message = "Completed successfully"

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.message = f"Failed: {str(e)}"
            # Log the full traceback for debugging
            print(f"Job {job_id} failed: {traceback.format_exc()}")

        finally:
            job.completed_at = datetime.now()
            self._cleanup_old_jobs()

    def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        job = self._jobs.get(job_id)
        return job.to_dict() if job else None

    def get_active_jobs(self, source_id: str = None) -> List[Dict[str, Any]]:
        """Get all active (pending/running) jobs, optionally filtered by source."""
        with self._job_lock:
            jobs = [
                job.to_dict() for job in self._jobs.values()
                if job.status in [JobStatus.PENDING, JobStatus.RUNNING]
                and (source_id is None or job.source_id == source_id)
            ]
        return sorted(jobs, key=lambda j: j['created_at'], reverse=True)

    def get_all_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get all jobs (for status display)."""
        with self._job_lock:
            jobs = [job.to_dict() for job in self._jobs.values()]
        return sorted(jobs, key=lambda j: j['created_at'], reverse=True)[:limit]

    def cancel(self, job_id: str) -> bool:
        """
        Request job cancellation.
        Note: This only marks the job as cancelled - the function must check
        for cancellation and stop gracefully.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            job.status = JobStatus.CANCELLED
            job.message = "Cancelled by user"
            job.completed_at = datetime.now()
            return True

        return False

    def is_cancelled(self, job_id: str) -> bool:
        """Check if a job has been cancelled (for functions to check)."""
        job = self._jobs.get(job_id)
        return job.status == JobStatus.CANCELLED if job else False

    def _cleanup_old_jobs(self):
        """Remove old completed jobs to prevent memory growth."""
        with self._job_lock:
            completed = [
                (job_id, job) for job_id, job in self._jobs.items()
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
            ]

            # Sort by completion time, keep most recent
            completed.sort(key=lambda x: x[1].completed_at or datetime.min, reverse=True)

            # Remove old ones beyond max_history
            for job_id, _ in completed[self._max_history:]:
                del self._jobs[job_id]


# Global instance
_job_manager = None

def get_job_manager() -> JobManager:
    """Get the global job manager instance."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
