"""
Background Job Manager for long-running tasks.

Provides:
- Job queue with unique IDs
- Background execution (thread-based)
- Progress tracking and status updates
- Persistence across page navigation
- Conflict prevention (one job per source at a time)
- Checkpoint system for resumable jobs
"""

import threading
import uuid
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import traceback


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"  # Was running when server stopped


def _result_to_dict(result: Any) -> Dict[str, Any]:
    """Convert a job result to a JSON-serializable dict.

    Handles:
    - dict: returned as-is
    - dataclass: converted via asdict()
    - objects with to_dict(): calls to_dict()
    - other: wrapped in {"result": str(result)}
    """
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    # Check if it's a dataclass
    if hasattr(result, '__dataclass_fields__'):
        return asdict(result)
    # Check if it has to_dict method
    if hasattr(result, 'to_dict') and callable(result.to_dict):
        return result.to_dict()
    # Fallback: wrap as string
    return {"result": str(result)}


# =============================================================================
# CHECKPOINT SYSTEM
# =============================================================================

@dataclass
class Checkpoint:
    """
    Checkpoint data for resumable jobs.

    Stored in BACKUP_PATH/_jobs/{source_id}_{job_type}.checkpoint.json

    Future: For parallel processing, worker_id will allow multiple workers
    to process different ranges (e.g., worker 0: articles 0-25000,
    worker 1: articles 25001-50000). Each worker has its own checkpoint
    and partial file. A merger step combines results when all complete.
    """
    job_type: str
    source_id: str
    progress: int = 0
    created_at: str = ""
    last_saved: str = ""

    # Worker ID for future parallel processing (0 = single worker/default)
    worker_id: int = 0
    total_workers: int = 1
    work_range_start: int = 0  # For parallel: start of this worker's range
    work_range_end: int = 0    # For parallel: end of this worker's range (0 = all)

    # Metadata-specific checkpoint data
    last_article_index: int = 0
    partial_file: str = ""  # Path to partial work file
    documents_processed: int = 0

    # Index-specific checkpoint data
    indexed_doc_ids: List[str] = field(default_factory=list)
    batch_number: int = 0

    # Error tracking
    errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        return cls(
            job_type=data.get("job_type", ""),
            source_id=data.get("source_id", ""),
            progress=data.get("progress", 0),
            created_at=data.get("created_at", ""),
            last_saved=data.get("last_saved", ""),
            worker_id=data.get("worker_id", 0),
            total_workers=data.get("total_workers", 1),
            work_range_start=data.get("work_range_start", 0),
            work_range_end=data.get("work_range_end", 0),
            last_article_index=data.get("last_article_index", 0),
            partial_file=data.get("partial_file", ""),
            documents_processed=data.get("documents_processed", 0),
            indexed_doc_ids=data.get("indexed_doc_ids", []),
            batch_number=data.get("batch_number", 0),
            errors=data.get("errors", [])
        )


def get_jobs_folder() -> Optional[Path]:
    """
    Get the _jobs folder path for checkpoint storage.

    Returns BACKUP_PATH/_jobs/ or None if backup folder not configured.
    """
    try:
        from admin.local_config import get_local_config
        config = get_local_config()
        backup_folder = config.get_backup_folder()
        if not backup_folder:
            return None
        jobs_folder = Path(backup_folder) / "_jobs"
        jobs_folder.mkdir(parents=True, exist_ok=True)
        return jobs_folder
    except Exception as e:
        print(f"[checkpoint] Error getting jobs folder: {e}")
        return None


def get_checkpoint_path(source_id: str, job_type: str) -> Optional[Path]:
    """Get the path for a checkpoint file."""
    jobs_folder = get_jobs_folder()
    if not jobs_folder:
        return None
    return jobs_folder / f"{source_id}_{job_type}.checkpoint.json"


def get_partial_file_path(source_id: str, job_type: str) -> Optional[Path]:
    """Get the path for a partial work file."""
    jobs_folder = get_jobs_folder()
    if not jobs_folder:
        return None
    return jobs_folder / f"{source_id}_{job_type}.partial.json"


def save_checkpoint(checkpoint: Checkpoint) -> bool:
    """
    Save a checkpoint to disk.

    Uses atomic write (temp file + rename) to prevent corruption.
    """
    checkpoint_path = get_checkpoint_path(checkpoint.source_id, checkpoint.job_type)
    if not checkpoint_path:
        print("[checkpoint] Cannot save - no jobs folder configured")
        return False

    checkpoint.last_saved = datetime.now().isoformat()
    if not checkpoint.created_at:
        checkpoint.created_at = checkpoint.last_saved

    try:
        # Atomic write: write to temp file, then rename
        temp_path = checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        temp_path.replace(checkpoint_path)
        print(f"[checkpoint] Saved: {checkpoint_path.name} (progress: {checkpoint.progress}%)")
        return True
    except Exception as e:
        print(f"[checkpoint] Error saving: {e}")
        return False


def load_checkpoint(source_id: str, job_type: str) -> Optional[Checkpoint]:
    """Load a checkpoint from disk if it exists."""
    checkpoint_path = get_checkpoint_path(source_id, job_type)
    if not checkpoint_path or not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        checkpoint = Checkpoint.from_dict(data)
        print(f"[checkpoint] Loaded: {checkpoint_path.name} (progress: {checkpoint.progress}%)")
        return checkpoint
    except Exception as e:
        print(f"[checkpoint] Error loading: {e}")
        return None


def delete_checkpoint(source_id: str, job_type: str) -> bool:
    """Delete a checkpoint and its partial file on successful completion."""
    checkpoint_path = get_checkpoint_path(source_id, job_type)
    partial_path = get_partial_file_path(source_id, job_type)

    deleted = False

    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            print(f"[checkpoint] Deleted: {checkpoint_path.name}")
            deleted = True
        except Exception as e:
            print(f"[checkpoint] Error deleting checkpoint: {e}")

    if partial_path and partial_path.exists():
        try:
            partial_path.unlink()
            print(f"[checkpoint] Deleted partial file: {partial_path.name}")
            deleted = True
        except Exception as e:
            print(f"[checkpoint] Error deleting partial file: {e}")

    return deleted


def get_interrupted_jobs() -> List[Dict[str, Any]]:
    """
    Scan the _jobs folder for checkpoint files (interrupted jobs).

    Returns list of checkpoint info for jobs that can be resumed.
    """
    jobs_folder = get_jobs_folder()
    if not jobs_folder:
        return []

    interrupted = []

    for checkpoint_file in jobs_folder.glob("*.checkpoint.json"):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Calculate age
            last_saved = data.get("last_saved", "")
            if last_saved:
                try:
                    saved_dt = datetime.fromisoformat(last_saved)
                    age_hours = (datetime.now() - saved_dt).total_seconds() / 3600
                    age_str = f"{int(age_hours)}h ago" if age_hours < 24 else f"{int(age_hours/24)}d ago"
                except:
                    age_str = "unknown"
            else:
                age_str = "unknown"

            interrupted.append({
                "source_id": data.get("source_id", ""),
                "job_type": data.get("job_type", ""),
                "progress": data.get("progress", 0),
                "last_saved": last_saved,
                "age": age_str,
                "last_article_index": data.get("last_article_index", 0),
                "checkpoint_file": checkpoint_file.name
            })
        except Exception as e:
            print(f"[checkpoint] Error reading {checkpoint_file.name}: {e}")

    return sorted(interrupted, key=lambda x: x.get("last_saved", ""), reverse=True)


def cleanup_stale_checkpoints(max_age_days: int = 7) -> Dict[str, Any]:
    """
    Delete checkpoint files older than max_age_days.

    Returns info about deleted files.
    """
    jobs_folder = get_jobs_folder()
    if not jobs_folder:
        return {"deleted": [], "errors": []}

    cutoff = datetime.now() - timedelta(days=max_age_days)
    deleted = []
    errors = []

    for checkpoint_file in jobs_folder.glob("*.checkpoint.json"):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            last_saved = data.get("last_saved", "")
            if last_saved:
                saved_dt = datetime.fromisoformat(last_saved)
                if saved_dt < cutoff:
                    # Delete checkpoint and partial file
                    source_id = data.get("source_id", "")
                    job_type = data.get("job_type", "")

                    checkpoint_file.unlink()
                    deleted.append(checkpoint_file.name)

                    # Also delete partial file if exists
                    partial_path = get_partial_file_path(source_id, job_type)
                    if partial_path and partial_path.exists():
                        partial_path.unlink()
                        deleted.append(partial_path.name)

                    print(f"[checkpoint] Cleaned up stale: {checkpoint_file.name}")
        except Exception as e:
            errors.append(f"{checkpoint_file.name}: {e}")

    return {"deleted": deleted, "errors": errors}


# =============================================================================
# JOB HISTORY PERSISTENCE
# =============================================================================

def get_history_file_path() -> Optional[Path]:
    """Get the path for the job history file."""
    jobs_folder = get_jobs_folder()
    if not jobs_folder:
        return None
    return jobs_folder / "history.json"


def save_job_to_history(job_dict: Dict[str, Any]) -> bool:
    """
    Append a completed job to the history file.

    Jobs are stored as JSON lines (one JSON object per line) for efficient appending.
    """
    history_path = get_history_file_path()
    if not history_path:
        return False

    try:
        # Append as JSON line
        with open(history_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(job_dict) + '\n')
        return True
    except Exception as e:
        print(f"[job_history] Error saving job: {e}")
        return False


def load_job_history(limit: int = 100, max_age_days: int = 7) -> List[Dict[str, Any]]:
    """
    Load job history from disk.

    Args:
        limit: Maximum number of jobs to return (most recent first)
        max_age_days: Only return jobs from the last N days

    Returns:
        List of job dicts, sorted by completion time (newest first)
    """
    history_path = get_history_file_path()
    if not history_path or not history_path.exists():
        return []

    cutoff = datetime.now() - timedelta(days=max_age_days)
    jobs = []

    try:
        with open(history_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    job = json.loads(line)
                    # Filter by age
                    completed_at = job.get("completed_at", "")
                    if completed_at:
                        try:
                            completed_dt = datetime.fromisoformat(completed_at)
                            if completed_dt >= cutoff:
                                jobs.append(job)
                        except:
                            jobs.append(job)  # Keep if can't parse date
                    else:
                        jobs.append(job)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"[job_history] Error loading history: {e}")
        return []

    # Sort by completion time (newest first) and limit
    jobs.sort(key=lambda x: x.get("completed_at", ""), reverse=True)
    return jobs[:limit]


def cleanup_old_history(max_age_days: int = 7) -> Dict[str, Any]:
    """
    Remove job history entries older than max_age_days.

    Rewrites the history file with only recent entries.
    """
    history_path = get_history_file_path()
    if not history_path or not history_path.exists():
        return {"removed": 0, "kept": 0}

    cutoff = datetime.now() - timedelta(days=max_age_days)
    kept_jobs = []
    removed_count = 0

    try:
        # Read all jobs
        with open(history_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    job = json.loads(line)
                    completed_at = job.get("completed_at", "")
                    if completed_at:
                        try:
                            completed_dt = datetime.fromisoformat(completed_at)
                            if completed_dt >= cutoff:
                                kept_jobs.append(job)
                            else:
                                removed_count += 1
                        except:
                            kept_jobs.append(job)
                    else:
                        kept_jobs.append(job)
                except json.JSONDecodeError:
                    continue

        # Rewrite file with only kept jobs
        with open(history_path, 'w', encoding='utf-8') as f:
            for job in kept_jobs:
                f.write(json.dumps(job) + '\n')

        if removed_count > 0:
            print(f"[job_history] Cleaned up {removed_count} old entries, kept {len(kept_jobs)}")

        return {"removed": removed_count, "kept": len(kept_jobs)}
    except Exception as e:
        print(f"[job_history] Error cleaning history: {e}")
        return {"removed": 0, "kept": 0, "error": str(e)}


@dataclass
class Job:
    id: str
    job_type: str  # e.g., "index", "download", "upload", "metadata"
    source_id: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0  # 0-100
    progress_current: int = 0  # Current item being processed
    progress_total: int = 0    # Total items to process
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
            "progress_current": self.progress_current,
            "progress_total": self.progress_total,
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
                    job.progress_current = int(current)
                    job.progress_total = int(total)
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

            # Add cancellation checker - functions can call this to check if they should stop
            def check_cancelled():
                return job.status == JobStatus.CANCELLED

            kwargs['cancel_checker'] = check_cancelled

            # Run the function
            result = func(*args, **kwargs)

            # Convert result to dict (handles dataclasses, dicts, etc.)
            result_dict = _result_to_dict(result)

            # Check if job was cancelled (either by check_cancelled or result status)
            if result_dict.get("status") == "cancelled":
                job.status = JobStatus.CANCELLED
                job.message = "Cancelled by user (checkpoint saved)"
                job.result = result_dict
            # Check if result indicates failure (success: False)
            elif result_dict.get("success") is False:
                job.status = JobStatus.FAILED
                job.error = result_dict.get("error", "Operation failed")
                job.message = f"Failed: {result_dict.get('error', 'Unknown error')}"
                job.result = result_dict
            else:
                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.result = result_dict
                job.message = "Completed successfully"

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.message = f"Failed: {str(e)}"
            # Log the full traceback for debugging
            print(f"Job {job_id} failed: {traceback.format_exc()}")

        finally:
            job.completed_at = datetime.now()
            # Save completed/failed/cancelled jobs to history
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                save_job_to_history(job.to_dict())
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

    def get_job_history(self, limit: int = 50, max_age_days: int = 7) -> List[Dict[str, Any]]:
        """
        Get completed job history from disk.

        Returns only completed/failed/cancelled jobs, not active ones.
        """
        return load_job_history(limit=limit, max_age_days=max_age_days)

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
