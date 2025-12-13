# Job Manager Documentation

## Overview

The job manager (`admin/job_manager.py`) provides background job execution with:
- Job queue with unique IDs
- Background execution (thread-based)
- Progress tracking and status updates
- Persistence across page navigation
- Conflict prevention (one job per source at a time)
- Checkpoint system for resumable jobs

## Job Types

### Single Jobs

| Job Type | Label | Description | Location |
|----------|-------|-------------|----------|
| `scrape` | Scrape | Scraping content from source | admin/routes/source_tools.py |
| `metadata` | Metadata | Generating document metadata | admin/routes/source_tools.py |
| `index_online` | Index (Online) | Creating 1536-dim vectors using OpenAI API | admin/routes/source_tools.py |
| `index_offline` | Index (Offline) | Creating vectors using local model (dimension varies) | admin/routes/source_tools.py |
| `clear_vectors` | Clear Vectors | Clearing ChromaDB entries | admin/routes/source_tools.py |
| `suggest_tags` | Suggest Tags | AI-powered tag suggestions | admin/routes/source_tools.py |
| `delete_source` | Delete | Removing source from system | admin/routes/source_tools.py |
| `zim_inspect` | ZIM Inspect | Inspecting ZIM file contents | admin/routes/source_tools.py |
| `zim_index` | ZIM Index | Indexing ZIM file contents | admin/routes/source_tools.py |
| `install_source` | Install | Download source from cloud + import to ChromaDB | admin/routes/source_tools.py |
| `upload` | Upload | Uploading source to cloud storage | admin/cloud_upload.py |
| `pinecone_sync` | Pinecone | Syncing vectors to Pinecone cloud | admin/cloud_upload.py |
| `pinecone_sync_dry` | Pinecone (Dry) | Checking what would sync to Pinecone | admin/cloud_upload.py |
| `visualisation` | Visualize | Generating 3D knowledge map | admin/routes/visualise.py |
| `model_download` | Model Download | Downloading AI models | admin/routes/models.py |
| `language_download` | Language Download | Downloading language packs | admin/routes/models.py |

### Combined Jobs

| Job Type | Label | Description | Phases |
|----------|-------|-------------|--------|
| `reindex_online` | Reindex (Online) | Clear + fresh 1536-dim index | clear_vectors (10%) + index_online (90%) |
| `reindex_offline` | Reindex (Offline) | Clear + fresh local index | clear_vectors (10%) + index_offline (90%) |
| `cloud_publish` | Cloud Publish | Sync to Pinecone + regenerate viz | pinecone_sync (80%) + visualisation (20%) |

## Embedding Dimensions

| Mode | Model | Dimension | Use Case |
|------|-------|-----------|----------|
| Online | OpenAI `text-embedding-3-small` (API) | 1536 (fixed) | Cloud search (Pinecone) |
| Offline | Configured local model | Variable (384/768/1024) | Local search (ChromaDB) |

**Local model examples:**
- `all-MiniLM-L6-v2` - 384-dim
- `all-mpnet-base-v2` - 768-dim (default)
- `bge-large-en-v1.5` - 1024-dim

## Combined Job Framework

The `run_combined_job` function allows chaining multiple jobs sequentially with unified progress tracking.

### JobPhase Dataclass

```python
@dataclass
class JobPhase:
    name: str           # Display name (e.g., "Syncing vectors")
    func: Callable      # Job function to call
    weight: int = 50    # Progress weight (0-100, all should sum to 100)
    args: tuple = ()    # Positional arguments
    kwargs: Dict = {}   # Keyword arguments
```

### Usage Example

```python
from admin.job_manager import JobPhase, run_combined_job

phases = [
    JobPhase(
        name="Clear existing vectors",
        func=clear_vectors_func,
        weight=10,
        args=(source_id, dimension)
    ),
    JobPhase(
        name="Create new index",
        func=create_index_func,
        weight=90,
        args=(source_id, limit, dimension)
    ),
]

result = run_combined_job(phases, progress_callback, cancel_checker)
```

### Resumable Combined Jobs

Combined jobs support checkpointing for resume capability:

```python
# First run - saves checkpoint after each phase
result = run_combined_job(
    phases,
    progress_callback,
    cancel_checker,
    source_id="my_source",
    job_type="generate_source",
    resume=False  # Start fresh
)

# Resume after failure - skips completed phases
result = run_combined_job(
    phases,
    progress_callback,
    cancel_checker,
    source_id="my_source",
    job_type="generate_source",
    resume=True  # Load checkpoint and skip completed phases
)
```

**How it works:**
1. After each phase completes, saves checkpoint with `phases_completed` list
2. On resume, loads checkpoint and skips phases in that list
3. On full completion, deletes checkpoint
4. Failed/cancelled jobs keep checkpoint for later resume

### Return Value

```python
{
    "success": True/False,
    "status": "success" | "cancelled" | None,
    "phases_completed": int,
    "phase_results": [
        {"phase": "Phase Name", "index": 0, "result": {...}},
        ...
    ],
    "message": "Summary message",
    "resumed": True/False,      # Was this resumed from checkpoint?
    "phases_skipped": int,      # Phases skipped due to resume
    # On failure:
    "current_phase": int,       # Where it stopped
    "phase_name": str,          # Name of failed phase
    "error": str                # Error message
}
```

## Checkpoint System

Jobs can save progress to `BACKUP_PATH/_jobs/` for resumability:

- `{source_id}_{job_type}.checkpoint.json` - Checkpoint state
- `{source_id}_{job_type}.partial.json` - Partial work file

### Key Functions

- `save_checkpoint(checkpoint)` - Save progress atomically
- `load_checkpoint(source_id, job_type)` - Load existing checkpoint
- `delete_checkpoint(source_id, job_type)` - Clean up on completion
- `get_interrupted_jobs()` - List resumable jobs
- `cleanup_stale_checkpoints(max_age_days)` - Remove old checkpoints

## API Usage

```python
from admin.job_manager import get_job_manager

manager = get_job_manager()

# Submit a job
job_id = manager.submit(
    job_type="index_online",
    source_id="my_source",
    func=my_index_function,
    arg1, arg2,
    kwarg1=value1
)

# Check status
status = manager.get_status(job_id)

# Get active jobs
active = manager.get_active_jobs(source_id="my_source")

# Cancel a job
manager.cancel(job_id)

# Get job history
history = manager.get_job_history(limit=50, max_age_days=7)
```

---

## Development Notes

### Recent Changes (Dec 2024)

**Job Naming Standardization:**
- Renamed `index` -> `index_online` (1536-dim, OpenAI API) / `index_offline` (local model, variable dim)
- Renamed `reindex` -> `reindex_online` / `reindex_offline` (dynamic based on dimension parameter)
- Consolidated `download` job into `install_source` (was duplicate functionality in packs.py)
- Job type is determined dynamically: `"index_online" if dimension == 1536 else "index_offline"`

**Checkpoint Support for Combined Jobs:**
- Added `phases_completed` and `phase_results` fields to Checkpoint class
- `run_combined_job` now accepts `source_id`, `job_type`, `resume` parameters
- Saves checkpoint after each phase completes
- On resume, skips phases already in `phases_completed` list

**Legacy Job Names (for backwards compatibility in job history):**
- `index` -> now `index_online` or `index_offline`
- `reindex` -> now `reindex_online` or `reindex_offline`
- `download` -> now `install_source`
- `vectors_768` -> still exists for pack vector generation

### Job Builder UI (Implemented)

A visual job chain builder allowing users to create custom combined jobs.

**Location:** `/useradmin/job-builder` (tab under Sources: "All Sources | Source Tools | Job Builder")

**Files:**
- `admin/job_schemas.py` - Job parameter schemas and validation
- `admin/routes/job_builder.py` - API endpoints
- `admin/templates/job_builder.html` - UI page

**Features:**
1. Click job -> adds to chain
2. Each job in chain is expandable to configure parameters
3. Drag to reorder jobs in chain
4. "Validate" - checks for issues (e.g., "Warning: Index before Metadata")
5. "Run Chain" - submits as combined job with checkpoint support
6. "Save Chain" - save as named template (localStorage)

**Predefined Combined Job Templates:**

| Job Type | Description | Phases | Mode |
|----------|-------------|--------|------|
| `generate_source` | Overnight batch processing | metadata + index_online + index_offline + suggest_tags | Resumable, incremental |
| `regenerate_source` | Full refresh/nuclear option | metadata (force) + clear + index_online + clear + index_offline + suggest_tags (force) | Non-resumable, clears first |
| `reindex_online` | Clear and rebuild cloud vectors | clear_vectors (1536) + index_online | Non-resumable |
| `reindex_offline` | Clear and rebuild local vectors | clear_vectors (768) + index_offline | Non-resumable |
| `cloud_publish` | Sync to Pinecone and visualize | pinecone_sync + visualisation | Resumable |

**Job Categories:**
- `content` - scrape, metadata, zim_inspect
- `indexing` - index_online, index_offline, clear_vectors
- `maintenance` - suggest_tags, delete_source
- `cloud` - install_source, upload, pinecone_sync, visualisation

**API Endpoints:**
- `GET /api/job-builder/schemas` - Get all job schemas
- `POST /api/job-builder/validate` - Validate job chain ordering
- `POST /api/job-builder/run` - Run a custom job chain
