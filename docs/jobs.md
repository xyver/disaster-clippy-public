# Jobs System

This document covers background job execution, the checkpoint system for resumable jobs, and the job builder UI.

---

## Table of Contents

1. [Overview](#overview)
2. [Job Types](#job-types)
3. [Combined Jobs](#combined-jobs)
4. [Checkpoint System](#checkpoint-system)
5. [Job Cancellation and Resume](#job-cancellation-and-resume)
6. [Job Builder UI](#job-builder-ui)
7. [API Usage](#api-usage)

---

## Overview

The job manager (`admin/job_manager.py`) provides background job execution with:
- Job queue with unique IDs
- Background execution (thread-based)
- Progress tracking and status updates
- Persistence across page navigation
- Conflict prevention (one job per source at a time)
- Checkpoint system for resumable jobs

---

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
| `zim_import` | ZIM Import | Import ZIM file (analyze + extract pages) | admin/routes/job_builder.py |
| `zim_index` | ZIM Index | Indexing ZIM file contents | admin/routes/source_tools.py |
| `install_source` | Install | Download source from cloud + import to ChromaDB | admin/routes/source_tools.py |
| `upload` | Upload | Uploading source to cloud storage | admin/cloud_upload.py |
| `pinecone_sync` | Pinecone | Syncing vectors to Pinecone cloud | admin/cloud_upload.py |
| `pinecone_sync_dry` | Pinecone (Dry) | Checking what would sync to Pinecone | admin/cloud_upload.py |
| `visualisation` | Visualize | Generating 3D knowledge map | admin/routes/visualise.py |
| `model_download` | Model Download | Downloading AI models | admin/routes/models.py |
| `language_download` | Language Download | Downloading language packs | admin/routes/models.py |

### Embedding Dimensions

| Mode | Model | Dimension | Use Case |
|------|-------|-----------|----------|
| Online | OpenAI `text-embedding-3-small` (API) | 1536 (fixed) | Cloud search (Pinecone) |
| Offline | Configured local model | Variable (384/768/1024) | Local search (ChromaDB) |

**Local model examples:**
- `all-MiniLM-L6-v2` - 384-dim
- `all-mpnet-base-v2` - 768-dim (default)
- `bge-large-en-v1.5` - 1024-dim

---

## Combined Jobs

Combined jobs chain multiple jobs sequentially with unified progress tracking.

### Combined Job Types

| Job Type | Label | Description | Phases |
|----------|-------|-------------|--------|
| `reindex_online` | Reindex (Online) | Clear + fresh 1536-dim index | clear_vectors (10%) + index_online (90%) |
| `reindex_offline` | Reindex (Offline) | Clear + fresh local index | clear_vectors (10%) + index_offline (90%) |
| `cloud_publish` | Cloud Publish | Sync to Pinecone + regenerate viz | pinecone_sync (80%) + visualisation (20%) |
| `generate_source` | Generate Source | Overnight batch processing | metadata + index_online + index_offline + suggest_tags |
| `regenerate_source` | Regenerate Source | Full refresh | metadata (force) + clear + index_online + clear + index_offline + suggest_tags (force) |

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

---

## Checkpoint System

Jobs can save progress to `BACKUP_PATH/_jobs/` for resumability.

### Checkpoint Storage

```
BACKUP_PATH/_jobs/
  {source_id}_{job_type}.checkpoint.json    # Checkpoint state
  {source_id}_{job_type}.partial.json       # Partial work file
```

### Jobs with Checkpoints

| Job Type | Checkpoint Data | Partial File | Status |
|----------|----------------|--------------|--------|
| Generate Metadata | `last_article_index` | `_metadata.partial.json` | IMPLEMENTED |
| Create Index (ZIM) | `indexed_doc_ids` | `_index.partial.json` | IMPLEMENTED |
| Create Index (HTML/PDF) | N/A | N/A | Uses Incremental Indexing |
| Upload to Cloud | N/A | N/A | NOT NEEDED - atomic file operations |
| Download from Cloud | N/A | N/A | ALREADY WORKS - smart skip by size |
| Scan Backup | N/A | N/A | NO (fast) |
| Validate | N/A | N/A | NO (fast) |

### Checkpoint File Structure

```json
{
  "job_type": "metadata",
  "source_id": "wikipedia-medical",
  "progress": 45,
  "created_at": "2025-12-06T20:00:00",
  "last_saved": "2025-12-06T20:15:00",
  "worker_id": 0,
  "total_workers": 1,
  "work_range_start": 0,
  "work_range_end": 450000,
  "last_article_index": 203847,
  "partial_file": "wikipedia-medical_metadata.partial.json",
  "documents_processed": 15234,
  "errors": [
    {"article_index": 1234, "error": "Parse error"}
  ]
}
```

### Partial File Structure

```json
{
  "source_id": "wikipedia-medical",
  "documents": {
    "zim_0": {"title": "...", "url": "...", "snippet": "...", ...},
    "zim_1": {"title": "...", "url": "...", "snippet": "...", ...}
  },
  "language_filtered": 5000,
  "last_article_index": 203847
}
```

### Key Functions

- `save_checkpoint(checkpoint)` - Save progress atomically
- `load_checkpoint(source_id, job_type)` - Load existing checkpoint
- `delete_checkpoint(source_id, job_type)` - Clean up on completion
- `get_interrupted_jobs()` - List resumable jobs
- `cleanup_stale_checkpoints(max_age_days)` - Remove old checkpoints

### Checkpoint Behavior

- **Save frequency:** Every 60 seconds OR every 2000 articles (whichever first)
- **On success:** Delete checkpoint file and partial file
- **On failure/interruption:** Keep files (allows resume)
- **Stale checkpoints:** Manual cleanup via Jobs page, or auto-delete after 7 days
- **Atomic writes:** Write to temp file, then rename (prevents corruption)

### Resumable Combined Jobs

Combined jobs support checkpointing:

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

---

## Job Cancellation and Resume

The Jobs page includes a "Stop" button to cancel running jobs. This is a **soft cancel** that saves progress for later resumption.

### Current Behavior

- Job status updates to "cancelled" immediately in the UI
- The background thread checks for cancellation periodically
- Progress is saved to checkpoint file before exiting
- Interrupted jobs appear in the "Interrupted Jobs" section on the Jobs page
- Users can click "Resume" to continue from where they left off

### Implementation

```python
# Inside the indexing loop
if cancel_checker and cancel_checker():
    # Save checkpoint before exiting
    save_checkpoint(checkpoint)
    with open(partial_path, 'w') as f:
        json.dump({"indexed_doc_ids": list(indexed_doc_ids), ...}, f)
    return {"success": False, "cancelled": True, "indexed_count": indexed}
```

### Resume Flow

```
User clicks "Generate Metadata"
    |
    v
Check: Checkpoint exists for (source_id, metadata)?
    |
    +-- YES --> Modal: "Incomplete job found (45%, 2 hrs ago)"
    |              [Resume] [Start Fresh] [Cancel]
    |
    +-- NO --> Start fresh job
```

**From Jobs page:** Interrupted jobs appear in a separate "Interrupted Jobs" section with Resume/Discard buttons.

### Incremental Indexing

All indexers (ZIM, HTML, PDF) use incremental indexing:
- Queries existing doc IDs, skips already-indexed
- Processes in batches of 100
- Persists after each batch
- If interrupted, just re-run - previously indexed docs are skipped automatically
- No checkpoint files needed - ChromaDB is the checkpoint

---

## Job Builder UI

A visual job chain builder allowing users to create custom combined jobs.

**Location:** `/useradmin/job-builder` (tab under Sources)

**Files:**
- `admin/job_schemas.py` - Job parameter schemas and validation
- `admin/routes/job_builder.py` - API endpoints
- `admin/templates/job_builder.html` - UI page

### Features

1. Click job -> adds to chain
2. Each job in chain is expandable to configure parameters
3. Drag to reorder jobs in chain
4. "Validate" - checks for issues (e.g., "Warning: Index before Metadata")
5. "Run Chain" - submits as combined job with checkpoint support
6. "Save Chain" - save as named template (localStorage)

### Job Categories

- `content` - scrape, metadata, zim_import
- `indexing` - index_online, index_offline, clear_vectors
- `maintenance` - suggest_tags, delete_source
- `cloud` - install_source, upload, pinecone_sync, visualisation

### API Endpoints

- `GET /api/job-builder/schemas` - Get all job schemas
- `POST /api/job-builder/validate` - Validate job chain ordering
- `POST /api/job-builder/run` - Run a custom job chain

---

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

### Dimension Tracking (Opportunistic Scheduling)

Jobs that interact with ChromaDB now track which dimension they're using. This enables:
- Monitoring which databases are busy
- Future opportunistic scheduling (if 768 is busy, work on 1536 first)
- Better visibility into concurrent job activity

**Job Class Fields** (`admin/job_manager.py`):
```python
current_dimension: Optional[int]  # 384, 768, 1024, or 1536
current_operation: Optional[str]  # "delete" or "index"
pending_dimensions: List[int]     # For future multi-dimension jobs
```

**JobManager Methods**:
- `get_dimension_status()` - Returns status of all 4 dimensions
- `is_dimension_busy(dim)` - Quick check for a single dimension
- `get_available_dimensions()` - List of dimensions not in use
- `update_job_dimension(job_id, dim, op)` - Update tracking state

**API Endpoint**: `GET /api/job-builder/dimension-status`
```json
{
  "384": {"busy": false, "job_id": null, "operation": null, "source_id": null},
  "768": {"busy": true, "job_id": "abc123", "operation": "index", "source_id": "wiki"},
  "1024": {"busy": false, "job_id": null, "operation": null, "source_id": null},
  "1536": {"busy": true, "job_id": "def456", "operation": "delete", "source_id": "bitcoin"}
}
```

**Jobs with Dimension Tracking**:
- `index_online` - tracks 1536 dimension with "index" operation
- `index_offline` - tracks selected dimension (384/768/1024) with "index" operation
- `clear_vectors` - tracks selected dimension with "delete" operation

**Key Insight**: Different dimensions use separate ChromaDB instances (`chroma_db_384/`, `chroma_db_768/`, etc.), so operations on different dimensions can run in true parallel without lock contention.

### Future: Delete Optimization

ChromaDB delete operations are blocking and can freeze the app for large sources. Current implementation at `offline_tools/vectordb/store.py:682`:

```python
self.collection.delete(where={"source": source_id})
```

This is already the optimized "filter-based delete" (ChromaDB handles finding + deleting internally), but it still:
- Scans the entire index to find matching documents
- Loads segments into memory
- Blocks until complete (no async API)

**Potential Optimization Options:**

1. **Batched Deletes with Yields** (quickest win)
   - Query IDs in batches of 1000
   - Delete batch
   - `time.sleep(0)` to yield GIL
   - Repeat
   - Lets web server respond between batches

2. **Subprocess Isolation**
   - Run delete in separate process via `multiprocessing`
   - Main app stays responsive
   - Risk: ChromaDB file locking conflicts

3. **Per-Source Collections** (architectural change)
   - Each source gets its own collection: `articles_bitcoin`, `articles_wiki_top_100`
   - Delete = `client.delete_collection(name)` (nearly instant)
   - Major refactor but makes deletes trivial

4. **Copy-and-Swap (rebuild strategy)**
   - Instead of deleting source X:
     1. Create temp collection
     2. Copy all docs WHERE source != X
     3. Drop old collection, rename temp
   - Avoids delete operation entirely but complex

**References:**
- [ChromaDB Delete Docs](https://docs.trychroma.com/docs/collections/delete-data)
- [ChromaDB Rebuilding Guide](https://cookbook.chromadb.dev/strategies/rebuilding/)

### Future: Parallel Processing

The checkpoint system is prepared for future parallel processing with:
- `worker_id`: Identifies which worker (0 = single worker default)
- `total_workers`: Number of parallel workers
- `work_range_start/end`: Article range for this worker

Future implementation would:
1. Divide articles into ranges (worker 0: 0-25000, worker 1: 25001-50000, etc.)
2. Each worker has its own checkpoint and partial file
3. A merger step combines all partial files when all workers complete

---

## Related Documentation

- [Source Tools](source-tools.md) - Source creation pipeline
- [Validation System](validation.md) - Source validation after jobs complete
- [Admin Guide](admin-guide.md) - Using the admin panel

---

*Last Updated: December 2025*
