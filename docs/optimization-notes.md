# Optimization Notes - Source Management Pipeline

This document captures optimization discussions, architecture decisions, and implementation plans for the offline-first system.

**This is the primary reference document for the dual embedding architecture and tier system.**

---

## Table of Contents
1. [Pipeline Overview](#pipeline-overview)
2. [Task-by-Task Analysis](#task-by-task-analysis)
3. [OpenAI Embedding API Bottleneck](#openai-embedding-api-bottleneck)
4. [Local LLM Options](#local-llm-options)
5. [Validation Pipeline](#validation-pipeline)
6. [Embedding Check Optimization](#embedding-check-optimization)
7. [Dual Embedding Architecture](#dual-embedding-architecture)
8. [User Tier System](#user-tier-system)
9. [Dimension Standardization](#dimension-standardization)
10. [RPi5 Consumer Tier Analysis](#rpi5-consumer-tier-analysis)
11. [Processing Time Reality](#processing-time-reality)
12. [Execution Plan](#execution-plan)

---

## Pipeline Overview

| Task | Bottleneck | Embedding Checks |
|------|-----------|------------------|
| 1. Scraping websites | Network I/O, rate limiting | None |
| 2. Scan HTML backup | Disk I/O, CPU parsing | None |
| 3. Generate Metadata | LLM API calls ($$$) | None |
| 4. Create Index | OpenAI embedding API | Yes - O(existing_docs) |
| 4b. Force Reindex | Same as #4 | None (deletes first) |
| 5. Suggest Tags | LLM API calls | None |
| 6. Validation | Network (URL checks) | None |
| 7. Upload to R2 | Network upload | None |
| 7b. Upload to Pinecone | Pinecone API | Yes - filters zeros |

---

## Task-by-Task Analysis

### 1. Scraping Websites
- Inherently slow to avoid bot detection
- Rate limiting is intentional
- No optimization needed

### 2. Scan HTML Backup
- **Text-heavy sites**: CPU bound (parsing)
- **Image-heavy sites**: Fast (images skipped)
- **Video-heavy sites**: Very fast (videos skipped)

Performance scales with **number of HTML files** and **text density**, not total file size.

| Site Type | 100MB | 1GB |
|-----------|-------|-----|
| Text-heavy | ~1 min | ~10 min |
| Image-heavy | ~30 sec | ~3 min |
| Video-heavy | ~10 sec | ~1 min |

### 3. Generate Metadata (LLM)
- Currently uses external LLM API
- Could use local LLM (see Local LLM Options)
- Cost vs speed tradeoff

### 4. Create Index
- OpenAI embedding API is the bottleneck
- NEW: Self-healing checks for zero embeddings
- See [Embedding Check Optimization](#embedding-check-optimization)

### 4b. Force Reindex
- "Delete all + redo from scratch"
- No embedding checks (nothing to check)
- Use when you want a clean slate

### 5. Suggest Tags
- LLM-based tag suggestions
- Could use local LLM
- Quality varies by model size

### 6. Validation
**Current checks:**
- URLs are reachable
- Content isn't empty
- Basic quality checks

**TODO - Add these checks:**
- Verify JSON backups match ChromaDB entries
- Check for orphaned files (backup exists, not indexed)
- Check for missing backups (indexed, no backup)
- Verify metadata files are consistent
- Check for zero/null embeddings

### 7. Upload to R2
- Standard network upload
- No special concerns

### 7b. Upload to Pinecone
- Filters out zero vectors before upload
- Prevents Pinecone rejection errors
- Checkpoint/resume support for large uploads

---

## OpenAI Embedding API Bottleneck

### Token Limits

OpenAI's `text-embedding-3-small` model has a limit of **8,191 tokens** per request.
- Average English word = ~1.3 tokens
- Roughly ~6,000 words per request
- Long articles (10,000+ words) will exceed this limit

### Current Chunking Strategy

Located in `offline_tools/embeddings.py`:

```
Original Document (15,000 words)
           |
           v
    [Check length]
           |
    [Exceeds limit?]
           |
     YES   |   NO
           v
    [Split at sentence boundaries]
           |
           v
    [Chunk 1]  [Chunk 2]  [Chunk 3]
        |          |          |
        v          v          v
    [Embed]    [Embed]    [Embed]
        |          |          |
        v          v          v
    [1536-dim] [1536-dim] [1536-dim]
        \          |          /
         \         |         /
          v        v        v
         [Average all embeddings]
                   |
                   v
            [Final 1536-dim vector]
```

**Implementation details:**
- MAX_CHARS = 32,000 (conservative estimate)
- Splits recursively at sentence boundaries
- Each chunk embedded separately
- Final embedding = element-wise average of all chunk embeddings

### Approaches for Long Documents

| Approach | Pros | Cons |
|----------|------|------|
| **Averaging (current)** | Simple, single vector per doc | Loses nuance, dilutes meaning |
| **Store chunks separately** | Preserves detail, better search | More vectors, complex retrieval |
| **Truncate** | Fast, simple | Loses end content entirely |
| **Summarize first** | Captures key points | Adds LLM cost, may miss details |

### Why Averaging Works for This Use Case

For disaster preparedness content:
- Most articles have a clear central topic
- Users search for topics, not specific paragraphs
- Averaged embedding still captures the main subject
- Simpler retrieval (1 doc = 1 vector)

### When to Consider Chunk Storage

Better for:
- Very long technical manuals
- Multi-topic documents
- When users need to find specific sections
- When search precision is critical

Would require changes to:
- ChromaDB schema (parent_doc_id field)
- Search logic (group results by parent)
- UI (show which section matched)

---

## Local LLM Options

### For Embeddings (768 vs 1536 dimensions)

| Model | Dimensions | Quality | Speed | Cost |
|-------|-----------|---------|-------|------|
| OpenAI text-embedding-3-small | 1536 | Best | Fast | $$$ |
| sentence-transformers (local) | 768 | Good | Medium | Free |
| Ollama + nomic-embed | 768 | Good | Slow | Free |

**WARNING**: Cannot mix dimensions in same ChromaDB collection. Pick one and stick with it.

### For Text Generation (Metadata, Tags)

| Model | Quality | Speed | Hardware Needed |
|-------|---------|-------|-----------------|
| GPT-4 | Excellent | Slow | API only |
| GPT-3.5 | Good | Fast | API only |
| Llama 70B | Good | Slow | 48GB+ VRAM |
| Llama 13B | Decent | Medium | 16GB VRAM |
| Mistral 7B | Good | Fast | 8GB VRAM |
| Llama 7B | Basic | Fast | 8GB VRAM |

**Recommendations:**
- For quality: GPT-4 or Llama 70B
- For speed on limited hardware: Mistral 7B
- For disaster/technical content: 13B+ recommended (understands domain terms)

### Use Cases

| Task | Can Use Local LLM? | Recommended |
|------|-------------------|-------------|
| Generate Metadata | Yes | Mistral 7B+ |
| Suggest Tags | Yes | Mistral 7B+ |
| Embeddings | Yes (768d) | Keep OpenAI for compatibility |

---

## Validation Pipeline

### Current State
Basic validation exists but doesn't cover all cases.

### Proposed "Validate Backups" Function

```
1. Compare JSON backup files vs ChromaDB entries
   - List all JSON files in backups/
   - List all doc IDs in ChromaDB for source
   - Report: orphaned files, missing backups

2. Verify embedding integrity
   - Check for zero/null embeddings
   - Report count and affected documents

3. Metadata consistency
   - Verify _metadata.json matches actual documents
   - Check for stale/outdated metadata

4. Optional auto-fix
   - Delete orphaned entries
   - Re-index missing documents
   - Re-embed zero vectors
```

---

## Embedding Check Optimization

### Problem
When re-indexing, we now check if existing documents have valid embeddings.
Current implementation: O(n) individual DB calls for n existing documents.

### Solution: Hybrid Batch + Parallel Approach

#### Phase 1: Batch the embedding checks

Instead of:
```python
for doc_id in existing_ids:  # 13,000 calls
    embedding = collection.get(ids=[doc_id], include=["embeddings"])
```

Do:
```python
# Batch in chunks of 1000 to avoid OOM
batch_size = 1000
bad_ids = set()

for offset in range(0, len(existing_ids), batch_size):
    batch_ids = list(existing_ids)[offset:offset + batch_size]
    result = collection.get(ids=batch_ids, include=["embeddings"])

    for doc_id, embedding in zip(result["ids"], result["embeddings"]):
        if embedding is None or all(v == 0 for v in embedding):
            bad_ids.add(doc_id)
```

**Performance:**
- 13,000 docs = 13 batch calls instead of 13,000 individual calls
- Memory bounded at ~40MB per batch (1000 * 1536 floats * 4 bytes)

#### Phase 2: Parallelize with metadata extraction

```
Timeline (current - sequential):
|-- Extract metadata (30s) --|-- Check embeddings (60s) --|-- Index (varies) --|

Timeline (optimized - parallel):
|-- Extract metadata (30s) --|-- Index (varies) --|
|-- Check embeddings (in background thread) --|
         ^-- Results ready before indexing needs them
```

#### Implementation Plan

```python
import concurrent.futures

def add_documents_incremental(self, documents, source_id, ...):
    # Start embedding check in background immediately
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit embedding check as background task
        future = executor.submit(
            self._find_invalid_embeddings_batched,
            source_id
        )

        # Continue with metadata extraction (happens in parallel)
        existing_ids = self.get_source_document_ids(source_id)

        # ... prepare documents ...

        # Now get the results (blocks if not ready, but usually is)
        bad_embedding_ids = future.result()

        # Filter: include new docs AND docs with bad embeddings
        # ... rest of indexing logic ...

def _find_invalid_embeddings_batched(self, source_id, batch_size=1000):
    """Find all doc IDs with zero/null embeddings for a source."""
    bad_ids = set()

    # Get all IDs for source
    result = self.collection.get(
        where={"source": source_id},
        include=["embeddings"]
    )

    # Process in memory (already fetched)
    for doc_id, embedding in zip(result["ids"], result["embeddings"] or []):
        if embedding is None or all(v == 0 for v in embedding):
            bad_ids.add(doc_id)

    return bad_ids
```

#### Memory Considerations

| Source Size | Embeddings Memory | Safe? |
|-------------|------------------|-------|
| 1,000 docs | ~6 MB | Yes |
| 10,000 docs | ~60 MB | Yes |
| 50,000 docs | ~300 MB | Caution |
| 100,000 docs | ~600 MB | Use batching |

For very large sources (50k+), use chunked fetching:
```python
def _find_invalid_embeddings_chunked(self, source_id, chunk_size=5000):
    # Fetch in chunks to avoid OOM
    all_ids = list(self.get_source_document_ids(source_id))
    bad_ids = set()

    for i in range(0, len(all_ids), chunk_size):
        chunk_ids = all_ids[i:i + chunk_size]
        result = self.collection.get(ids=chunk_ids, include=["embeddings"])
        # ... check embeddings ...

    return bad_ids
```

---

## Dual Embedding Architecture

### Problem Discovery

Current offline search does NOT use semantic search. Investigation revealed:

1. User query comes in
2. `AIService.search()` checks connection mode
3. If offline: calls `store.search_offline()` (keyword search)
4. If online: calls `store.search()` (semantic search with OpenAI embedding)

The `search_offline()` method does simple text matching - no embeddings at all.
The stored 1536-dim vectors are **completely unused** in offline mode.

**Root cause:** To do semantic search, the query must be embedded in the SAME dimension
as the stored vectors. Offline mode has no API access to generate 1536-dim query embeddings.

### Solution: Dual Embedding Standard

Standardize on two embedding dimensions:

| Context | Dimension | Model | Purpose |
|---------|-----------|-------|---------|
| Online (Pinecone) | 1536 | OpenAI text-embedding-3-small | Cloud search |
| Offline (local ChromaDB) | 768 | sentence-transformers / Ollama | Local search |

Global system stores BOTH. Users download only the 768-dim version for offline use.

### Three User Perspectives

```
PUBLIC USER                    LOCAL ADMIN                   GLOBAL ADMIN
===========                    ===========                   ============

[Chat interface]               [768 world only]              [Both worlds]
      |                              |                             |
      v                              v                             v
[Pinecone 1536]               [Download 768]                [Generate both]
      |                              |                             |
Never sees                    [Local ChromaDB 768]          [Validate both exist]
embeddings                           |                             |
                              [Create local 768]            [Sync 1536 -> Pinecone]
                                     |                             |
                              [Submit to global]            [Sync both -> R2]
                                     |
                              (Global re-embeds             [768 ready for download]
                               to 1536 if needed)
```

**Public user:** Searches Pinecone (1536-dim). Never sees embeddings.

**Local admin:** Lives entirely in "768 world":
- Downloads source packs with 768-dim vectors
- Creates local content with 768-dim (local models)
- Never needs to think about 1536
- True semantic search works offline

**Global admin:** Manages both worlds:
- Creates BOTH 768 and 1536 embeddings during indexing
- Validates both vector files exist
- Syncs 1536 to Pinecone (for online search)
- Syncs both to R2 (768 for user downloads, 1536 as backup)

### File Structure Changes

**Current:**
```
{source_id}/
    _manifest.json
    _metadata.json
    _index.json
    _vectors.json        # Single file, 1536-dim
```

**New:**
```
{source_id}/
    _manifest.json
    _metadata.json
    _index.json
    _vectors_768.json    # Offline embeddings (downloaded by users)
    _vectors_1536.json   # Online embeddings (Pinecone + backup)
```

### Schema Analysis

The existing `VectorsFile` schema in `offline_tools/schemas.py` already supports this:

```python
@dataclass
class VectorsFile:
    schema_version: int = 3
    source_id: str = ""
    embedding_model: str = "text-embedding-3-small"  # Already tracked
    dimensions: int = 1536                            # Already tracked
    document_count: int = 0
    created_at: str = ""
    vectors: Dict[str, List[float]] = field(default_factory=dict)
```

**Changes needed in schemas.py:**

1. Add new file naming functions:
```python
def get_vectors_file_768() -> str:
    return "_vectors_768.json"

def get_vectors_file_1536() -> str:
    return "_vectors_1536.json"

# Keep old function for backward compatibility (defaults to 768 for local)
def get_vectors_file() -> str:
    return "_vectors_768.json"
```

2. Update `validate_source_files()` for global admin:
```python
# Add to result dict:
"has_vectors_768": False,
"has_vectors_1536": False,

# Check both files:
result["has_vectors_768"] = (path / get_vectors_file_768()).exists()
result["has_vectors_1536"] = (path / get_vectors_file_1536()).exists()

# Global admin validation requires BOTH
# Local admin validation requires only 768
```

### Search Flow After Implementation

**Online (unchanged):**
```
Query -> OpenAI 1536-dim -> Pinecone -> Results
```

**Offline (new - true semantic search):**
```
Query -> Local 768-dim model -> Local ChromaDB 768 -> Results
```

### Sync Button Changes (Global Admin)

| Button | Current Behavior | New Behavior |
|--------|-----------------|--------------|
| Sync to Pinecone | Uses `_vectors.json` | Uses `_vectors_1536.json` only |
| Sync to R2 | Uploads all files | Uploads both vector files |
| Validate | Checks single vector file | Checks BOTH vector files exist |

### Download Flow

```
User clicks "Download Pack"
           |
           v
    [API fetches from R2:]
      - _manifest.json
      - _metadata.json
      - _index.json
      - _vectors_768.json    <-- Only 768, NOT 1536
           |
           v
    [Import to local ChromaDB]
           |
           v
    [768-dim collection ready]
           |
           v
    [Offline semantic search works!]
```

### R2 Storage Structure

```
r2://disaster-clippy/
    sources/
        appropedia/
            _manifest.json
            _metadata.json
            _index.json
            _vectors_768.json     <-- Users download this
            _vectors_1536.json    <-- Backup only (not downloaded)
            pages.zip             <-- Optional HTML backup
```

### Submission Flow (User-Created Content)

When a local admin creates content and submits to global:

1. Local admin creates pack with 768-dim vectors (using local model)
2. Submits to global admin (submissions bucket)
3. Global admin reviews submission
4. Global admin sees: "768-dim only - needs 1536 embedding"
5. Global admin generates 1536-dim version with OpenAI
6. Global admin publishes: both files uploaded to R2, 1536 to Pinecone

### Benefits

1. **True offline semantic search** - Not just keyword matching
2. **Smaller downloads** - 768-dim is half the size of 1536-dim
3. **No API needed offline** - Local model generates query embeddings
4. **No dimension conflicts** - Each context has one standard
5. **Clean separation** - Local admins never see 1536-dim complexity

### Storage Size Comparison

| Docs | 768-dim | 1536-dim | Savings |
|------|---------|----------|---------|
| 1,000 | ~3 MB | ~6 MB | 50% |
| 10,000 | ~30 MB | ~60 MB | 50% |
| 91,000 | ~275 MB | ~550 MB | 50% |

---

## User Tier System

The system supports four distinct user tiers, each with different capabilities, hardware requirements, and responsibilities.

### Tier Overview

| Tier | Hardware | Role | Primary Actions |
|------|----------|------|-----------------|
| **Consumer** | RPi5 / Field device | End user | Download, search, browse |
| **Local Admin** | Laptop 8-16GB | Content creator | Experiment, create sources, submit |
| **Global Admin** | Desktop + API | Curator | Review, standardize, publish, maintain |
| **Super Powered** | Cloud/GPU farm | Heavy processing | Mass indexing, re-embedding, migrations |

### Tier 1: Consumer Offline (RPi5 / Field Device)

**Use case:** Download packs, ask questions, browse content offline

**Key insight:** Consumers never embed documents - only embed queries (fast, single text).

**Hardware requirements:**
- RPi5 8GB RAM (recommended) or 4GB (minimum)
- 32-64GB storage
- No GPU needed

**What they do:**
- Download pre-embedded source packs (768-dim)
- Search using local 768-dim model for query embedding
- Optionally use small LLM for response generation
- Browse ZIM/HTML backups offline

**What they DON'T do:**
- Create new sources
- Generate embeddings for documents
- Submit content

**Optimizations for this tier:**
- Pre-load embedding model on startup (eliminate cold start)
- Cache recent query embeddings
- Memory-mapped ChromaDB to reduce RAM
- Optional: pre-computed common question cache

### Tier 2: Local Admin (Laptop/Desktop)

**Use case:** Experiment with source tools, process small-medium sources, mix online/offline

**Hardware requirements:**
- 8-16GB RAM
- 100GB+ storage
- Optional: GPU for faster local embedding

**What they do:**
- Download source packs from cloud
- Create new sources from HTML/ZIM/PDF
- Generate embeddings (any dimension based on their hardware)
- Submit sources to global admin for review
- Mix of online and offline operation

**Capabilities:**
- ChromaDB (local vector store)
- R2 backups read access
- R2 submissions write access
- Local embedding (384 or 768 dim)
- Optional API access for 1536-dim

**Limits:**
- No Pinecone write access
- No R2 backups write access
- Recommended max source size: 10-20k docs (local embedding)

### Tier 3: Global Admin (Current Production)

**Use case:** Review submissions, standardize dimensions, publish to production, maintain global index

**Hardware:** Personal desktop/laptop + OpenAI API access

**Responsibilities:**
- Review submissions from local admins
- Re-embed to standardized dimensions (768 + 1536)
- Publish to Pinecone and R2
- Maintain quality and consistency
- Process new large sources

**Capabilities:**
- Full ChromaDB access
- Full Pinecone read/write
- Full R2 read/write (both buckets)
- OpenAI API for 1536-dim embeddings
- Local model for 768-dim embeddings

**Current workflow:**
```
Submission arrives (variable dimension)
    |
    v
[Review content, license, tags]
    |
    v
[Re-embed with OpenAI 1536] -----> Pinecone
    |
    v
[Re-embed with local 768] -------> R2 (for consumer download)
    |
    v
[Validate both exist]
    |
    v
[Publish]
```

**Current pain points:**
- 10+ hour processing for large sources
- No parallelization
- Manual babysitting required
- No cost visibility

### Tier 4: Super Powered (Future)

**Use case:** Process massive sources fast, parallel processing, cloud infrastructure

**Hardware:** Cloud instances, GPU clusters, multiple API keys

**What it adds over Global Admin:**

| Feature | Global Admin | Super Powered |
|---------|--------------|---------------|
| Parallelism | 1 thread | 5-20 parallel workers |
| API calls | Sequential | Batched + concurrent |
| Hardware | Single machine | Cloud instances / GPU |
| Rate limits | Default tier | Higher tier / multiple keys |
| Cost tracking | None | Real-time dashboard |
| Time for 100k docs | ~10 hours | ~30 min |

**Implementation phases:**
1. Parallel API calls (async, 5-10x speedup)
2. Batch embedding endpoint (2-3x fewer round trips)
3. Cost tracking dashboard
4. Distributed workers (multiple machines)

### Tier Capabilities Matrix

| Capability | Consumer | Local Admin | Global Admin | Super Powered |
|------------|----------|-------------|--------------|---------------|
| Download packs | Yes | Yes | N/A | N/A |
| Search (semantic) | Yes (768) | Yes (768) | Yes (both) | Yes (both) |
| Create sources | No | Yes | Yes | Yes |
| Embed documents | No | Yes (any dim) | Yes (768+1536) | Yes (parallel) |
| Submit to global | No | Yes | N/A | N/A |
| Pinecone write | No | No | Yes | Yes |
| R2 backups write | No | No | Yes | Yes |
| Re-embed submissions | No | No | Yes | Yes |
| Parallel processing | No | No | No | Yes |
| Cost tracking | N/A | N/A | No | Yes |

### Who Creates What

| Role | Creates 768? | Creates 1536? | Downloads 768? | Uses Pinecone? |
|------|--------------|---------------|----------------|----------------|
| Consumer | No | No | Yes | No |
| Local Admin | Optional (any dim) | Optional (if API) | Yes | No |
| Global Admin | Yes | Yes | N/A | Yes (write) |
| Super Powered | Yes (batch) | Yes (batch) | N/A | Yes (write) |

---

## Dimension Standardization

### The Decision

| Context | Dimension | Model | Owner |
|---------|-----------|-------|-------|
| **Online (Pinecone)** | 1536 | OpenAI text-embedding-3-small | Global Admin creates |
| **Offline (Local)** | 768 | all-mpnet-base-v2 / nomic-embed-text | Global Admin creates |

**Global Admin creates BOTH dimensions. Users download 768 only.**

### Why 768 for Offline?

1. **RPi5 compatible** - Can run mpnet model for query embedding
2. **Good quality** - Better than 384-dim alternatives
3. **Reasonable size** - Half the storage of 1536-dim
4. **Established model** - sentence-transformers well-supported

### Why Not 384?

While 384-dim (all-MiniLM-L6-v2) is faster and smaller:
- Noticeably lower search quality
- 768 is still feasible on RPi5 for query embedding
- Document embedding is done by Global Admin, not consumers

### Local Admin Flexibility

Local admins can use ANY dimension for their own work:
- 384-dim on constrained hardware
- 768-dim on standard hardware
- 1536-dim if they have API access

**On submission, Global Admin re-embeds to standardized 768 + 1536.**

### Manifest Embedding Fields

```json
{
    "source_id": "appropedia",
    "name": "Appropedia",
    "has_vectors_768": true,
    "has_vectors_1536": true,
    "vectors_768_model": "all-mpnet-base-v2",
    "vectors_768_size_bytes": 27500000,
    "vectors_1536_model": "text-embedding-3-small",
    "vectors_1536_size_bytes": 55000000,
    "document_count": 13522
}
```

### Tracking Variable Local Models

When local admins use different embedding models, track:

| Field | Purpose |
|-------|---------|
| `embedding_model` | Reproducibility - same model = same vectors |
| `embedding_dimensions` | Collection compatibility |
| `embedding_provider` | "sentence-transformers", "ollama", "openai" |
| `model_quality_tier` | "high" (768+), "medium" (384), for score weighting |

### Multi-Dimension Search (Local Admin)

If a local admin has mixed dimensions (downloaded 768 + created 384):

```
Query -> Detect dimensions in local collections
    |
    +-> 768-dim sources: embed with mpnet
    +-> 384-dim sources: embed with MiniLM
    |
    v
Search each, merge by rank -> Results
```

**Score merging options:**
1. Rank-based merge (interleave by rank, not score)
2. Normalized percentile within collection
3. Quality weighting (multiply 384 scores by 0.9)

---

## RPi5 Consumer Tier Analysis

### The Question

Can an RPi5 run semantic search with 768-dim indexes AND provide an intelligent LLM feel?

### Answer: Yes

**Two separate components needed:**

| Component | Purpose | Model | RAM | RPi5 Feasible? |
|-----------|---------|-------|-----|----------------|
| Embedding model | Query to 768-dim vector | all-mpnet-base-v2 | ~500MB | Yes |
| LLM | Generate intelligent response | Llama 3.2 3B Q4 | ~2GB | Yes, but slow |

### RPi5 8GB Memory Budget

```
OS + ChromaDB:        ~1.5GB
Embedding model:      ~0.5GB (mpnet 768-dim)
LLM (3B quantized):   ~2.0GB
Buffer:               ~1.0GB
---------------------------------
Total needed:         ~5GB of 8GB available
```

### LLM Options for RPi5

| Model | Parameters | RAM (Q4) | Tokens/sec | Quality |
|-------|------------|----------|------------|---------|
| TinyLlama | 1.1B | 0.7GB | 15-30 | Basic |
| Llama 3.2 | 1B | 1.0GB | 20-40 | Decent |
| Qwen 2.5 | 1.5B | 1.0GB | 15-25 | Good |
| **Llama 3.2** | **3B** | **2.0GB** | **8-15** | **Recommended** |
| Phi-3 Mini | 3.8B | 2.5GB | 5-12 | Good |

### Realistic User Experience

```
User types: "how do I purify water in an emergency?"
    |
    v
[Embed query to 768-dim] - 200-500ms
    |
    v
[Search ChromaDB] - 50-100ms
    |
    v
[Retrieve top 5 docs] - 50ms
    |
    v
[Generate response ~150 tokens @ 10 tok/sec] - 15 seconds
    |
    v
Total: ~15-20 seconds (subsequent queries)
First query: +15-30 sec for model loading
```

### Search Mode vs Conversation Mode

User choice to optimize experience:

| Mode | What Happens | Response Time | Use Case |
|------|--------------|---------------|----------|
| **Search Mode** | Semantic search, show docs only | 1-2 sec | Quick lookup, browsing |
| **Conversation Mode** | Search + LLM response | 15-20 sec | Detailed answers, synthesis |

**UI implementation:**
- Toggle in chat interface
- Default to Search Mode on RPi5
- Auto-detect hardware and suggest appropriate mode

### Consumer Tier Optimizations

| Optimization | Impact | Implementation |
|--------------|--------|----------------|
| Pre-load models on startup | Eliminates cold start | Load embedding + LLM at app start |
| Query embedding cache | Repeat questions instant | LRU cache of recent queries |
| Memory-mapped ChromaDB | Reduces RAM ~50% | Configure on init |
| Pre-bundled models | No download wait | Ship with installer |
| Streaming responses | Feels faster | Show tokens as generated |

### What Consumer Pack Contains

```
source_pack/
    _manifest.json         # Metadata including embedding info
    _vectors_768.json      # Pre-embedded (user never re-embeds)
    _index.json            # Full content for display
    content.zim            # Optional: browsable backup
```

---

## Processing Time Reality

### By Tier and Source Size

| Source Size | Consumer | Local Admin (768) | Global Admin (API) | Super Powered |
|-------------|----------|-------------------|-------------------|---------------|
| 1,000 docs | N/A | ~10 min | ~2 min | ~30 sec |
| 10,000 docs | N/A | ~2 hours | ~15 min | ~3 min |
| 50,000 docs | N/A | ~10 hours | ~1 hour | ~10 min |
| 100,000 docs | N/A | ~20 hours | ~2 hours | ~20 min |
| 450,000 docs | N/A | **4+ days** | ~10 hours | ~2 hours |

### Practical Limits by Tier

| Tier | Recommended Max | Beyond This |
|------|-----------------|-------------|
| Local Admin (local embedding) | 20,000 docs | Use API or split source |
| Global Admin (API embedding) | 100,000 docs | Use Super Powered |
| Super Powered | 1,000,000+ docs | Distributed processing |

### Cost Estimates (OpenAI text-embedding-3-small)

| Source Size | Tokens (est) | Cost | Super Powered Time |
|-------------|--------------|------|-------------------|
| 10,000 docs | 10M | ~$1 | 3 min |
| 100,000 docs | 100M | ~$10 | 20 min |
| 450,000 docs | 450M | ~$45 | 2 hours |
| 1,000,000 docs | 1B | ~$100 | 4 hours |

### Hardware Detection for Local Admin

```python
def detect_local_admin_tier():
    ram_gb = get_system_ram()

    if ram_gb < 8:
        return {
            "tier": "constrained",
            "max_source_size": 2000,
            "recommended_embedding": "384",
            "recommended_model": "all-MiniLM-L6-v2",
            "warning": "Limited RAM - consider smaller sources"
        }
    elif ram_gb < 16:
        return {
            "tier": "standard",
            "max_source_size": 10000,
            "recommended_embedding": "768",
            "recommended_model": "all-mpnet-base-v2"
        }
    else:
        return {
            "tier": "capable",
            "max_source_size": 50000,
            "recommended_embedding": "768",
            "note": "For larger sources, consider API embedding"
        }
```

---

## Execution Plan

### Phase 1: Schema Updates (Foundation)

**Priority: HIGH - Must do first**

1. Update `offline_tools/schemas.py`:
   - [ ] Add `get_vectors_file_768()` function
   - [ ] Add `get_vectors_file_1536()` function
   - [ ] Update `get_vectors_file()` to return 768 (backward compat)
   - [ ] Add `has_vectors_768` and `has_vectors_1536` to validation
   - [ ] Create `validate_source_files_global()` for global admin (requires both)
   - [ ] Create `validate_source_files_local()` for local admin (requires only 768)

2. Update `SourceManifest` dataclass:
   - [ ] Add `has_vectors_768: bool` field
   - [ ] Add `has_vectors_1536: bool` field
   - [ ] Add `vectors_768_size_bytes: int` field
   - [ ] Add `vectors_1536_size_bytes: int` field

### Phase 2: Global Admin Indexing

**Priority: HIGH - Creates the dual embeddings**

1. Update indexing pipeline to generate both:
   - [ ] Modify indexer to call OpenAI for 1536-dim
   - [ ] Modify indexer to call local model for 768-dim
   - [ ] Save to separate files: `_vectors_768.json`, `_vectors_1536.json`
   - [ ] Update manifest with both file sizes

2. Options for 768-dim generation:
   - Option A: Run sentence-transformers locally on global admin machine
   - Option B: Use Ollama with nomic-embed-text
   - Option C: Use a cloud 768-dim service if available

### Phase 3: Sync Updates

**Priority: HIGH - Gets data to right places**

1. Update Pinecone sync:
   - [ ] Change to read from `_vectors_1536.json` specifically
   - [ ] Add validation that 1536 file exists before sync

2. Update R2 sync:
   - [ ] Upload both vector files
   - [ ] Update manifest upload with new fields

3. Update validation UI:
   - [ ] Show status of both vector files
   - [ ] Warning if only one exists
   - [ ] "Generate missing embeddings" button

### Phase 4: Download API

**Priority: MEDIUM - Enables local admins**

1. Update download endpoint:
   - [ ] Fetch only `_vectors_768.json` (not 1536)
   - [ ] Update manifest to reflect downloaded version

2. Update local import:
   - [ ] Expect 768-dim vectors
   - [ ] Configure ChromaDB for 768-dim collection

### Phase 5: Local Search Updates

**Priority: MEDIUM - Enables offline semantic search**

1. Update `EmbeddingService`:
   - [ ] Default to 768-dim local model when `EMBEDDING_MODE=local`
   - [ ] Ensure `all-mpnet-base-v2` or similar 768-dim model

2. Update `VectorStore.search()`:
   - [ ] Use local embedding service for query embedding
   - [ ] Works offline with 768-dim vectors

3. Update `AIService.search()`:
   - [ ] When offline: use semantic search (not keyword fallback)
   - [ ] Query embedding via local 768-dim model

### Phase 6: Submission Flow

**Priority: LOW - Nice to have**

1. Submission validation:
   - [ ] Check dimensions of submitted vectors
   - [ ] Show warning: "768-dim only - will be re-embedded"

2. Global admin re-embedding:
   - [ ] "Generate 1536 embeddings" button for submissions
   - [ ] Uses same indexing pipeline

### Migration Plan

For existing sources with only `_vectors.json` (1536-dim):

1. Keep existing file as `_vectors_1536.json` (rename)
2. Generate `_vectors_768.json` from content using local model
3. Update manifest

Script needed: `migrate_to_dual_embeddings.py`

### Testing Checklist

- [ ] Global admin can generate both embedding files
- [ ] Pinecone sync uses 1536-dim file only
- [ ] R2 upload includes both files
- [ ] Download returns 768-dim file only
- [ ] Local import creates 768-dim ChromaDB collection
- [ ] Offline semantic search works with local model
- [ ] Submission shows dimension warning
- [ ] Migration script handles existing sources

---

## Summary of Planned Optimizations

### Completed
- [x] Self-healing embedding checks (implemented)
- [x] Zero vector filtering for Pinecone (implemented)

### Immediate (Low effort, high impact)
- [ ] Batch embedding checks (reduces 13k calls to 13 calls)
- [ ] Parallel embedding check during metadata extraction

### Major Initiative: Dual Embedding Architecture
- [ ] Phase 1: Schema updates (file naming, validation)
- [ ] Phase 2: Global admin dual indexing (768 + 1536)
- [ ] Phase 3: Sync updates (Pinecone uses 1536, R2 gets both)
- [ ] Phase 4: Download API (serves 768 only)
- [ ] Phase 5: Local search updates (true offline semantic search)
- [ ] Phase 6: Submission flow (dimension validation)
- [ ] Migration script for existing sources

### User Tier Implementation
- [ ] Consumer tier: Pre-bundled models, search/conversation mode toggle
- [ ] Local admin tier: Hardware detection, processing limits guidance
- [ ] Global admin tier: Dual embedding generation workflow
- [ ] Super Powered tier: Parallel API calls, cost tracking dashboard

### Medium-term
- [ ] Validate Backups function
- [ ] Local LLM option for metadata generation
- [ ] "Generate missing embeddings" button in admin UI
- [ ] RPi5 consumer package with bundled models

### Long-term
- [ ] Automatic backup verification on startup
- [ ] Dashboard showing data integrity status
- [ ] User-created content submission workflow
- [ ] Distributed worker system for Super Powered tier

---

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Offline dimension | 768 | Works on RPi5, good quality, reasonable size |
| Online dimension | 1536 | OpenAI standard, best quality |
| Who creates both | Global Admin | Centralized quality control |
| What users download | 768 only | Smaller, works offline |
| Local admin submissions | Any dimension | Flexibility, re-embedded on approval |
| Backward compat default | 768 | Offline-first philosophy |
| Consumer LLM | Llama 3.2 3B Q4 | Best quality/speed for RPi5 |
| Embedding model (local) | all-mpnet-base-v2 | 768-dim, good quality |

---

*Last updated: December 2025*