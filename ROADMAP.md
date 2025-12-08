# Disaster Clippy - Roadmap

Future plans and feature development priorities.

---

## Current Status: v0.9 (Pre-release)

### Completed Features

**Core Functionality:**
- Conversational search with 1000+ documents
- Vector embeddings (OpenAI or local sentence-transformers)
- Source attribution with clickable links
- Document classification (Guide, Article, Research, Product)
- External API for embedding on other sites
- Metadata index for fast sync

**Admin Panel:**
- FastAPI admin dashboard at /useradmin/
- 5-step Source Tools wizard
- Source validation with status boxes (Config, Backup, Metadata, Embeddings, License)
- Install/download cloud source packs
- Auto-discovery of indexed sources
- License compliance tracking

**Content Tools:**
- HTML backup system for offline browsing
- PDF ingestion with intelligent chunking
- Substack scraper with paid content support
- Web scrapers (MediaWiki, Fandom, static sites)
- ZIM file indexing

**Infrastructure:**
- Unified codebase (merged private/public repos)
- CLI tools for local admin, ingestion, and sync
- Cloudflare R2 cloud storage integration
- Pinecone sync functionality
- Cloud sync (Pinecone)

**Refactoring Completed (Dec 2025):**
- Folder reorganization (`admin/`, `offline_tools/`, `cli/`)
- Scrapers ported from private repo to `offline_tools/scraper/`
- CLI tools consolidated into `cli/` folder
- Route refactoring (extracted API into modular route files)
- Removed all legacy/backwards compatibility code
- Deleted empty folders and duplicate configs

---

## In Progress

### Deployment Mode Gating

Unified VECTOR_DB_MODE controls all access levels.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- Single `VECTOR_DB_MODE` environment variable with 3 modes: local, pinecone, global
- `local` = Admin UI, local ChromaDB, R2 read/submissions write
- `pinecone` = No admin UI (public mode), cloud search only
- `global` = Admin UI, Pinecone R/W, full R2 access
- `require_global_admin()` dependency blocks cloud write endpoints
- `/api/admin-available` endpoint for frontend feature detection
- Protected endpoints: `/api/upload-backup`, `/api/pinecone-push`, `/api/pinecone-namespace`

---

### Schema Standardization

Unified file naming and schema version across all tools.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- All file creation uses `schemas.py` getter functions
- Removed "v3 schema" references - current schema is the only schema
- Legacy file patterns documented in DEVELOPER.md for cleanup
- TESTING_CHECKLIST.md created for pipeline validation

---

### ZIM Metadata Extraction

Auto-populate source metadata from ZIM header_fields.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- ZIMIndexer extracts metadata from header_fields (title, description, license, creator, language, tags)
- Auto-populates _manifest.json with ZIM metadata during indexing
- User edits preserved (existing values not overwritten on re-index)
- Stores ZIM-specific fields: language, publisher, zim_date

---

### Tag Taxonomy Expansion

Expanded topic keywords for better automatic tag suggestions.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- Expanded from 8 to 28 tag categories in TOPIC_KEYWORDS
- Categories: water, sanitation, solar, energy, wind, biogas, fuel, food, agriculture, livestock, aquaculture, foraging, shelter, construction, tools, medical, herbal, mental-health, nutrition, emergency, fire, earthquake, flood, hurricane, nuclear, pandemic, navigation, communication, security, knots, appropriate-tech, electronics, vehicles, reference, how-to
- Increased suggestion limit from 5 to 10 tags
- Helps categorize content for cross-source search

---

### Language Filtering for Multi-Language ZIMs

Filter articles by language during metadata generation to deduplicate multi-language content.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- Language filter dropdown in Source Tools Step 2 (Generate Metadata)
- 39 languages supported across 6 regional groups:
  - Common: English, Spanish, French, Portuguese, Arabic, Chinese
  - Humanitarian Priority: Haitian Creole, Swahili, Bengali, Nepali, Urdu, Hindi, Tagalog, Indonesian
  - European: German, Italian, Dutch, Polish, Ukrainian, Russian, Romanian, Greek, Turkish
  - Asian: Japanese, Korean, Vietnamese, Thai, Malay, Burmese, Khmer, Lao
  - South Asian: Tamil, Telugu, Sinhala
  - Middle East/Africa: Persian/Farsi, Hebrew, Amharic
- Detection via URL patterns (/en/, /es/), title suffixes ((Spanish), (Chinese)), and separators (Title - Vietnamese)
- Filter applied at metadata generation so metadata and index stay in sync
- Eliminates duplicate search results from same content in multiple languages

---

### Source Filtering in Chat

Allow users to filter search by specific sources.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- "Select Sources" dropdown in chat interface header
- Checkboxes for each indexed source with document counts
- "Select All" and "Select None" quick actions
- Selection persisted to localStorage (`clippy_selected_sources`)
- API support via `sources` parameter in chat requests
- ChromaDB filter using `{"source": {"$in": [...]}}`

---

### Chat UX Improvements

Better user experience for chat interactions.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- Links in chat responses and article sidebar now open in new tabs (preserves chat history)
- ZIM article links: `target="_blank"` with zim-link class
- External URLs: `target="_blank"` with `rel="noopener noreferrer"`
- Markdown link parsing for AI responses

---

### Pipeline Testing

Validate the 5-step wizard and file creation tools work correctly.

**Status:** In Progress

**Test Areas:**
- Fresh source creation (HTML backup)
- ZIM file indexing
- PDF collection workflow
- Source rename
- Cleanup redundant files
- Cloud download

---

## Near Term

### Job Checkpoint/Resume System

Resume interrupted long-running jobs (metadata generation, indexing) instead of restarting from scratch.

**Status:** IMPLEMENTED (Generate Metadata + Create Index for ZIM)

**Implemented Features:**
- Checkpoint infrastructure in `job_manager.py` (Checkpoint class, save/load/delete functions)
- Generate Metadata saves checkpoints every 60 seconds OR every 2000 articles
- Create Index (ZIM) saves checkpoints every 60 seconds OR every 500 documents
- Full document backup in partial file (not just position)
- Resume modal in Source Tools: "Incomplete job found - Resume / Start Fresh / Cancel"
- Interrupted Jobs section on Jobs page with Resume/Discard buttons
- Manual "Cleanup Old Checkpoints (7+ days)" button
- Prepared for future parallel processing with worker_id fields

**Checkpoint Storage:**
```
BACKUP_PATH/_jobs/
  {source_id}_{job_type}.checkpoint.json    # Checkpoint state
  {source_id}_{job_type}.partial.json       # Full partial work backup
```

**Checkpoint by Job Type:**

| Job Type | Checkpoint Data | Status |
|----------|----------------|--------|
| Generate Metadata | last_article_index, documents | IMPLEMENTED |
| Create Index (ZIM) | indexed_doc_ids, last_article_index | IMPLEMENTED |
| Create Index (HTML/PDF) | N/A | Uses Incremental Indexing |
| Upload to Cloud | N/A | NOT NEEDED - atomic file operations |
| Download from Cloud | N/A | ALREADY WORKS - smart skip by file size |

**Analysis Results:**

1. **Download from Cloud** already resumes naturally via smart file skipping:
   ```python
   if local_path.exists() and local_size == remote_size:
       skipped.append(filename)  # Already downloaded
   ```

2. **Upload to Cloud** uses atomic per-file operations. ZIM = single file, HTML = zip+upload.
   No partial state to track.

3. **Create Index** currently does all-or-nothing (extract all, embed all, add all).
   The solution is **Incremental Indexing** (see below) not checkpointing.

**Files Changed:**
- `admin/job_manager.py` - Added Checkpoint class and checkpoint functions
- `offline_tools/source_manager.py` - Updated `_generate_zim_metadata()` with checkpointing
- `offline_tools/indexer.py` - Updated `ZIMIndexer.index()` with checkpointing
- `admin/routes/source_tools.py` - Added checkpoint API endpoints, resume parameter
- `admin/templates/source_tools.html` - Added resume modal
- `admin/templates/jobs.html` - Added Interrupted Jobs section with Resume/Discard buttons

**Future: Parallel Processing**
The checkpoint system is prepared for parallel processing:
- `worker_id`, `total_workers`, `work_range_start/end` fields ready
- Multiple workers would each process a range of articles
- Merger step would combine partial files when all workers complete

---

### Incremental Indexing (Create Index Resume)

**Status:** IMPLEMENTED

All indexers (ZIM, HTML, PDF) now use incremental indexing via `VectorStore.add_documents_incremental()`.

**How it works:**
1. Query ChromaDB for existing doc IDs for this source
2. Filter out already-indexed documents
3. Process remaining docs in batches of 100
4. For each batch: compute embeddings, add to ChromaDB (auto-persists)
5. Repeat until done

**Resume behavior:**
- If interrupted, re-run Create Index
- Already-indexed documents are automatically skipped
- Only new documents are processed
- No checkpoint files needed - ChromaDB is the checkpoint

**Files changed:**
- `offline_tools/vectordb/store.py` - Added `get_source_document_ids()` and `add_documents_incremental()`
- `offline_tools/indexer.py` - Updated ZIMIndexer, HTMLBackupIndexer, PDFIndexer to use incremental mode

**Bug fixes included:**
- Fixed `delete_source` endpoint using wrong field name (`source_id` vs `source`)

**Benefits over checkpointing:**
- Simpler implementation (no checkpoint files to manage)
- ChromaDB handles persistence naturally
- Resume is automatic (query existing IDs)
- Works even if server crashes hard
- Deleting a source still works (queries by `source` metadata field)

---

### PDF Collection System

Structured ingestion and management for PDF documents.

**Status:** Partial (indexer ready, needs serving routes)

**What exists:**
- `PDFIndexer` class in [offline_tools/indexer.py](offline_tools/indexer.py) - extracts text, chunks long docs
- Metadata extraction (title, author from PDF properties)
- Text extraction via PyMuPDF or pypdf

**What's missing:**
- `/pdf/{source_id}/{filename}` route to serve PDFs locally (browser handles rendering)
- R2 upload for cloud distribution
- Dashboard UI for PDF ingestion

**URL handling:**
- `local_url`: `/pdf/{source_id}/doc.pdf` - needs serving route (not yet implemented)
- `url`: `https://r2.../sources/{source_id}/doc.pdf` - for cloud access (PDFs often have no original online location)

**Features (planned):**
- Collection-based organization (group PDFs by topic/author)
- Two-level metadata (collection + document)
- DOI detection and CrossRef citation lookup
- License classification (public domain, open access, restricted)
- R2 hosting for public domain PDFs
- Smart chunking with section header detection

**Input Methods:**
- Single PDF file
- Folder of PDFs
- ZIP archive
- URL download

**Collection Structure:**
```
pdf_inbox/
  flu_preparedness/
    _collection.json    # Collection metadata
    FluSCIM_Guide.pdf
    CDC_Plan.pdf
  uncategorized/        # Default landing spot
```

**CLI Commands:**
```bash
python cli/ingest.py pdf add <file_or_folder> --collection <name>
python cli/ingest.py pdf list
python cli/ingest.py pdf create-collection <name> --license <type>
```

---

### Search Result Diversity

Balance results across sources so large collections don't dominate.

**Status:** Planning

**Problem:** Large sources (e.g., 135 PDF chunks) dominate search results, drowning out smaller but potentially more relevant sources.

**Solution:**
1. Source-aware re-ranking with max results per source
2. Title/exact match boosting
3. Fetch more results, then diversify

**Implementation:**
```python
def diversify_results(articles: List[dict], max_per_source: int = 2) -> List[dict]:
    """Re-rank results to ensure diversity across sources."""
    by_source = {}
    diversified = []
    for article in articles:
        source = article.get("metadata", {}).get("source", "unknown")
        if source not in by_source:
            by_source[source] = 0
        if by_source[source] < max_per_source:
            diversified.append(article)
            by_source[source] += 1
    return diversified
```

---

### Export Formats

Support multiple export formats for different use cases.

**Status:** Planning

**Formats:**

| Format | Use Case | Add/Remove | Best For |
|--------|----------|------------|----------|
| Folder + manifest | Working copy | Easy | Active development |
| ZIP (.dcpack) | Distribution | Rebuild | Sharing collections |
| ZIM | Offline browsing | Rebuild | HTML website snapshots |

**ZIM Conversion (Future):**
- Convert HTML backups (builditsolar, solarcooking) to ZIM
- Requires zimwriterfs tool
- Best for stable, rarely-updated content
- Enables Kiwix offline browsing

---

### Knowledge Map Visualization

Interactive graph showing document relationships based on embedding similarity.

**Status:** Planning

**Features:**
- UMAP projection of embeddings to 2D
- Plotly scatter plot colored by source/doc_type
- Hover for document details
- Outlier detection (low neighbor similarity)
- Community detection for topic clusters

**Purpose:**
- Visualize source overlap
- Find topic gaps
- Identify misclassified content
- Detect redundancy

---

### HTML Scraper Pipeline

Integrate the existing HTML scraper into the admin dashboard for creating custom backups.

**Status:** Backend ready, needs dashboard UI

**What exists:**
- `HTMLBackupScraper` class in [offline_tools/backup/html.py](offline_tools/backup/html.py) - downloads pages to `pages/` folder
- `HTMLBackupIndexer` class in [offline_tools/indexer.py](offline_tools/indexer.py) - indexes HTML backups
- `backup_manifest.json` format for URL-to-filename mapping
- Metadata generation from HTML files

**What's missing:**
- Dashboard UI to configure and run scrapes (currently CLI only)
- Language filtering at scrape time (filter pages before downloading)
- License detection (scan for /license, /terms, Creative Commons badges)
- Better metadata extraction from site `<meta>` tags
- Progress tracking in Jobs page

**URL handling:**
- `local_url`: `/backup/{source_id}/{filename}` - served from backup folder
- `url`: Original site URL (e.g., `https://ready.gov/plan`) - for online access

**Comparison with ZIM pipeline:**
| Aspect | ZIM | HTML Scraper |
|--------|-----|--------------|
| Backup source | Single `.zim` file | `pages/` folder with HTML files |
| Language filter | Applied during indexing | Applied during scraping |
| Metadata | Embedded in ZIM header | Must extract from site or user input |
| Indexing | Same `save_all_outputs()` | Same `save_all_outputs()` |

---

### Multi-Dimension Local Search

Support mixed embedding dimensions in local ChromaDB for flexibility between downloaded packs and user-created sources.

**Status:** Planning

**Problem:**
- Downloaded source packs use OpenAI 1536-dim embeddings (high quality)
- Local indexing on low-power devices (RPi 5) needs smaller 384-dim models
- ChromaDB collections are fixed-dimension - can't mix in same collection
- Users want both: downloaded packs AND their own local sources

**Solution: Per-Source Collections**
```
ChromaDB
  +-- collection: "ready_gov_site" (1536-dim, downloaded)
  +-- collection: "wikipedia_climate" (1536-dim, downloaded)
  +-- collection: "my_local_pdfs" (384-dim, user-created)
  +-- collection: "neighborhood_guide" (384-dim, user-created)
```

**Search Flow:**
```
User query: "how to filter water"
    |
    v
Group sources by dimension:
  - 1536-dim: ready_gov, wikipedia (need OpenAI or skip if offline)
  - 384-dim: my_local_pdfs, neighborhood_guide (local model)
    |
    v
Embed query once per unique dimension
    |
    v
Search all collections in parallel
    |
    v
Merge results by score, return top N
```

**Manifest Tracking:**
```json
{
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dimensions": 384,
  "embedding_mode": "local"
}
```

**Implementation:**
1. ChromaDB store: Create collection per source_id (not one global)
2. EmbeddingService: Cache models, support multiple dimensions
3. Search: Detect dimensions, embed query appropriately, merge results
4. Downloaded packs: Load `_vectors.json` into source-specific collection

**Hardware Considerations (RPi 5):**

| Model | Dimensions | Size | Speed |
|-------|------------|------|-------|
| all-MiniLM-L6-v2 | 384 | 22MB | ~50-100 docs/sec |
| all-mpnet-base-v2 | 768 | 420MB | ~10-20 docs/sec |
| OpenAI API | 1536 | N/A | Network-bound |

**Offline Mode:**
- If 1536-dim sources exist but no API key: skip those collections
- Show warning: "3 sources unavailable (requires internet)"
- Or: re-embed locally on first offline use (one-time cost)

---

### Government Scrapers

Add FEMA, Cal Fire, EPA sources.

**Status:** Partial (PDF ingestion ready, need specific scrapers)

**Targets:**
- FEMA preparedness guides (Public Domain)
- Cal Fire wildfire resources (Public Domain)
- EPA emergency response (Public Domain)
- CDC health emergency guides

**Approach:**
- PDF scraper for downloadable reports
- Static site scraper for HTML guides
- Auto-detect Public Domain license

---

### Language Packs

Downloadable translation packs for multi-language support without duplicating content.

**Status:** Planning (after indexing/chat functions stable)

**Problem:**
- Multi-language ZIMs contain the same content in 39+ languages
- Storing all translations = gigabytes of duplicated information
- Current approach: keep English canonical, filter other languages at index time

**Solution: Offline Translation Packs**
- Keep one canonical language (English) in the index
- Downloadable translation packs (~50MB each) for offline use
- Multi-lingual dictionaries for word/phrase lookup
- "Preferred language" selector in chat interface
- Translate on-demand from English to user's preferred language

**Pack Structure:**
```
language_packs/
  es_spanish.pack       # ~50MB Spanish translation data
  fr_french.pack        # ~50MB French translation data
  ar_arabic.pack        # ~50MB Arabic translation data
```

**User Experience:**
1. User downloads base content (English) + language pack
2. Selects preferred language in chat settings
3. Search happens in English (canonical)
4. Results displayed/translated in preferred language
5. Works fully offline with downloaded pack

**Benefits:**
- Storage: 1 copy of content + small language packs vs N copies in N languages
- Search quality: Single consistent index instead of fragmented language indexes
- Distribution: Smaller base download, optional language packs
- Offline: Full translation capability without internet

**UI Placeholders:**
- Source pages already have language pack download placeholders
- Chat settings will add "Preferred Language" dropdown

**Note:** Prioritizing offline translation over online browser translation tools since the core use case is disaster preparedness without internet access.

---

## Medium Term (v2.0)

### Location-Aware Search

Prioritize locally-relevant content based on user location.

**Features:**
- Location metadata on documents (state, county, region)
- Extract location from user queries
- Boost location-specific results
- Display location badges in results

**Implementation:**
- Add location fields to ScrapedPage
- NER or regex for location extraction
- Location filtering in vector search
- Session-based location storage

---

### Distributed Job Processing

Offload long-running jobs (indexing, embeddings) to multiple machines for parallel processing.

**Status:** Planning

**Problem:**
- Large sources (450k+ articles) take hours to index on a single machine
- Embedding generation is CPU/GPU intensive
- Single-machine approach wastes available compute resources
- Users may have multiple machines, GPUs, or cloud resources available

**Solution: Coordinator/Worker Model**
- Coordinator tracks overall job progress and assigns work chunks
- Workers claim chunks, process them, and report back
- Each worker checkpoints independently (integrates with Job Checkpoint system)
- If a worker dies, coordinator reassigns its chunk to another worker

**Architecture:**
```
Coordinator (Main Machine)
    |
    +-- Tracks: job_id, source_id, total_articles, chunk assignments
    |
    +-- Chunk Status: pending, in_progress (worker_id), completed
    |
    v
Workers (Other Machines)
    |
    +-- Poll coordinator for available chunks
    +-- Process assigned chunk (e.g., articles 10000-19999)
    +-- Write partial results to shared storage or upload to coordinator
    +-- Report completion, request next chunk
```

**Chunk Assignment:**
```json
{
  "job_id": "abc123",
  "source_id": "wikipedia-medical",
  "total_articles": 450000,
  "chunk_size": 10000,
  "chunks": [
    {"chunk_id": 0, "start": 0, "end": 9999, "status": "completed"},
    {"chunk_id": 1, "start": 10000, "end": 19999, "status": "in_progress", "worker_id": "gpu-server"},
    {"chunk_id": 2, "start": 20000, "end": 29999, "status": "pending"}
  ]
}
```

**Implementation Phases:**

| Phase | Scope | Network |
|-------|-------|---------|
| 1. Local Multi-GPU | Multiple GPUs on same machine | N/A (process parallelism) |
| 2. LAN Workers | Other machines on local network | HTTP API, shared storage |
| 3. Cloud Workers | Remote cloud instances | HTTP API, S3/R2 upload |

**Phase 1: Local Multi-GPU**
- Detect available GPUs
- Spawn worker processes per GPU
- Each process handles a chunk range
- Merge results when all complete

**Phase 2: LAN Workers**
- Simple HTTP API for worker registration
- Workers poll for work: `GET /api/jobs/{job_id}/next-chunk`
- Workers report completion: `POST /api/jobs/{job_id}/chunks/{chunk_id}/complete`
- Shared network storage for partial results (or upload to coordinator)

**Phase 3: Cloud Workers (Future)**
- Workers authenticate with API key
- Results uploaded to R2/S3
- Coordinator aggregates from cloud storage
- Cost-aware scheduling (spot instances)

**Worker Setup:**
```bash
# On worker machine (LAN example)
python cli/worker.py --coordinator http://192.168.1.10:8000 --gpu 0
```

**Coordinator API Endpoints:**
```
GET  /api/jobs/{job_id}/status          # Overall job progress
GET  /api/jobs/{job_id}/next-chunk      # Claim next available chunk
POST /api/jobs/{job_id}/chunks/{id}/complete  # Mark chunk done, upload results
POST /api/jobs/{job_id}/chunks/{id}/heartbeat # Worker still alive
```

**Failure Handling:**
- Workers send heartbeat every 30 seconds
- No heartbeat for 2 minutes = chunk returned to pending
- Chunk retry limit (3) before marking failed
- Failed chunks can be manually retried or skipped

**Integration with Checkpoint System:**
- Each worker maintains its own checkpoint file
- Coordinator checkpoint tracks chunk assignments
- On coordinator restart, reload chunk status, reassign orphaned chunks

**Test Plan:**
1. Test local multi-GPU on 6-GPU machine
2. Test LAN worker on local network (two machines)
3. Test cloud worker with remote instance

---

### Personal Cloud Backup

Self-hosted backup system for users with their own hosting infrastructure.

**Status:** Planning (placeholder in Settings page)

**Problem:**
- "Global" mode requires central Pinecone/R2 access
- "Local" mode is single-machine only
- Users with own servers/VPS want a middle ground
- Want to sync between devices without using project's cloud

**Solution: Personal Cloud Mode**
- User provides their own S3-compatible storage endpoint
- User optionally provides their own vector DB endpoint
- System syncs to user's infrastructure instead of project's cloud

**Settings Page Fields:**
```
Personal Cloud Backup
---------------------
Storage Type: [S3-Compatible v]
Endpoint URL: [https://my-minio.example.com]
Bucket Name: [disaster-clippy]
Access Key: [********]
Secret Key: [********]
[Test Connection] [Save]

Optional: Vector Database
-------------------------
Type: [Pinecone v]
API Key: [********]
Environment: [us-east-1]
[Test Connection] [Save]
```

**Sync Behavior:**
- Manual sync button (not automatic)
- Uploads: _manifest.json, _metadata.json, _vectors.json, backups
- Downloads: Pull sources from personal cloud to new device
- Conflict resolution: Newer timestamp wins (with confirmation)

**Use Cases:**
- Family/community shared knowledge base
- Prepper groups with dedicated server
- Organizations with internal hosting requirements
- Developers testing without affecting production cloud

---

### ZIM as Foundation Layer

Use Kiwix ZIM format as the primary backup/distribution format.

**Why ZIM:**
- Single file distribution (not thousands of HTML files)
- Built-in full-text search
- Works in Kiwix readers on all platforms
- Wikipedia uses it (proven at scale)
- Easy to torrent/IPFS distribute

**Layer Architecture:**
```
Layer 3: AI/RAG (Optional, requires API keys)
Layer 2: Vector Index (Optional, semantic search)
Layer 1: Kiwix/ZIM (Foundation, always works offline)
```

**User Entry Points:**
- Emergency user: Just the ZIM, browse offline
- Power user: ZIM + local vectors
- Full experience: ZIM + vectors + AI

**Distribution:**
```
disaster-clippy-core.zim     # 500MB - Essential survival
disaster-clippy-medical.zim  # 200MB - Medical deep dive
disaster-clippy-solar.zim    # 150MB - Solar/energy
```

---

### Offline AI Assistant

Provide AI capabilities without internet.

**Option A: Pre-trained Small Model (2-4 GB)**
- Phi-3, Llama-3.2, or Mistral 7B quantized
- Runs on CPU (8GB RAM minimum)
- Fine-tuned system prompt for disaster prep domain
- Via Ollama or llama.cpp

**Option B: Cached Response Database (50-200 MB)**
- Pre-computed answers to 10,000+ common questions
- Fuzzy matching to find similar questions
- No inference needed, instant responses
- Works on any device including phones

**Option C: Hybrid (Recommended)**
- Check cached answers first (instant)
- If no match + local model: run inference
- If no match + no model: show relevant ZIM pages
- Always offer "browse in Kiwix" fallback

**Hardware Tiers:**

| Tier | RAM | Capability |
|------|-----|------------|
| Phone/RPi | 512MB-2GB | Cached answers + ZIM browse |
| Old Laptop | 4-8GB | + Phi-3 Mini (slow) |
| Modern Laptop | 8-16GB | + Llama-3.2/Mistral-7B |
| Desktop GPU | 16GB+ | + Larger models, fast |

**Offline Download Packages:**
- Minimal (500 MB): ZIM only, browse and search
- Standard (700 MB): + cached AI answers
- Full (3 GB): + local LLM model
- Power User (5 GB): + larger model, dev tools

---

### Unified AI Pipeline (IMPLEMENTED Dec 2025)

Unified search and response generation across online/offline modes.

**Status:** IMPLEMENTED

**Architecture:**
- `admin/ai_service.py` - Unified AIService class
- `admin/connection_manager.py` - Smart connectivity detection
- Streaming support via SSE endpoints

**Search Pipeline:**

| Mode | Method | Fallback |
|------|--------|----------|
| Online | Semantic (embedding API) | Error |
| Hybrid | Semantic | Keyword search |
| Offline | Keyword search | N/A |

**Response Pipeline:**

| Mode | Method | Fallback |
|------|--------|----------|
| Online | Cloud LLM (OpenAI/Claude) | Error with message |
| Hybrid | Cloud LLM | Local Ollama | Simple response |
| Offline | Local Ollama | Simple response |

**Smart Ping Logic:**
- 5-minute ping interval (configurable)
- Reset timer on successful API call
- Immediate ping on API failure
- No pinging in offline_only mode (user chose offline)
- Ping on page focus after being away

**Mode Behaviors:**

| Mode | Tries Online | Fallback | Pings | Use Case |
|------|-------------|----------|-------|----------|
| Online | Always | Error message | Yes (warn on loss) | Normal use |
| Hybrid | First | Yes, to local | Yes (detect recovery) | Unreliable internet |
| Offline | Never | N/A | No | Known offline |

**Streaming Endpoints:**
- `POST /api/v1/chat/stream` - SSE streaming for real-time response
- `GET /api/v1/connection-status` - Get current connection status
- `POST /api/v1/ping` - Trigger connectivity check

**Future Improvements (Offline AI Update):**
- Local embedding models (sentence-transformers) for offline semantic search
- Portable Ollama bundled with app
- Response caching for common questions
- Better offline keyword search (synonyms, stemming)

---

## Long Term (v3.0+)

### Source Packs & Marketplace

Enable users to create, share, and install curated content packs.

**Concept:**
A "Source Pack" is a curated, shareable bundle containing indexed content that users can install into their own Clippy instance.

**Pack Contents:**
```
solar-cooking-pack/
  manifest.json        # Pack metadata, version, author
  sources.json         # Source definitions, scraper configs, licenses
  vectors.parquet      # Pre-computed embeddings (optional)
  metadata/            # Document metadata (titles, URLs, hashes)
```

**Quality Tiers:**

| Tier | Requirements | Visibility |
|------|--------------|------------|
| Personal | Any state, messy OK | Private only |
| Community | 80%+ indexed, licenses reported | Public, peer reviewed |
| Official | 100% indexed, verified licenses, backups complete | Featured, maintainer reviewed |

**Pack Health Checklist:**
- All sources have verified licenses
- No license conflicts
- 100% of pages indexed successfully
- HTML/ZIM backups complete for offline use
- No broken source URLs
- Metadata complete

**Download Options (Hybrid Model):**
- Quick Start: Vectors only (small, fast queries)
- Standard: Vectors + metadata (can re-scrape)
- Full Offline: Vectors + ZIM backups (works like Kiwix)
- Source Configs Only: Just scraper configs (user runs scrapers)

---

### Multi-User Platform

Transform from single-user admin to multi-user platform.

**User Journey:**

```
Stage 1: Anonymous Browser
  - Uses main app, queries all official sources
  - No account needed

Stage 2: Registered User
  - Creates account
  - Customizes source selection (checkboxes)
  - Can upload private docs
  - Preferences saved to profile

Stage 3: Power User
  - Exports config + downloads packs
  - Sets up local offline system
  - Can sync preferences with main app

Stage 4: Self-Hoster
  - Full clone of system
  - Own Pinecone + own backups
  - Can contribute back to official
```

**Source Selection UI:**
```
My Source Preferences:
  [x] Solar Cooking (176 docs)
  [x] Medical Emergency (500 docs)
  [ ] Homesteading (1200 docs)
  [x] My Uploads (23 docs)

  [Save] [Export for Offline]
```

---

### Account Tiers

**Free Tier:**
- Query all official sources
- Save preferences
- 10 private doc uploads
- Rate limited
- Export: vectors only

**Pro Tier ($5/mo):**
- Unlimited queries
- 100 private uploads
- Priority queue
- Export: full backups included
- Community pack submissions

**Self-Hosted (free, DIY):**
- Everything runs on their hardware
- No ongoing cost to project
- Can sync preferences with main app
- Contributor status

---

## Version Targets

| Version | Focus | Key Features |
|---------|-------|--------------|
| v0.5 | Unified codebase | Merged repos, admin dashboard, scrapers (COMPLETE) |
| v0.75 | Production polish | Mode gating, schema updates, testing (COMPLETE) |
| v0.9 | Pre-release | Final testing, documentation cleanup (CURRENT) |
| v1.0 | Official Release | Stable API, ZIM distribution, source packs |
| v2.0 | Platform | User accounts, marketplace, federated queries |

---

## Technical Debt

### Known Bugs

**EMBEDDING_MODE=local not respected**
- Location: `offline_tools/embeddings.py`
- Issue: EmbeddingService always tries to initialize OpenAI client even when EMBEDDING_MODE=local
- Impact: Crashes without OPENAI_API_KEY even if user wants local embeddings
- Fix: Check EMBEDDING_MODE before initializing OpenAI client

### Fixed Bugs (Dec 2025)

**Pinecone sync "Pushed: 0 documents" despite finding documents to push**
- Location: `offline_tools/source_manager.py`, `offline_tools/packager.py`
- Issue: Document IDs in `_metadata.json` didn't match IDs in ChromaDB
- Cause: Metadata used `zim_0, zim_1...` format, indexer used `md5(source_id:url)` hash format
- Fix: Updated metadata generation to use same hash ID format as indexer
- Recovery: Regenerate metadata for affected sources, then retry Pinecone sync

**Progress count showing "0 / 100 items" instead of actual counts**
- Location: `offline_tools/indexer.py`
- Issue: Progress callback received percentages (0-100) instead of actual item counts
- Fix: Changed progress reporting to pass actual `(current, total, message)` values

**Token limit errors for long articles - got zero vectors instead of embeddings**
- Location: `offline_tools/embeddings.py`
- Issue: Articles exceeding 8192 tokens got zero vectors and wouldn't appear in search
- Fix: Added `_embed_with_chunking()` - splits long text in half, embeds both, averages result
- Recursively splits up to 8 chunks for very long articles

### Improvements (Dec 2025)

**Search result diversity**
- Location: `app.py` - `ensure_source_diversity()`
- Change: Search now retrieves 15 candidates, applies source diversity (max 2 per source), returns 5
- Benefit: Prevents single source from dominating results when multiple sources are relevant

**Minimum content length filter increased**
- Location: `source_manager.py`, `zim_utils.py`
- Change: Increased from 50 to 100 characters
- Benefit: Filters out stub pages like "0.1.0" or single-word redirects

**URL samples pagination in Source Tools**
- Location: `source_tools.html`
- Change: URL sample tables now show 5 at a time with "Show More" button
- Benefit: Cleaner UI when sources have 20+ sample articles

### Testing

**Automated Tests (TODO):**
- Unit tests for scrapers (mock HTTP)
- Integration tests for search flow
- Load tests for concurrent sessions

**Manual Testing Checklist:**

Before release, verify the following workflows:

| Test | Steps | Verify |
|------|-------|--------|
| Fresh HTML Source | Source Tools -> Create Backup -> Create Index | All 5 status boxes green, search works |
| ZIM Indexing | Place .zim in backup path -> Source Tools -> Create Index | Schema files created, search returns ZIM content |
| ZIM Language Filter | Step 3 -> Select language -> Force Re-index | Only target language articles indexed |
| PDF Collection | Add PDFs to folder -> Create Index | PDFs chunked and searchable |
| Source Rename | Select source -> Rename | Folder renamed, ChromaDB updated, old folder deleted |
| Cloud Download | Sources -> Cloud tab -> Download | Files downloaded, ChromaDB populated |
| Source Filtering | Chat -> Select Sources dropdown | Only selected sources return results |
| Link Behavior | Click article links in chat | Opens in new tab, chat history preserved |
| Tag Suggestions | Source Tools Step 4 -> Get Suggestions | Tags suggested based on content |
| Railway Proxy | Set RAILWAY_PROXY_URL, no R2 keys | Cloud sources accessible via proxy |
| Public Mode | Set VECTOR_DB_MODE=pinecone | Admin UI blocked, chat works |

**Schema File Verification:**

Each source should have these files with `schema_version: 3`:
- `_manifest.json` - Source identity and config
- `_metadata.json` - Document list
- `_index.json` - Full content
- `_vectors.json` - Embeddings
- `backup_manifest.json` - URL to file mapping

**Common Issues:**
- Status box red but files exist: Check filename matches schema (e.g., `_metadata.json` not `{source_id}_metadata.json`)
- Search returns no results: Verify ChromaDB has vectors, check source_id matches
- "Source not found": Verify `_manifest.json` exists in source folder

### Code Quality
- Type hints (Python 3.10+)
- Docstrings (Google style)
- Linting (ruff)
- CI/CD with GitHub Actions

### Documentation
- API documentation (OpenAPI/Swagger)
- Deployment runbook
- Scraper development guide
- Pack creation guide

---

## Cost Estimates

**Project Infrastructure:**
- Pinecone Serverless: ~$3-10/mo (1-10M vectors)
- Cloudflare R2: ~$1-5/mo (100GB, free egress)
- Railway: ~$5-20/mo (backend hosting)
- Total: ~$10-35/mo at moderate scale

**User Costs (self-hosted):**
- Pinecone free tier: 100K vectors ($0)
- Local storage: their drives ($0)
- Optional cloud backup: ~$5/mo

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Quick start and project overview |
| [DEVELOPER.md](DEVELOPER.md) | Technical details, CLI tools, security |
| [SUMMARY.md](SUMMARY.md) | Executive summary (non-technical) |
| [CONTEXT.md](CONTEXT.md) | Architecture and design decisions |

---

*Last Updated: December 2025*
