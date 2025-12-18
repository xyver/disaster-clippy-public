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
- Source validation with 6 status boxes (Config, Backup, Metadata, 1536, 768, License)
- Validation gates: can_submit (local admin) and can_publish (global admin)
- Human verification flags (license_verified, links_verified_offline, links_verified_online)
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

### Validation System

Unified validation architecture with permission gates and caching.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- Two permission gates: `can_submit` (local admin) and `can_publish` (global admin)
- Two validation tiers: Light (<100ms) and Deep (5-30 sec for integrity checks)
- Human verification flags: license_verified, links_verified_offline, links_verified_online
- Validation caching in `_validation_status.json` with mtime-based invalidation
- ALLOWED_LICENSES list (12 licenses including Custom with notes)
- 6 status boxes in UI (Config, Backup, Metadata, 1536, 768, License)

**Key Files:**
- `offline_tools/validation.py` - Core validation module
- `docs/validation.md` - Full specification

---

### Job Builder UI

Visual job chain builder for custom combined jobs.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- Visual UI at `/useradmin/job-builder`
- Click to add jobs to chain
- Drag to reorder
- Parameter configuration per job
- Chain validation (warns about ordering issues)
- Checkpoint support for resumable chains
- Save/load chain templates (localStorage)

**Combined Job Framework:**
- `JobPhase` dataclass for defining phases
- `run_combined_job()` for executing chains
- Predefined templates: generate_source, regenerate_source, reindex_online, reindex_offline, cloud_publish

**Key Files:**
- `admin/job_schemas.py` - Job parameter schemas
- `admin/routes/job_builder.py` - API endpoints
- `admin/templates/job_builder.html` - UI

---

### Connection State Management

Smart connectivity detection with visual feedback.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- 6 connection states: online, checking, unstable, disconnected, offline, recovering
- Color-coded indicators (green, blue, yellow, red, gray)
- Unified `/api/v1/connection-status` endpoint
- Smart ping logic (5-min interval, reset on success, immediate on failure)
- No pinging in offline_only mode
- Consistent display across Dashboard, Settings, and Chat pages

**Key Files:**
- `admin/connection_manager.py` - State machine
- `docs/ai-service.md` - Full documentation

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

**Status:** CORE IMPLEMENTED (Dec 2025)

**Implemented (Dec 2025):**
- `PDFIndexer` class with page-aware extraction and chunking
- Page tracking: each chunk knows which pages it spans (page_start, page_end)
- 300-char chunk overlap for context preservation at boundaries
- Browser-navigable URLs using `#page=N` (works in Chrome, Firefox, Edge)
- PDF server at `/pdf/{source_id}/{filename}` for local serving
- Enhanced metadata extraction (title, author, page_count, file_size)
- PDF-specific fields in DocumentMetadata schema

**Key Files:**
- `offline_tools/indexer.py` - PDFIndexer class with new methods:
  - `_extract_text_with_pages()` - preserves page boundaries
  - `_extract_enhanced_metadata()` - comprehensive PDF metadata
  - `_chunk_pages_with_overlap()` - 300-char overlap, page tracking
- `admin/pdf_server.py` - serves PDFs with browser page navigation
- `offline_tools/schemas.py` - PDF fields (parent_pdf, page_start, page_end, chunk_index, total_chunks, total_pages)

**URL Pattern:**
- `url`: `/pdf/{source_id}/filename.pdf#page=47` - browser jumps to page
- `local_url`: `file:///path/to/doc.pdf#page=47` - works for local files

**What's Still Missing:**
- R2 upload for cloud distribution
- Dashboard UI for PDF ingestion
- Collection-based organization UI

**Features (planned):**
- Collection-based organization (group PDFs by topic/author)
- Two-level metadata (collection + document)
- DOI detection and CrossRef citation lookup
- License classification (public domain, open access, restricted)
- R2 hosting for public domain PDFs

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

### OCR for Scanned PDFs

Many historical and archival PDFs are image-only scans without a text layer.

**Status:** PLANNED (High Priority)

**The Problem:**
- Book scans, historical documents, government archives often have no text layer
- `PDFIndexer` extracts 0 characters from these files
- Valuable content is invisible to search

**Recommended Tools:**

| Tool | Description | Platform |
|------|-------------|----------|
| **OCRmyPDF** | Adds text layer to PDFs (preserves original) | Python, CLI |
| **Tesseract** | Core OCR engine (used by OCRmyPDF) | C++, many bindings |
| **pdftoppm** (poppler-utils) | Converts PDF pages to images for OCR | CLI |

**Proposed Approach: Preprocessing Pipeline**

OCR should be a separate preprocessing step, not part of indexing:

```
1. Detect: Check if PDF has text layer (char_count < threshold)
2. Convert: pdftoppm -> page images (PNG/TIFF)
3. OCR: Tesseract -> text per page
4. Embed: OCRmyPDF -> creates searchable PDF with text layer
5. Index: PDFIndexer processes the OCR'd PDF normally
```

**Why Separate Preprocessing:**
- OCR is slow (30-60 sec per page) - shouldn't block indexing
- Can run in batch overnight for large collections
- Preserves original PDF (OCRmyPDF creates new file)
- Can use GPU acceleration (Tesseract with CUDA)
- Results can be reviewed/corrected before indexing

**Installation:**
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr ocrmypdf poppler-utils

# macOS
brew install tesseract ocrmypdf poppler

# Windows (via chocolatey or manual install)
choco install tesseract poppler
pip install ocrmypdf
```

**OCRmyPDF Usage:**
```bash
# Basic usage - adds text layer to PDF
ocrmypdf input.pdf output_searchable.pdf

# Skip if already has text
ocrmypdf --skip-text input.pdf output.pdf

# Force re-OCR (replace existing text layer)
ocrmypdf --force-ocr input.pdf output.pdf

# Optimize for size after OCR
ocrmypdf --optimize 3 input.pdf output.pdf
```

**Future Implementation Files:**
- `cli/ocr_preprocess.py` - CLI tool for batch OCR
- `offline_tools/ocr.py` - OCR detection and processing functions
- Integration with Source Tools (detect + offer OCR option)

**Language Support:**
Tesseract supports 100+ languages via language packs:
```bash
# Install additional languages
sudo apt install tesseract-ocr-spa tesseract-ocr-fra tesseract-ocr-deu

# Use specific language
ocrmypdf -l spa input.pdf output.pdf

# Multiple languages
ocrmypdf -l eng+spa input.pdf output.pdf
```

**Quality Considerations:**
- Scanned quality affects accuracy (300 DPI minimum recommended)
- Preprocessing (deskew, denoise) improves results
- Historical fonts may need custom training data
- Consider confidence scores for questionable OCR

**See:** [docs/document-type-weighting.md](document-type-weighting.md) (OCR PDFs section)

---

### Search Result Diversity

Balance results across sources so large collections don't dominate.

**Status:** ONGOING (Dec 2025) - Core implemented, tuning weights

**Problem:** Large sources (e.g., 135 PDF chunks) dominate search results, drowning out smaller but potentially more relevant sources.

**Implemented:**
- `ensure_source_diversity()` in app.py
- Retrieves 15 candidates, applies max 2 per source, returns 5
- Round-robin selection by score
- Doc type prioritization (guides over articles)

**Still Exploring:**
- Different weighting strategies
- Title/exact match boosting
- Query intent detection for dynamic weights
- Per-source quality scores

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

Interactive 3D visualization of document embeddings for admin content curation.

**Status:** IMPLEMENTED (Dec 2025)

**Implemented Features:**
- PCA projection of embeddings to 3D coordinates
- Plotly.js interactive 3D scatter plot
- Color-coded by source with legend
- Hover for document details (title, source)
- Click to open article
- Filter by source checkboxes
- Edge building from internal links in `_metadata.json`
- Per-source lazy loading (URLs and edges loaded on demand)
- Regenerate button to recompute
- Publish to R2 for public access

**Access:** `/useradmin/visualise` (admin-only)

**Key Files:**
- `admin/routes/visualise.py` - Backend API and generation
- `admin/templates/visualise.html` - Plotly 3D frontend

**Future Enhancements (not implemented):**
- UMAP algorithm option (better cluster preservation, requires `umap-learn`)
- Density scoring for duplicate detection (nearest neighbor distances)
- Graph layout algorithms (igraph + Leiden community detection)
- Outlier highlighting (low neighbor similarity)

---

### HTML Scraper Pipeline

Integrate the existing HTML scraper into the admin dashboard for creating custom backups.

**Status:** IMPLEMENTED (Dec 2025)

**Implemented:**
- `HTMLBackupScraper` class in [offline_tools/backup/html.py](offline_tools/backup/html.py)
- `HTMLBackupIndexer` class in [offline_tools/indexer.py](offline_tools/indexer.py)
- Integrated into Source Tools wizard (Step 1)
- Configurable: page limit, max depth, delay, follow links, include assets
- URL discovery: XML sitemap, sitemap index, HTML sitemap pages, link following
- Breadth-first crawling for broad coverage
- Resume support (checks backup_manifest.json for existing pages)
- Progress tracking in Jobs page

**Possible Enhancements:**
- Language filtering at scrape time (filter pages before downloading)
- License detection (scan for /license, /terms, Creative Commons badges)
- Better metadata extraction from site `<meta>` tags

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

### Dual Embedding Architecture

Standardized dual-dimension system for online/offline semantic search.

**Status:** IMPLEMENTED (Dec 2025)

**See DEVELOPER.md "Offline Architecture" section for full details.**

**The Decision:**

| Context | Dimension | Model | Owner |
|---------|-----------|-------|-------|
| Online (Pinecone) | 1536 | OpenAI text-embedding-3-small | Global Admin |
| Offline (ChromaDB) | 768 | all-mpnet-base-v2 | Global Admin |

**Key Points:**
- Global Admin creates BOTH 768 and 1536 embeddings
- Users download 768-dim only (smaller, works offline)
- Local admins can use any dimension locally, re-embedded on submission
- True offline semantic search (not keyword fallback)

**File Structure:**
```
{source_id}/
    _vectors_768.json    # Offline (downloaded by users)
    _vectors_1536.json   # Online (Pinecone + backup)
```

### User Tier System

Four-tier system supporting different hardware and use cases.

**Status:** IMPLEMENTED - See DEVELOPER.md "Offline Architecture" section

| Tier | Hardware | Role | Primary Actions |
|------|----------|------|-----------------|
| Consumer | RPi5 / Field | End user | Download, search, browse |
| Local Admin | Laptop 8-16GB | Creator | Create sources, submit |
| Global Admin | Desktop + API | Curator | Review, standardize, publish |
| Super Powered | Cloud/GPU | Processing | Mass indexing, parallel API |

**Consumer Tier (RPi5):**
- Downloads pre-embedded 768-dim packs
- Query embedding: ~200-500ms (feasible on RPi5)
- Optional LLM: Llama 3.2 3B Q4 (~15-20 sec responses)
- Search Mode vs Conversation Mode toggle

**Processing Time Reality:**

| Source Size | Local Admin (768) | Global Admin (API) | Super Powered |
|-------------|-------------------|-------------------|---------------|
| 10,000 docs | ~2 hours | ~15 min | ~3 min |
| 100,000 docs | ~20 hours | ~2 hours | ~20 min |
| 450,000 docs | 4+ days | ~10 hours | ~2 hours |

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

**Status:** PHASE 1 COMPLETE (Dec 2025)

**See [docs/language-packs.md](docs/language-packs.md) for full implementation details.**

**Phase 1 Complete - Article Translation:**
- LanguageRegistry with 8 priority languages (MarianMT models)
- TranslationService with article caching
- API endpoints for download/install/set-active
- Languages tab in admin panel (Sources page)
- ZIM viewer auto-translation with visual indicator badge
- Translations cached for instant repeat visits

**How It Works Now:**
1. User downloads a language pack from Languages tab (~300MB each)
2. User clicks "Set Active" on the installed pack
3. User browses any ZIM article - it's automatically translated
4. Green badge shows "Translated to [Language]" in corner
5. Translations are cached for instant repeat visits

**8 Priority Languages:**
- Spanish, French, Arabic, Chinese (Simplified)
- Portuguese, Hindi, Swahili, Haitian Creole

**Future Phases (not implemented):**

*Phase 2 - Chat Translation:*
- User types in their language, query translated to English for search
- LLM response translated back to user's language
- Requires `translate_to_english()` method in TranslationService

*Phase 3 - NLLB Universal Model:*
- Single ~2.4GB model supporting 200 languages
- Language dropdown for all online users
- Auto-detect source language
- Suitable for Railway deployment (server-side translation)

*Phase 4 - Source Localization (Best Experience):*
- Local Admin pre-translates entire sources at index time
- Creates `source_es/` variant with translated metadata
- Re-embeds translated text (Spanish embeddings for Spanish queries)
- Zero runtime translation - embeddings match query language natively
- Multilingual LLMs (Aya 23, Suzume-Llama-3) for native responses
- Storage doubles per language (local only, not on global R2)

**Key Files:**
- `offline_tools/translation.py` - TranslationService + TranslationCache
- `offline_tools/language_registry.py` - 8 language packs metadata
- `admin/routes/models.py` - Language API endpoints
- `admin/templates/sources.html` - Languages tab UI
- `admin/zim_server.py` - Auto-translation integration

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

**Status:** IMPLEMENTED (Dec 2025)

**Implemented Features:**
- Provider dropdown in Settings: Cloudflare R2, AWS S3, Backblaze B2, DigitalOcean Spaces, Custom S3-Compatible
- Auto-fill endpoint URL based on provider selection
- Credential storage in `local_settings.json` (gitignored)
- "Test Connection" button validates before saving
- Masked credentials in UI (only last 4 chars of access key shown)

**Supported Providers:**

| Provider | Endpoint Template | Typical Cost |
|----------|------------------|--------------|
| Cloudflare R2 | `https://ACCOUNT-ID.r2.cloudflarestorage.com` | $0.015/GB, no egress |
| AWS S3 | `https://s3.amazonaws.com` | $0.023/GB + egress |
| Backblaze B2 | `https://s3.us-west-002.backblazeb2.com` | $0.005/GB |
| DigitalOcean Spaces | `https://nyc3.digitaloceanspaces.com` | $5/mo for 250GB |
| Custom/MinIO | User-defined | Varies |

**Use Cases:**
- Family/community shared knowledge base
- Prepper groups with dedicated server
- Organizations with internal hosting requirements
- Developers testing without affecting production cloud

**See:** [docs/deployment.md](docs/deployment.md) for full configuration details

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

**Status:** IMPLEMENTED (Dec 2025) - See DEVELOPER.md "Offline Architecture" section

**Recommended for RPi5 (Consumer Tier):**
- Embedding: all-mpnet-base-v2 (768-dim, ~500MB RAM)
- LLM: Llama 3.2 3B Q4 (~2GB RAM, 8-15 tokens/sec)
- Total RAM needed: ~5GB of 8GB available

**Search Mode vs Conversation Mode:**

| Mode | Response Time | Use Case |
|------|---------------|----------|
| Search Mode | 1-2 sec | Quick lookup, just show docs |
| Conversation Mode | 15-20 sec | LLM synthesizes answer |

**User Experience:**
- First query: +15-30 sec for model loading
- Subsequent queries: ~15-20 seconds
- Streaming responses to feel faster

**Hardware Tiers (Updated):**

| Tier | RAM | Capability |
|------|-----|------------|
| RPi5 Consumer | 4-8GB | 768-dim search + optional 3B LLM |
| Local Admin | 8-16GB | Full source tools + 7B LLM |
| Global Admin | 16GB+ | Dual embedding + larger models |

**Consumer Optimizations:**
- Pre-load models on startup
- Cache recent query embeddings
- Memory-mapped ChromaDB
- Pre-bundled models (no download wait)

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

## Search Pipeline Architecture

Understanding how queries become results - useful for improving search quality.

### Pipeline Flow

```
USER QUERY: "what are solar projects I can build at home?"
                              |
                              v
+------------------------------------------------------------------+
|  1. EMBEDDING GENERATION (ai_service.py)                         |
|     - Query sent to OpenAI text-embedding-3-small                |
|     - Returns 1536-dimensional vector                            |
|     - Vector captures semantic meaning of query                  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  2. VECTOR SEARCH (pinecone_store.py / store.py)                 |
|     - Compare query vector to all stored document vectors        |
|     - Cosine similarity scoring (0-1, higher = more similar)     |
|     - Returns top 15 closest matches                             |
|     - Pre-filtered by source if user selected specific sources   |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  3. DOC TYPE PRIORITIZATION (app.py)                             |
|     - Boost guides/how-to content over reference articles        |
|     - Detect user intent (wants products? tutorials? research?)  |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  4. SOURCE DIVERSITY (app.py - ensure_source_diversity)          |
|     - Max 2 results per source                                   |
|     - Round-robin selection by score                             |
|     - Returns final 5 results                                    |
+------------------------------------------------------------------+
                              |
                              v
                      FINAL 5 RESULTS
```

### What Gets Embedded (Step 4 - Create Index)

Each document's embedding is generated from:
- `title` - article title
- `content` - first ~8000 characters of page text

**Location:** `offline_tools/packager.py` - `_prepare_document_for_embedding()`

### Metadata Stored (Step 3 - Generate Metadata)

Stored alongside embeddings but NOT included in vector search:
- `title` - for display
- `source` - source identifier (e.g., "builditsolar2")
- `url` - online URL for attribution
- `local_url` - offline browsing path
- `categories` - tags from URL path and meta tags
- `doc_type` - classification (article, guide, product)

**Location:** `offline_tools/source_manager.py` - metadata generation functions

### Improving Search Quality

**Option 1: Include Tags in Embedding (Recommended)**
Modify `_prepare_document_for_embedding()` to append categories:
```python
text_to_embed = f"{title}\n\nCategories: {', '.join(categories)}\n\n{content}"
```
This makes tag matches influence semantic similarity.

**Option 2: Hybrid Search**
Combine semantic results with keyword matching on tags/title.
Post-filter or re-rank based on exact keyword matches.

**Option 3: Metadata Boosting**
After vector search, boost scores for results matching query keywords in metadata.

### Key Files

| File | Role in Pipeline |
|------|------------------|
| `admin/ai_service.py` | Orchestrates search, selects method |
| `offline_tools/vectordb/pinecone_store.py` | Cloud vector search |
| `offline_tools/vectordb/store.py` | Local ChromaDB search |
| `offline_tools/packager.py` | Creates embeddings from content |
| `offline_tools/source_manager.py` | Generates metadata (step 3) |
| `app.py` | Doc type prioritization, source diversity |

### Conversation Context

The chat maintains conversation history and article references for follow-up queries.

**History Management (app.py):**
- Line 641: `history = session["history"][-20:]` - keeps last 20 messages (10 exchanges)
- Lines 647-648: Appends user message and AI response to history
- Line 596: Session initialization with `"last_results": []`

**Article References (app.py):**
- Line 605: Detects "more like", "similar to", "like #" phrases
- Line 607: `handle_similarity_query(message, session["last_results"])`
- Line 635: Stores current search results in `session["last_results"]`
- Lines 2011-2021: `handle_similarity_query()` - finds similar articles by ID

**Supported Phrases:**
- "tell me more about #2"
- "more like the first one"
- "similar to the solar heating article"
- Follow-up questions referencing previous context

**To Modify:**
- Increase history: Change `[-20:]` to `[-40:]` for 20 exchanges
- Add more trigger phrases: Edit line 605's phrase list
- Change similarity logic: Edit `handle_similarity_query()` at line 2011

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
| [DEVELOPER.md](DEVELOPER.md) | Setup guide and documentation index |
| [SUMMARY.md](SUMMARY.md) | Executive summary (non-technical) |
| [CONTEXT.md](CONTEXT.md) | Architecture and design decisions |
| [docs/architecture.md](docs/architecture.md) | Modes, security, data flow, offline architecture |
| [docs/source-tools.md](docs/source-tools.md) | SourceManager, indexers, scrapers, tags |
| [docs/ai-service.md](docs/ai-service.md) | Search, chat, connection modes |
| [docs/validation.md](docs/validation.md) | Permission gates, validation tiers |
| [docs/jobs.md](docs/jobs.md) | Background jobs, checkpoints, job builder |
| [docs/deployment.md](docs/deployment.md) | Deployment scenarios, cloud backup |
| [docs/admin-guide.md](docs/admin-guide.md) | Admin panel, CLI tools, troubleshooting |
| [docs/language-packs.md](docs/language-packs.md) | Offline translation system |

---

*Last Updated: December 13, 2025*
