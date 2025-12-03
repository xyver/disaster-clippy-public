# Disaster Clippy - Project Context

**Purpose:** Complete context for AI assistants working on this codebase. Read this instead of reading all the code.

**Last Updated:** December 2, 2025
**Version:** 1.1

---

## Repository Structure

This project uses TWO repositories:

| Repository | Purpose | Who Uses It |
|------------|---------|-------------|
| disaster-clippy-public | Public code for local admins | Anyone |
| disaster-clippy (private) | Full code + scrapers + admin tools | Maintainer only |

**Public repo contains:**
- app.py (FastAPI chat - but /useradmin/ enabled for local use)
- useradmin/ (local admin panel)
- sourcepacks/, vectordb/, offline_tools/, storage/
- All documentation except DEVELOPER-PARENT.md

**Private repo adds:**
- admin/ (Streamlit global admin dashboard)
- scraper/ (API scrapers)
- ingest.py, sync.py (CLI tools)
- Pinecone write access, full R2 access

**Shared tools** (manually synced from private to public):
- sourcepacks/pack_tools.py
- offline_tools/*.py
- storage/r2.py
- vectordb/*.py

---

## What Is This Project?

Disaster Clippy is a RAG (Retrieval-Augmented Generation) system that provides evidence-based emergency preparedness guidance. Users ask questions in natural language and get answers synthesized from curated educational sources, with full source attribution.

**Example:**
- User: "How do I purify water in an emergency?"
- System: Searches 800+ documents, finds relevant guides, generates answer with links to sources

---

## Current State (v0.9)

### Working Features
- Conversational search via web UI and API
- 1,015+ documents from 6 sources
- Vector embeddings (OpenAI API or local sentence-transformers)
- Cloud database (Pinecone) with local sync
- Admin dashboard (Streamlit) for source management
- PDF ingestion with intelligent chunking
- Substack newsletter ingestion (CSV export)
- External API for embedding on other websites
- Auto-discovery of indexed sources in admin dashboard
- HTML backup system for offline archival

### Content Sources
| Source | Documents | Type |
|--------|-----------|------|
| Appropedia | 150 | Appropriate technology wiki (CC-BY-SA) |
| BuildItSolar | 337 | DIY solar projects |
| SolarCooking Wiki | 176 | Solar cooking guides |
| The Barracks (Substack) | 191 | Emergency preparedness newsletter |
| PDF Uploads | 61 | Pandemic preparedness (FluSCIM) |
| Bitcoin Docs | 100 | Reference documentation |

---

## User Tiers

| Tier | Access | Vector DB | R2 Storage | Dashboard |
|------|--------|-----------|------------|-----------|
| End User | Public website | None (uses cloud) | None | Chat only |
| Local Admin | Public GitHub | ChromaDB (local) | Read backups/, Write submissions/ | /useradmin/ |
| Global Admin | Private GitHub | Pinecone (write) | Full access | Streamlit admin/ |

**End Users** just chat on the Railway-hosted site.
**Local Admins** run their own instance, add personal sources, submit packs for review.
**Global Admin** (maintainer) curates sources, approves submissions, manages Pinecone.

---

## Data Architecture

Two-layer storage model:

| Layer | What | Where | Purpose |
|-------|------|-------|---------|
| Indexes | Vector embeddings | Pinecone (cloud) / ChromaDB (local) | Fast semantic search |
| Backups | Source files (ZIM, HTML, PDF) | R2 cloud / local disk | Archival, offline use |

**R2 Cloud Storage folders:**
- `backups/` - Approved source packs (global admin)
- `submissions/` - Pending review (local admin uploads)
- `metadata/` - JSON metadata for each source

**Pinecone** stores vectors for production search. Local users use ChromaDB instead.

---

## Architecture

```
User --> FastAPI (app.py) --> Vector Search --> LLM --> Response
                                   |
                              ChromaDB/Pinecone
                                   ^
                              Scrapers (ingest.py) [private repo only]
```

### Key Components

| Component | File(s) | Purpose |
|-----------|---------|---------|
| Backend | `app.py` | FastAPI server, chat endpoint, web UI |
| Ingestion | `ingest.py` | CLI for scraping and indexing |
| Sync | `sync.py` | CLI for database synchronization |
| Scrapers | `scraper/*.py` | Fetch content from sources |
| Vector DB | `vectordb/*.py` | Embeddings, storage, search |
| Admin | `admin/app.py` | Streamlit dashboard |

### Data Flow

1. **Ingestion Pipeline:**
   - Scraper fetches content from source
   - Content converted to ScrapedPage object
   - Embedding generated (OpenAI or local)
   - Stored in ChromaDB with metadata
   - Metadata index updated (JSON files)

2. **Query Pipeline:**
   - User query received
   - Query embedded
   - Semantic search finds top N documents
   - LLM generates response with context
   - Response returned with source links

3. **Sync Pipeline:**
   - Compare local metadata with remote
   - Identify new/changed documents
   - Push only what's needed to Pinecone

---

## Key Files and Their Roles

### Main Application
- `app.py` - FastAPI backend, all HTTP endpoints, LLM chain

### CLI Tools
- `ingest.py` - Unified ingestion CLI (all scrapers)
- `sync.py` - Database sync CLI (local <-> Pinecone)

### Scraper Module (`scraper/`)
- `__init__.py` - Factory: `get_scraper()`, `SCRAPER_REGISTRY`
- `base.py` - `BaseScraper`, `ScrapedPage`, `RateLimitMixin`
- `mediawiki.py` - Generic MediaWiki API scraper
- `appropedia.py` - Appropedia-specific preset
- `fandom.py` - Fandom wiki scraper
- `static_site.py` - HTML site scraper
- `pdf.py` - PDF document scraper with chunking
- `substack.py` - Substack newsletter scraper (CSV export)

### Vector Database Module (`vectordb/`)
- `__init__.py` - Exports: `VectorStore`, `get_vector_store`
- `store.py` - ChromaDB implementation
- `pinecone_store.py` - Pinecone cloud implementation
- `factory.py` - `get_vector_store()` based on env
- `embeddings.py` - `EmbeddingService` (OpenAI or local)
- `metadata.py` - `MetadataIndex` for fast sync
- `sync.py` - `SyncManager` for DB comparison

### Configuration
- `config/sources.json` - Source registry
- `config/ingest_config.json` - Ingestion job config
- `.env` - Environment variables (API keys, modes)

### Data
- `data/chroma/` - Local ChromaDB files
- `data/metadata/` - JSON metadata index
  - `_master.json` - Summary of all sources
  - `{source}.json` - Per-source document list

---

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...            # For embeddings

# LLM Selection
LLM_PROVIDER=openai              # or 'anthropic'
ANTHROPIC_API_KEY=sk-ant-...     # If using Claude

# Embedding Mode
EMBEDDING_MODE=openai            # or 'local' (free)

# Database Mode
VECTOR_DB_MODE=local             # or 'pinecone'
PINECONE_API_KEY=...             # If using Pinecone

# Optional
BACKUP_PATH=D:\disaster-backups  # For HTML/PDF backups
```

---

## Common Development Tasks

### Add new content source
```bash
python ingest.py scrape mediawiki --url https://wiki.example.com --search "topic"
python sync.py --remote pinecone push --commit
```

### Add new scraper type
1. Create `scraper/mysite.py` extending `BaseScraper`
2. Implement `get_page_list()` and `scrape_page()`
3. Register in `scraper/__init__.py`: `SCRAPER_REGISTRY["mysite"] = MyScraper`

### Run locally
```bash
python app.py              # Chat UI at localhost:8000
streamlit run admin/app.py # Admin at localhost:8501
```

### Sync to production
```bash
python sync.py --remote pinecone compare --verbose
python sync.py --remote pinecone push --commit
```

---

## Planned Features (from ROADMAP.md)

### Near Term
- **Custom DB Builder** - Admin selects sources, generates downloadable vector DB (like Kiwix)
- **Knowledge Map** - UMAP visualization of document embeddings in admin

### Medium Term
- **Source Packs Marketplace** - Curated, shareable content bundles with quality tiers
- **Location-Aware Search** - Prioritize locally-relevant content
- **Partner Namespaces** - Isolated DBs for organizations (counties, hospitals)
- **Government Scrapers** - FEMA, Cal Fire, EPA (Public Domain)

### Long Term
- **Two-Layer Architecture** - Pinecone indexes + Cloud storage (S3/R2) for ZIM backups
- **Multi-User Platform** - User accounts, personal sources, federated queries
- **Offline AI Assistant** - Tiered offline models (Phi-3 Mini to Llama 3.2)
- **ZIM Foundation Layer** - Kiwix-compatible offline archives

### Recently Completed
- **Substack Scraper** - Ingest newsletters from CSV export (thebarracks.substack.com)
- **Auto-discovery** - Dashboard auto-discovers indexed sources from metadata
- **License Compliance** - Dashboard shows license status for all sources

### Architecture Direction (v2.0+)

The long-term architecture separates into three layers:

```
Layer 3: AI/RAG (Optional) - Vector search + LLM, requires API keys
Layer 2: Vector Index (Optional) - ChromaDB/Pinecone semantic search
Layer 1: Kiwix/ZIM (Foundation) - Always works offline, zero setup
```

Key concepts:
- **Source Packs** - Curated bundles (vectors + ZIM backups) with quality tiers
- **Two-Layer Storage** - Indexes (small, Pinecone) vs Backups (large, S3/ZIM)
- **Offline AI** - Cached answers (instant) + optional local LLM (Phi-3/Llama)
- **Multi-User** - Source selection UI, personal namespaces, federated queries

See ROADMAP.md and DEVELOPER.md for full details.

---

## Code Patterns

### Scraper Factory
```python
from scraper import get_scraper
scraper = get_scraper("mediawiki", base_url="https://...")
pages = scraper.scrape_all(limit=100)
```

### Vector Store Factory
```python
from vectordb import get_vector_store
store = get_vector_store()  # Uses VECTOR_DB_MODE from .env
results = store.search("query", n_results=5)
```

### Adding Documents
```python
from vectordb import VectorStore
store = VectorStore()
store.add_documents([
    {"title": "...", "content": "...", "url": "...", "source": "..."}
])
```

### Error Handling (Admin)
```python
from admin.app import handle_errors

@handle_errors(default_return={}, error_prefix="Failed to load")
def load_something():
    ...
```

---

## Recent Changes (December 2025)

1. **Substack Scraper** - New scraper for Substack newsletters via CSV export
   - `scraper/substack.py` - Main scraper class
   - `offline_tools/substack_backup.py` - HTML backup handler
   - Successfully indexed 191 posts from thebarracks.substack.com

2. **Admin Dashboard Improvements**
   - Auto-discovery of indexed sources from metadata files
   - License Compliance section shows all sources including auto-discovered
   - Edit button available for all source types (online, offline, indexed)
   - Fixed NoneType error when ZIM source_id is null
   - Fixed inflated backup count (was using size estimation)

3. **Generalized ingest.py** - Now works with any scraper type via `python ingest.py scrape <type> ...`

4. **RateLimitMixin** - Consolidated rate limiting code in `scraper/base.py`

5. **Scraper Registry** - Factory pattern: `get_scraper()`, `SCRAPER_REGISTRY`

6. **Admin Error Handling** - Added `@handle_errors` decorator, `safe_execute()` helper

7. **Future Architecture Documented** - ROADMAP.md updated with Source Packs, Multi-User Platform, Offline AI plans

---

## Known Issues

1. **MD5 for hashing** - Using MD5 for content deduplication (works but not cryptographically secure)

2. **PDF memory** - Large PDFs loaded entirely into memory before chunking

3. **No tests** - Unit tests not yet implemented

4. **Hardcoded chunk size** - 4000 chars default, should be configurable per-source

---

## File Locations Summary

**PUBLIC REPO (disaster-clippy-public):**
```
|-- app.py                 # FastAPI backend (useradmin enabled)
|-- local_cli.py           # CLI for local admins
|-- useradmin/             # Local admin panel
|-- sourcepacks/           # Pack tools (shared)
|-- vectordb/              # Vector database layer (shared)
|-- offline_tools/         # Backup/indexing tools (shared)
|-- storage/               # R2 client (shared)
|-- config/sources.json    # Source registry
|-- templates/             # Web UI templates
|-- static/                # Frontend assets
```

**PRIVATE REPO (disaster-clippy) adds:**
```
|-- admin/app.py           # Streamlit global admin dashboard
|-- scraper/               # API scrapers (Appropedia, MediaWiki, etc.)
|-- ingest.py              # Scraping CLI
|-- sync.py                # Pinecone sync CLI
|-- DEVELOPER-PARENT.md    # Maintainer documentation
```

**Local data (gitignored):**
```
|-- data/chroma/           # Local ChromaDB
|-- data/metadata/         # Metadata index (JSON)
|-- .env                   # API keys
```

---

## How to Work on This Project

1. **Read this file first** - You now have full context

2. **Identify which repo you're in:**
   - Public repo = local admin features, shared tools
   - Private repo = global admin, scrapers, Pinecone access

3. **For local admin features** (public repo):
   - useradmin/ - Local admin panel
   - sourcepacks/pack_tools.py - Shared pack utilities
   - offline_tools/ - Backup and indexing tools

4. **For global admin features** (private repo only):
   - admin/app.py - Streamlit dashboard
   - scraper/ - API scrapers
   - ingest.py, sync.py - CLI tools

5. **For shared tools** - Edit in private repo, then manually copy to public repo:
   - sourcepacks/pack_tools.py
   - offline_tools/*.py
   - storage/r2.py
   - vectordb/*.py

6. **Test locally:**
   ```bash
   # Public repo (local admin)
   python app.py              # Chat + useradmin at localhost:8000

   # Private repo (global admin)
   python app.py              # Chat at localhost:8000 (useradmin disabled)
   streamlit run admin/app.py # Admin at localhost:8501
   ```

---

## Deployment

- **Live site:** https://disaster-clippy.up.railway.app/
- **Railway** deploys from private repo (disaster-clippy)
- Railway has /useradmin/ disabled - end users only see chat
- Public repo users run locally with /useradmin/ enabled

---

*This document should be updated when significant architecture changes occur.*

---

## Metadata Schema Redesign (December 2025)

This section documents the target metadata architecture. The current implementation is inconsistent and needs migration to this design.

### Design Goals

1. **Scalability** - Work for small sources (100 docs) and large sources (50,000+ docs)
2. **Portability** - Each source is a self-contained folder that can be moved/shared
3. **Diffing** - Enable fast comparison between scrapes without loading large files
4. **Incremental updates** - Only re-embed changed documents
5. **Merging** - Combine partial scrapes into unified sources

### Two Processing Pipelines

**Pipeline 1: Backup** (source-specific)
- Input: Raw source (ZIM file, PDF folder, website URL)
- Output: Backup files + backup metadata
- Creates: `backup_manifest.json` + actual files (pages/, *.zim, *.pdf)

**Pipeline 2: Index** (universal)
- Input: Backup folder (any type)
- Output: Embeddings + index metadata
- Creates: `_metadata.json` + `_index.json`

Both pipelines contribute to `_manifest.json` which describes the whole source.

---

### Source Types

| Type | Backup Format | Examples |
|------|---------------|----------|
| html | `pages/` folder with HTML files | BuildItSolar, SolarCooking, Appropedia |
| zim | Single `.zim` file | Bitcoin Wiki, Wikipedia subsets |
| pdf | PDF files in folder | Humanitarian Standards, FluSCIM guides |
| substack | `pages/` folder (like html) | The Barracks |

Future types: academic papers, government reports, zip archives

---

### Structure Types: Flat vs Hierarchical

Sources can be structured two ways, declared in the manifest:

**Flat** (default, for most sources < 2,000 docs):
```
source/
  _manifest.json
  _metadata.json
  _index.json
  backup_manifest.json  (HTML only)
  pages/ or *.zim or *.pdf
```

**Hierarchical** (for large or naturally categorized sources):
```
source/
  _manifest.json          # Identity for whole collection
  _master.json            # Index of child sources
  backup_manifest.json    # SHARED - all URL->file mappings
  pages/                  # SHARED - all backup files
  chroma/                 # SHARED - combined vector DB (optional)

  water/                  # Child source (flat structure)
    _manifest.json
    _metadata.json
    _index.json

  solar/                  # Another child source
    _manifest.json
    _metadata.json
    _index.json
```

Key insight: A **collection** contains child **sources**. Each child is a flat source. The structure is recursive - collections can contain collections if needed (e.g., Wikipedia > Science > Physics).

**What's shared vs layered:**

| Component | Shared at Collection Root | Per Child Source |
|-----------|---------------------------|------------------|
| Backup files (pages/, *.zim) | Yes | No - single copy |
| ChromaDB | Yes (optional) | No |
| `_manifest.json` | Yes (collection identity) | Yes (child identity) |
| `_master.json` | Yes (lists children) | No |
| `_metadata.json` | No | Yes |
| `_index.json` | No | Yes |
| `backup_manifest.json` | Yes | No |

---

### File Schemas

#### 1. `_manifest.json` - Source Identity (tiny, never grows with doc count)

```json
{
  "schema_version": 2,
  "source_id": "builditsolar",

  // === IDENTITY ===
  "name": "BuildItSolar",
  "description": "DIY solar energy projects",
  "source_type": "html",
  "base_url": "https://builditsolar.com",

  // === STRUCTURE ===
  "structure": "flat",
  // OR for collections:
  // "structure": "hierarchical",
  // "child_count": 12,

  // For children of a collection:
  // "parent_id": "appropedia",

  // === LICENSING ===
  "license": "Fair Use",
  "license_url": "",
  "license_verified": true,
  "license_notes": "",

  // === TAGS (for filtering/discovery, ~12 max) ===
  "tags": ["solar", "DIY", "energy", "heating", "cooling"],

  // === STATS (summary only) ===
  "document_count": 337,
  "total_chars": 2555673,

  // === BACKUP INFO (from backup pipeline) ===
  "backup": {
    "type": "html",
    "created_at": "2025-12-02",
    "size_bytes": 150000000
  },

  // === INDEX INFO (from index pipeline) ===
  "index": {
    "indexed_at": "2025-12-02",
    "embedding_model": "text-embedding-3-small",
    "dimensions": 1536
  }
}
```

For PDF collections, add:
```json
{
  "pdf_documents": {
    "Sphere-Handbook-2018-EN.pdf": {
      "title": "The Sphere Handbook",
      "authors": ["Sphere Association"],
      "publication_date": "2018-10-10",
      "page_count": 458,
      "chunk_count": 297
    }
  }
}
```

#### 2. `_metadata.json` - Document Lookup Table (small, for diffing)

```json
{
  "schema_version": 2,
  "source_id": "builditsolar",
  "document_count": 337,
  "total_chars": 2555673,
  "last_updated": "2025-12-02",
  "documents": {
    "abc123def456": {
      "title": "Solar Water Heater",
      "url": "/Projects/Water/SolarHeater",
      "content_hash": "abc123def456",
      "char_count": 5000,
      "categories": [],
      "scraped_at": "2025-12-02T10:00:00"
    }
  }
}
```

Key fields for diffing:
- `url` - Unique identifier across scrapes
- `content_hash` - Detect content changes (same hash = skip)
- `scraped_at` - Know which scrape is newer

#### 3. `_index.json` - Vectors + Content (large, for search)

```json
{
  "schema_version": 2,
  "source_id": "builditsolar",
  "embedding_model": "text-embedding-3-small",
  "dimensions": 1536,
  "document_count": 337,
  "documents": {
    "abc123def456": {
      "content": "Full text content here...",
      "embedding": [0.1, 0.2, ...]
    }
  }
}
```

#### 4. `backup_manifest.json` - URL to File Mapping (HTML sources only)

```json
{
  "schema_version": 2,
  "source_id": "solarcooking",
  "created_at": "2025-12-02",
  "page_count": 173,
  "total_size_bytes": 27133679,
  "pages": {
    "/wiki/Solar_Cooker": {
      "filename": "wiki_Solar_Cooker.html",
      "title": "Solar Cooker | Fandom",
      "size": 150000
    }
  }
}
```

#### 5. `_master.json` - Collection Index (hierarchical sources only)

At root level (D:\disaster-backups\_master.json):
```json
{
  "schema_version": 2,
  "last_updated": "2025-12-02",
  "sources": {
    "builditsolar": { "type": "source", "path": "builditsolar/" },
    "appropedia": { "type": "collection", "path": "appropedia/" },
    "bitcoin": { "type": "source", "path": "bitcoin/" }
  }
}
```

At collection level (appropedia/_master.json):
```json
{
  "schema_version": 2,
  "parent": "appropedia",
  "sources": {
    "water": { "type": "source", "path": "water/", "document_count": 500 },
    "solar": { "type": "source", "path": "solar/", "document_count": 800 },
    "food": { "type": "source", "path": "food/", "document_count": 300 }
  }
}
```

---

### Folder Structure Examples

**Flat source (BuildItSolar):**
```
builditsolar/
  _manifest.json
  _metadata.json
  _index.json
  backup_manifest.json
  pages/
  assets/
```

**ZIM source (Bitcoin):**
```
bitcoin/
  _manifest.json
  _metadata.json
  _index.json
  bitcoin.zim
```

**PDF collection:**
```
pdf_humanitarian_standards/
  _manifest.json
  _metadata.json
  _index.json
  Sphere-Handbook-2018-EN.pdf
  Base-Camp-Handbook.pdf
```

**Hierarchical collection (Appropedia):**
```
appropedia/
  _manifest.json
  _master.json
  backup_manifest.json
  pages/

  water/
    _manifest.json
    _metadata.json
    _index.json

  solar/
    _manifest.json
    _metadata.json
    _index.json
```

**Root backup folder:**
```
D:\disaster-backups\
  _master.json
  chroma/
  builditsolar/
  bitcoin/
  appropedia/
  pdf_humanitarian_standards/
```

---

### How Search Works (Reference)

Understanding how search works clarifies what each file is for:

1. **User asks question** - "How do I purify water in an emergency?"
2. **Query embedded** - Question becomes a vector using same model as index
3. **Vector similarity search** - ChromaDB/Pinecone finds closest document vectors
4. **Results returned with metadata** - Each result includes title, URL, source
5. **LLM reads content** - Full text sent to LLM for answer synthesis
6. **Response with links** - User sees answer + clickable source links

**Where the URL lives:**
- Stored in ChromaDB/Pinecone metadata (returned with search results)
- Stored in `_metadata.json` (for diffing/merging operations)
- Base URL in `_manifest.json` (combined at display time)
- `backup_manifest.json` maps URL to local filename for offline access

**What each file does:**

| File | Used For | When |
|------|----------|------|
| `_index.json` | Vector similarity search | Every query |
| `_metadata.json` | Get title/URL after match, diffing | Every query + maintenance |
| `_manifest.json` | Filter by source, show license | UI display |
| Tags | Pre-filter before search, faceted browse | Optional filtering |

**Tags vs Vectors:**
- Tags are for organization and filtering (surface level)
- Vectors capture semantic meaning (deep level)
- A query "make water safe using sunlight" matches "Solar Water Disinfection" via vectors even if those exact words aren't in tags

---

### Metadata Use Cases

**1. Detecting changes over time:**
```python
old_docs = old_metadata["documents"]
new_docs = new_metadata["documents"]

added = new_docs.keys() - old_docs.keys()
removed = old_docs.keys() - new_docs.keys()
changed = [k for k in old_docs
           if k in new_docs
           and old_docs[k]["content_hash"] != new_docs[k]["content_hash"]]
```

**2. Merging partial scrapes:**
```python
combined = {}
for partial in [water_meta, solar_meta, food_meta]:
    for doc_id, doc in partial["documents"].items():
        if doc_id not in combined:
            combined[doc_id] = doc
        elif doc["content_hash"] != combined[doc_id]["content_hash"]:
            handle_conflict(doc)
```

**3. Incremental index updates:**
```python
existing = existing_metadata["documents"].keys()
new = new_metadata["documents"].keys()
to_embed = new - existing  # Only embed these
```

---

### Migration from Current State

Current files (inconsistent):
- `_source.json` - source config
- `_documents.json` - doc metadata
- `_embeddings.json` - vectors + content
- `_manifest.json` - mixed purpose
- `_collection.json` - PDF-specific
- `_backup_manifest.json` - URL mappings
- `manifest.json` - duplicate

Target files (clean):
- `_manifest.json` - Source identity (merge _manifest + _source + _collection)
- `_metadata.json` - Doc lookup (rename from _documents)
- `_index.json` - Vectors + content (rename from _embeddings)
- `backup_manifest.json` - URL mappings (rename from _backup_manifest)

Files to delete after migration:
- `_source.json` (merged into _manifest)
- `_documents.json` (renamed to _metadata)
- `_embeddings.json` (renamed to _index)
- `_collection.json` (merged into _manifest)
- `manifest.json` (duplicate, delete)

---

### Open Questions

**Tags:**
- What's the standard vocabulary? Controlled list vs free-form?
- How do tags roll up from documents to source level?
- Current code has some tag handling - needs review before changes

**Categories in hierarchical:**
- Categories derived from document tags
- Tooling groups docs by primary tag, suggests category structure
- Manual override available via `_master.json` edits
- Threshold for flat vs hierarchical: manual decision with tooling suggestions

**Future structure types:**
- Could add `"structure": "sqlite"` for massive sources if JSON files become unwieldy
- Current flat/hierarchical covers expected use cases through 50K+ docs

---

### Implementation Status

**Working (needs cleanup):**
- Flat source structure (files exist but inconsistent naming)
- Backup pipeline for HTML, ZIM, PDF
- Index pipeline with embeddings
- ChromaDB storage with URL in metadata

**Needs implementation:**
- Consistent file naming per this schema
- Hierarchical collection support
- Migration script for existing sources
- Tooling to auto-detect structure type

**Upcoming sources requiring this:**
- Appropedia (full site, 5000+ docs) - needs hierarchical
- Akvopedia (2211 articles) - could be flat or hierarchical
- Wikipedia subsets via Kiwix ZIM - definitely hierarchical
