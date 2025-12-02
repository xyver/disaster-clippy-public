# Disaster Clippy - Project Context

**Purpose:** Complete context for AI assistants working on this codebase. Read this instead of reading all the code.

**Last Updated:** December 2025
**Version:** 0.9

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

## Architecture

```
User --> FastAPI (app.py) --> Vector Search --> LLM --> Response
                                   |
                              ChromaDB/Pinecone
                                   ^
                              Scrapers (ingest.py)
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

```
disaster-clippy/
|-- app.py                 # FastAPI backend
|-- ingest.py              # Ingestion CLI
|-- sync.py                # Sync CLI
|-- scraper/               # Content scrapers
|-- vectordb/              # Vector database layer
|-- admin/app.py           # Streamlit dashboard
|-- config/sources.json    # Source registry
|-- data/chroma/           # Local vector DB
|-- data/metadata/         # Metadata index (JSON)
|-- templates/index.html   # Chat UI template
|-- static/                # Frontend assets
```

---

## How to Work on This Project

1. **Read this file first** - You now have full context

2. **For code changes** - Check the relevant module (scraper/, vectordb/, app.py)

3. **For new scrapers** - Extend BaseScraper, register in SCRAPER_REGISTRY

4. **For admin features** - Modify admin/app.py (Streamlit)

5. **For API changes** - Modify app.py (FastAPI endpoints)

6. **Test locally** before pushing:
   ```bash
   python app.py  # Test chat
   streamlit run admin/app.py  # Test admin
   python sync.py --remote pinecone compare  # Check sync status
   ```

---

*This document should be updated when significant architecture changes occur.*
