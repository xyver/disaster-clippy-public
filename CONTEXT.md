# Disaster Clippy - Project Context

This document provides the context needed to understand and work on the codebase. Read this first when onboarding.

---

## What This Project Does

Disaster Clippy is an AI-powered search assistant for disaster preparedness content. Users ask questions in natural language and get answers sourced from curated, verified content (wikis, guides, PDFs).

**Example:** "How do I purify water in an emergency?" returns answers with citations from Appropedia, CDC guides, survival manuals, etc.

---

## Architecture Overview

```
User Question
     |
     v
[FastAPI App] --> [Vector Search] --> [LLM] --> Response with Citations
     |                  |
     v                  v
[Admin Panel]    [ChromaDB/Pinecone]
/useradmin/            |
     |                  v
     v            [Source Backups]
[Source Tools]   HTML/ZIM/PDF files
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Chat API | `app.py` | Main FastAPI app, chat endpoints |
| Admin Panel | `admin/` | Source management UI at /useradmin/ |
| Indexers | `offline_tools/indexer.py` | HTML, ZIM, PDF content indexing |
| Scrapers | `offline_tools/scraper/` | Web content scrapers |
| Vector Store | `offline_tools/vectordb/` | ChromaDB (local) and Pinecone (cloud) |
| Cloud Storage | `offline_tools/cloud/r2.py` | Cloudflare R2 for backups |
| CLI Tools | `cli/` | Command-line utilities |

### Folder Structure

```
disaster-clippy/
|-- app.py                    # FastAPI chat interface
|-- local_settings.json       # User configuration (single source of truth)
|
|-- cli/                      # Command-line tools
|   |-- local.py              # Local admin CLI (metadata, indexing, export)
|   |-- ingest.py             # Scraping and ingestion CLI
|   |-- sync.py               # Vector DB sync CLI
|
|-- admin/                    # Admin panel (/useradmin/)
|   |-- app.py                # FastAPI routes + page routes
|   |-- local_config.py       # User settings management
|   |-- job_manager.py        # Background job queue
|   |-- ollama_manager.py     # Portable Ollama management
|   |-- cloud_upload.py       # R2 upload endpoints
|   |-- routes/               # API route modules
|   |   |-- sources.py        # Source listing API
|   |   |-- source_tools.py   # Source management API
|   |   |-- packs.py          # Pack management API
|   |   |-- jobs.py           # Job status API
|   |-- templates/            # Admin UI templates
|   |-- static/               # Admin CSS/JS
|
|-- offline_tools/            # Core business logic
|   |-- schemas.py            # Data structures
|   |-- embeddings.py         # Embedding service
|   |-- indexer.py            # HTML/ZIM/PDF indexing
|   |-- source_manager.py     # Source CRUD operations
|   |-- packager.py           # Pack tools and metadata
|   |-- registry.py           # Source pack registry
|   |-- backup/               # Backup utilities
|   |   |-- html.py           # HTML website backup
|   |   |-- substack.py       # Substack newsletter backup
|   |-- cloud/                # Cloud storage
|   |   |-- r2.py             # Cloudflare R2 client
|   |-- scraper/              # Web scrapers
|   |   |-- base.py           # Base scraper class
|   |   |-- mediawiki.py      # MediaWiki API scraper
|   |   |-- appropedia.py     # Appropedia-specific
|   |   |-- fandom.py         # Fandom wiki scraper
|   |   |-- static_site.py    # Static HTML sites
|   |   |-- pdf.py            # PDF extraction
|   |   |-- pdf_collections.py # PDF collection management
|   |   |-- substack.py       # Substack scraper
|   |-- vectordb/             # Vector database
|   |   |-- store.py          # ChromaDB local store
|   |   |-- pinecone_store.py # Pinecone cloud store
|   |   |-- metadata.py       # Fast JSON metadata index
|   |   |-- factory.py        # Store factory
|   |   |-- sync.py           # Sync operations
|
|-- templates/                # Main app templates
|-- static/                   # Main app static files
```

---

## Two Admin Modes

The same codebase supports both local and global admin modes:

| Mode | Vector DB | R2 Access | Use Case |
|------|-----------|-----------|----------|
| Local (default) | ChromaDB | Read backups/, Write submissions/ | End users running locally |
| Global | Pinecone | Full access | Maintainer managing official sources |

```bash
python app.py              # Local mode (default)
ADMIN_MODE=global python app.py  # Global mode
```

---

## Data Flow

### Local Admin

```
[Backup Files] --> [Indexer] --> [ChromaDB] --> [Search]
     |
     v
[Validation] --> [R2 submissions/] --> Global admin reviews
```

### Global Admin

```
[R2 submissions/] --> [Review] --> [Validation] --> [R2 backups/]
                                         |
                                         v
                                    [Pinecone]
```

---

## Source Pack Structure

Each source is a self-contained folder:

```
BACKUP_PATH/
|-- _master.json                         # Master source index (optional)
|-- my_wiki/                             # Source folder
|   |-- _manifest.json                   # Source config (identity + distribution)
|   |-- _metadata.json                   # Document metadata
|   |-- _index.json                      # Full content for display
|   |-- _vectors.json                    # Vector embeddings
|   |-- backup_manifest.json             # URL to file mapping
|   |-- pages/                           # HTML backup content
|-- bitcoin.zim                          # ZIM files at backup root
+-- chroma/                              # ChromaDB data (all sources)
```

### R2 Cloud Structure

```
r2-bucket/
  backups/
    _master.json           # Global source index
    {source_id}/           # Individual source packages
  submissions/
    {source_id}/           # Pending packages from local admins
```

---

## Validation System

A source is ready for distribution when it has 5 status boxes green:

1. **Config** - `_manifest.json` exists with required fields
2. **Backup** - HTML pages, ZIM file, or PDF files exist
3. **Metadata** - Document list file exists
4. **Embeddings** - Vectors created for search
5. **License** - Specified (not "Unknown") and verified

The Source Tools wizard (admin panel) guides users through fixing each status.

---

## Key Design Decisions

### No Backward Compatibility

This is a first-version codebase. Old data formats, file names, and import paths do not need to be supported. If something breaks, re-index from backup files rather than writing migration scripts.

### File Naming

| File | Purpose |
|------|---------|
| `_manifest.json` | Source identity + distribution info |
| `_metadata.json` | Document lookup table |
| `_index.json` | Full content for scanning/display |
| `_vectors.json` | Embeddings only |
| `backup_manifest.json` | URL to local file mapping |

File naming is centralized in `offline_tools/schemas.py` via getter functions.

---

## Important Files

| File | What It Does |
|------|--------------|
| `app.py` | Main FastAPI app, chat endpoints, search logic |
| `admin/app.py` | Admin panel routes and page handlers |
| `admin/templates/source_tools.html` | 5-step source creation wizard |
| `offline_tools/source_manager.py` | Source CRUD, validation, license detection |
| `offline_tools/indexer.py` | HTMLBackupIndexer, ZIMIndexer, PDFIndexer |
| `offline_tools/vectordb/store.py` | ChromaDB vector store |
| `offline_tools/vectordb/pinecone_store.py` | Pinecone cloud store |
| `local_settings.json` | User configuration (backup path, mode) |

---

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | Embeddings and chat | Yes (unless EMBEDDING_MODE=local) |
| `ANTHROPIC_API_KEY` | Claude chat (optional) | No |
| `PINECONE_API_KEY` | Cloud vector DB | If VECTOR_DB_MODE=pinecone or global |
| `R2_ACCESS_KEY_ID` | Cloud storage | For R2 upload/download |
| `R2_SECRET_ACCESS_KEY` | Cloud storage | For R2 upload/download |
| `EMBEDDING_MODE` | "openai" or "local" | No (defaults to openai) |
| `VECTOR_DB_MODE` | "local", "pinecone", or "global" | No (defaults to local) |

**VECTOR_DB_MODE controls deployment:**
- `local` - Admin UI visible, local ChromaDB, R2 read backups + R/W submissions
- `pinecone` - Admin UI blocked (public mode), Pinecone cloud search only
- `global` - Admin UI visible, Pinecone R/W, R2 full access

---

## Current Work (v0.9 Pre-release)

### Completed (Dec 2025)

The codebase was recently consolidated from two repos (private + public):

- Folder reorganization (`admin/`, `offline_tools/`, `cli/`)
- Scrapers ported from private repo to `offline_tools/scraper/`
- CLI tools consolidated into `cli/` folder
- Source Tools wizard (5-step flow) implemented
- Status boxes for validation
- Pinecone sync functionality
- Route refactoring (extracted API into modular route files)
- Removed all legacy/backwards compatibility code

### Features (Complete)

- ADMIN_MODE gating for global-only features
- Schema standardization (all tools use schemas.py)
- ZIM metadata extraction (auto-populates license, description, tags from ZIM files)
- Tag taxonomy expansion (28 categories for better content categorization)
- Language filtering for multi-language ZIM files
- Source filtering in chat UI
- Chat UX improvements (links open in new tabs)
- Unified AI Pipeline with streaming support

### Security (Complete)

- VECTOR_DB_MODE=pinecone blocks admin UI (public Railway deployment)
- VECTOR_DB_MODE=global enables full cloud write access
- Two-bucket R2 system (backups + submissions) with separate credentials
- Railway proxy endpoints for local admins without API keys
- Rate limiting on all public endpoints (5-30 requests/min)
- Cloud-only mode (Railway works without BACKUP_PATH)
- EMBEDDING_MODE=local error handling improved

### In Progress (v0.9)

- Pipeline testing with real data sources
- Documentation cleanup and consolidation
- Final validation before v1.0 release

See [ROADMAP.md](ROADMAP.md) for full details.

---

## Quick Start for Development

```bash
# Install
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your API keys

# Run
python app.py

# Access
# Chat: http://localhost:8000
# Admin: http://localhost:8000/useradmin/
```

---

## Scrapers Available

| Scraper | Location | Use For |
|---------|----------|---------|
| MediaWiki | `offline_tools/scraper/mediawiki.py` | Any MediaWiki site |
| Appropedia | `offline_tools/scraper/appropedia.py` | Appropedia.org |
| Fandom | `offline_tools/scraper/fandom.py` | Fandom wiki sites |
| Static Site | `offline_tools/scraper/static_site.py` | Generic HTML sites |
| PDF | `offline_tools/scraper/pdf.py` | PDF documents |
| Substack | `offline_tools/scraper/substack.py` | Substack newsletters |

### Scraper Usage

```bash
# Scrape a MediaWiki site
python cli/ingest.py scrape mediawiki --url https://wiki.example.org --limit 100

# Scrape Appropedia by category
python cli/ingest.py scrape appropedia --category "Water" --limit 50
```

---

## Indexers Available

| Indexer | Class | Input |
|---------|-------|-------|
| HTML Backup | `HTMLBackupIndexer` | Folder with pages/ directory |
| ZIM Archive | `ZIMIndexer` | .zim file |
| PDF | `PDFIndexer` | PDF file or folder |

### Indexer Usage

```bash
# Index HTML backup
python cli/local.py index-html --path ./backups/mysite --source-id mysite

# Index ZIM file
python cli/local.py index-zim --path ./backups/wikipedia.zim --source-id wikipedia
```

---

## Testing

```bash
# Run the app
python app.py

# Test chat API
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I filter water?"}'

# Check admin panel
# Visit http://localhost:8000/useradmin/
```

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Quick start and project overview |
| [DEVELOPER.md](DEVELOPER.md) | Technical details, CLI tools, security |
| [SUMMARY.md](SUMMARY.md) | Executive summary (non-technical) |
| [ROADMAP.md](ROADMAP.md) | Future plans and testing checklist |

---

*Last Updated: December 2025*
