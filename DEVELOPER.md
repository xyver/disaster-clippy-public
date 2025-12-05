# Disaster Clippy - Developer Documentation

This guide covers both development setup and local user configuration.

---

## Repository Structure

The repository has been consolidated into a single codebase with mode switching for local vs global admin features.

### Folder Structure

```
disaster-clippy/
|-- app.py                    # FastAPI chat interface
|-- local_settings.json       # User configuration (single source of truth)
|
|-- cli/                      # Command-line tools
|   |-- __init__.py
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
|   |-- __init__.py
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

## Mode Switching (Local vs Global Admin)

The same codebase supports both local and global admin modes:

```bash
# Local admin (default)
python app.py

# Global admin
ADMIN_MODE=global python app.py
```

| Feature | Local Admin | Global Admin |
|---------|-------------|--------------|
| Vector DB | ChromaDB (local) | Pinecone (cloud) |
| R2 backups/ | Read only | Read/Write |
| R2 submissions/ | Write only | Read only |
| Pinecone page | Hidden | Visible |
| Submissions page | Hidden | Visible |

### Access Levels

| Role | Vector DB | R2 Storage | What They Use |
|------|-----------|------------|---------------|
| End User | ChromaDB (local) | Read backups/ only | Chat interface |
| Local Admin | ChromaDB (local) | Write submissions/, Read backups/ | Admin panel + CLI tools |
| Global Admin | Pinecone (write) | Full access | Admin panel (global mode) |

---

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/xyver/disaster-clippy-public.git
cd disaster-clippy-public
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```bash
# Minimum required - for embeddings
OPENAI_API_KEY=your-api-key-here

# Optional - use Claude for chat instead of GPT
# ANTHROPIC_API_KEY=your-anthropic-key
# LLM_PROVIDER=anthropic

# Use local database (default for personal use)
VECTOR_DB_MODE=local

# Optional - use free local embeddings (no API cost)
# EMBEDDING_MODE=local
```

### 3. Start the Application

```bash
python app.py
```

Open your browser:
- **Chat Interface**: http://localhost:8000
- **Local Admin**: http://localhost:8000/useradmin/

---

## Key Modules

| Module | Purpose |
|--------|---------|
| `offline_tools/source_manager.py` | Unified source creation interface |
| `offline_tools/packager.py` | Metadata and index generation |
| `offline_tools/indexer.py` | Indexers for HTML, ZIM, PDF |
| `offline_tools/backup/` | HTML and Substack backup tools |
| `offline_tools/scraper/` | Web scrapers (MediaWiki, Fandom, static sites) |
| `offline_tools/vectordb/` | Vector store implementations |
| `offline_tools/cloud/r2.py` | Cloudflare R2 client |

---

## Data Flow Architecture

```
LOCAL ADMIN                              GLOBAL ADMIN
-----------                              ------------

[Local Backup Folder]                    [R2 Cloud Storage]
      |                                         ^
      v                                         |
[Validation] -----> [R2 submissions/] --------->+
      |              (review queue)             |
      v                                         v
[ChromaDB]                               [Validation]
(local search)                                  |
                                                v
                                         [R2 backups/]
                                         [Pinecone]
                                         (production)
```

**Key Points:**
- BACKUP_PATH is the working directory for all source data
- Global admin uploads finished sources to Pinecone + R2 (source of truth)
- Local users download from R2 to their BACKUP_PATH
- Local admins can submit sources to R2 `submissions/` for review

---

## Source Tools

### SourceManager (`offline_tools/source_manager.py`)

High-level interface for source creation workflow:

```python
from offline_tools.source_manager import SourceManager

manager = SourceManager()  # Uses BACKUP_PATH automatically

# Create an index (auto-detects source type)
result = manager.create_index("my_wiki")

# Validate source before distribution
validation = manager.validate_source("my_wiki", source_config)
# Returns: has_backup, has_index, has_license, detected_license, suggested_tags
```

### Indexers (`offline_tools/indexer.py`)

| Class | Source Type | Input |
|-------|-------------|-------|
| `HTMLBackupIndexer` | HTML websites | Backup folder with pages/ |
| `ZIMIndexer` | ZIM archives | .zim file |
| `PDFIndexer` | PDF documents | PDF file or folder |

### ZIM Metadata Extraction

When indexing ZIM files, metadata is automatically extracted from the ZIM header_fields:

| ZIM Field | Manifest Field | Description |
|-----------|----------------|-------------|
| Title | name | Human-readable source name |
| Description | description | Content description |
| Creator | attribution | Organization that created content |
| Publisher | publisher | Organization that created ZIM |
| License | license | License information |
| Language | language | ISO language code (e.g., 'eng') |
| Tags | tags | Semicolon-separated topic tags |
| Source | base_url | Original URL of content |
| Date | zim_date | Creation date |

User-edited values in `_manifest.json` are preserved on re-index.

### Tag System

Sources can be tagged for categorization and search filtering. Tags are either:
- **Auto-suggested** during indexing based on content keywords
- **Manual** - set by user in `_manifest.json`

Available tag categories (defined in `SourceManager.TOPIC_KEYWORDS`):

| Category | Example Keywords |
|----------|------------------|
| water, sanitation | filtration, purification, well, hygiene |
| solar, energy, wind, biogas, fuel | photovoltaic, generator, battery, off-grid |
| food, agriculture, livestock, aquaculture, foraging | cooking, farming, permaculture, fishing |
| shelter, construction, tools | building, masonry, earthbag, workshop |
| medical, herbal, mental-health, nutrition | first aid, medicinal plant, vitamin |
| emergency, fire, earthquake, flood, hurricane, nuclear, pandemic | survival, preparedness, evacuation |
| navigation, communication, security, knots | compass, radio, ham radio, rope |
| appropriate-tech, electronics, vehicles | low tech, arduino, bicycle |
| reference, how-to | manual, handbook, tutorial, diy |

### Source Validation

A source is ready for distribution when it has 5 status boxes green:

1. **Config** - `_manifest.json` exists with name, license, base_url
2. **Backup** - HTML pages, ZIM file, or PDF files exist
3. **Metadata** - `_metadata.json` exists
4. **Embeddings** - `_vectors.json` created for search
5. **License** - Specified and verified

### File Structure in BACKUP_PATH

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
+-- chroma/                              # ChromaDB data
```

---

## Local Admin Panel

Access at: `http://localhost:8000/useradmin/`

### Features

- **Dashboard** - System status and statistics
- **Sources** - Browse all sources with status boxes, install cloud sources
- **Source Tools** - 5-step wizard for creating/editing sources
- **Settings** - Configure backup paths, connection modes

### Connection Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| **Online Only** | Always uses internet for queries | When you have reliable internet |
| **Hybrid** (Recommended) | Uses internet when available, falls back to offline | Best of both worlds |
| **Offline Only** | Never connects to internet | Air-gapped systems, no internet |

### Settings File

Your settings are saved to: `local_settings.json` (in the project root)

```json
{
  "backup_path": "D:\\disaster-backups",
  "offline_mode": "hybrid",
  "auto_fallback": true,
  "cache_responses": true
}
```

---

## CLI Tools

Command-line tools are in the `cli/` folder:

```bash
# Generate metadata from HTML backup
python cli/local.py metadata --path ./backups/mysite --output metadata.json

# Index HTML backup to local ChromaDB
python cli/local.py index-html --path ./backups/mysite --source-id mysite

# Index ZIM file to local ChromaDB
python cli/local.py index-zim --path ./backups/wikipedia.zim --source-id wikipedia

# Scrape a MediaWiki site
python cli/ingest.py scrape mediawiki --url https://wiki.example.org --limit 100

# Sync to Pinecone (global admin only)
python cli/sync.py push --source-id mysite
```

Run `python cli/local.py --help` for full usage.

---

### Getting ZIM Files

ZIM files are compressed offline archives. Download from:

1. **Kiwix Library**: https://library.kiwix.org/
   - Wikipedia (by topic: medicine, technology, etc.)
   - Wikihow
   - StackExchange sites

2. **Direct Downloads**:
   - Search for `[topic] kiwix zim download`
   - Files range from 50MB to several GB

Recommended starter ZIMs:
- `wikipedia_en_medicine` (~500MB) - Medical reference
- `wikihow_en_all` (~2GB) - How-to guides
- `wikibooks_en_all` (~1GB) - Technical books

### Creating HTML Backups

**Option 1: Browser Save**
- Visit any page and use "Save As" -> "Webpage, Complete"
- Organize saved pages into folders by source

**Option 2: Use the built-in scrapers**
```bash
python cli/ingest.py scrape mediawiki --url https://wiki.example.org --output ./backups/mywiki
```

**Option 3: HTTrack / wget**
- Use [HTTrack](https://www.httrack.com/) to mirror websites
- Or use wget: `wget -r -l 2 -p https://example.com`

---

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | Embeddings and chat | Yes (unless EMBEDDING_MODE=local) |
| `ANTHROPIC_API_KEY` | Claude chat (optional) | No |
| `PINECONE_API_KEY` | Cloud vector DB | Global admin only |
| `R2_ACCESS_KEY_ID` | Cloud storage | For R2 upload/download |
| `R2_SECRET_ACCESS_KEY` | Cloud storage | For R2 upload/download |
| `ADMIN_MODE` | "local" or "global" | No (defaults to local) |
| `EMBEDDING_MODE` | "openai" or "local" | No (defaults to openai) |
| `VECTOR_DB_MODE` | "local" or "pinecone" | No (defaults to local) |

---

## Quick Reference

| Task | Command/URL |
|------|-------------|
| Run chat locally | `python app.py` -> localhost:8000 |
| Admin panel | localhost:8000/useradmin/ |
| Index local backups | `python cli/local.py index-html ...` |
| Scrape content | `python cli/ingest.py scrape ...` |
| Sync to Pinecone | `python cli/sync.py push` |

---

## Troubleshooting

### "No sources indexed yet"

Your database is empty. Index some content using the Local Admin panel:
1. Go to `/useradmin/` -> Sources tab
2. Use Source Tools to create a new source
3. Or install a cloud source pack

### "Unable to connect to OpenAI"

Check your API key in `.env`:
```bash
OPENAI_API_KEY=your-key-here
```

Or switch to local embeddings (free):
```bash
EMBEDDING_MODE=local
```

### "Port 8000 already in use"

The system automatically tries port 8001. Or kill the process using port 8000:

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <pid> /F

# Mac/Linux
lsof -i :8000
kill -9 <pid>
```

### Slow performance

- Use local embeddings: `EMBEDDING_MODE=local`
- Reduce `n_results` in searches
- Index only the sources you need

---

## Legacy Migration Code (DELETE AFTER MIGRATIONS COMPLETE)

The following locations contain legacy file format handling that can be deleted once all sources have been migrated to the current schema. Search for "LEGACY" or "legacy" to find these.

**Files with legacy fallback code to remove:**

| File | Lines | Description |
|------|-------|-------------|
| `offline_tools/backup/html.py` | 61-87 | Legacy manifest paths and migration logic |
| `offline_tools/indexer.py` | 555-556 | Legacy backup manifest path fallbacks |
| `offline_tools/packager.py` | 123-134 | Legacy ZIM location check |
| `offline_tools/packager.py` | 402-424 | Legacy metadata/manifest fallback reads |
| `offline_tools/packager.py` | 582-598 | `load_metadata()` legacy format fallback |
| `offline_tools/source_manager.py` | 78-97 | `SourceHealth` legacy fields |
| `offline_tools/source_manager.py` | 547-650 | `cleanup_redundant_files()` - keep function but simplify |
| `offline_tools/source_manager.py` | 1185-1189 | Legacy manifest path check |
| `offline_tools/source_manager.py` | 1391-1413 | Legacy embeddings format checks |
| `offline_tools/source_manager.py` | 1472-1504 | Legacy file detection in health check |
| `offline_tools/source_manager.py` | 1597-1642 | Legacy metadata/manifest fallbacks |
| `offline_tools/source_manager.py` | 1713-1716 | Legacy metadata fallback |
| `offline_tools/vectordb/pinecone_store.py` | 332 | Legacy _master.json path |

**Legacy file patterns being checked:**
- `{source_id}_metadata.json` -> now `_metadata.json`
- `{source_id}_backup_manifest.json` -> now `backup_manifest.json`
- `{source_id}_source.json` -> now `_manifest.json`
- `{source_id}_documents.json` -> now `_metadata.json`
- `{source_id}_embeddings.json` -> now `_vectors.json`
- `{source_id}_index.json` -> now `_index.json`
- `{source_id}_manifest.json` -> merged into `_manifest.json`

**To clean up after migrations:**
1. Run `cleanup_redundant_files()` on all sources via Source Tools
2. Verify no sources have legacy files remaining
3. Delete the fallback code paths listed above
4. Simplify `SourceHealth` dataclass to remove legacy fields

---

## Other Documentation

- [CONTEXT.md](CONTEXT.md) - Architecture and design decisions (AI onboarding)
- [SUMMARY.md](SUMMARY.md) - Executive summary
- [README.md](README.md) - Project overview
- [ROADMAP.md](ROADMAP.md) - Future plans

---

*Last Updated: December 2025*
