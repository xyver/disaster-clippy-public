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
|   |-- zim_inspect.py        # ZIM file diagnostic tool
|
|-- admin/                    # Admin panel (/useradmin/)
|   |-- app.py                # FastAPI routes + page routes
|   |-- local_config.py       # User settings management
|   |-- ai_service.py         # Unified AI search/response service
|   |-- connection_manager.py # Smart connectivity detection
|   |-- job_manager.py        # Background job queue
|   |-- ollama_manager.py     # Portable Ollama management
|   |-- cloud_upload.py       # R2 upload endpoints
|   |-- zim_server.py         # ZIM content server for offline browsing
|   |-- routes/               # API route modules
|   |   |-- sources.py        # Source listing API
|   |   |-- source_tools.py   # Source management API (includes ZIM tools)
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
|   |-- zim_utils.py          # ZIM inspection and metadata utilities
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

## Security Architecture

### Deployment Modes (VECTOR_DB_MODE)

A single environment variable controls your deployment mode:

| Mode | Admin UI | Vector DB | R2 Access |
|------|----------|-----------|-----------|
| `local` (default) | Yes | ChromaDB | Read backups, R/W submissions |
| `pinecone` | No | Pinecone (read) | None (public chat only) |
| `global` | Yes | Pinecone (R/W) | Full access |

### Feature Access by Mode

| Feature | local | pinecone | global |
|---------|-------|----------|--------|
| Chat UI | Yes | Yes | Yes |
| Admin UI (`/useradmin/`) | Yes | No | Yes |
| Local ChromaDB | Yes | No | No |
| Pinecone Read (search) | Via proxy | Yes | Yes |
| Pinecone Write (sync) | No | No | Yes |
| R2 Read (download backups) | Via proxy | No | Yes |
| R2 Write (submissions) | Via proxy | No | Yes |
| R2 Full (official backups) | No | No | Yes |

### Rate Limits (Railway)

| Endpoint | Limit | Purpose |
|----------|-------|---------|
| `/chat`, `/api/v1/chat` | 10/min | Chat requests |
| `/api/cloud/sources` | 30/min | List sources |
| `/api/cloud/download/{id}` | 10/min | List files |
| `/api/cloud/download/{id}/{file}` | 5/min | Download file |
| `/api/cloud/submit` | 5/min | Submit content |

### Railway Proxy for Local Admins

Local admins without API keys can access cloud features through Railway proxy endpoints:

```
GET  /api/cloud/sources          - List available backups
GET  /api/cloud/download/{id}    - Stream backup download
POST /api/cloud/submit           - Submit content for review
```

Configure with `RAILWAY_PROXY_URL` in local settings.

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
| `admin/ai_service.py` | Unified AI search and response service |
| `admin/connection_manager.py` | Smart connectivity detection and mode switching |
| `admin/zim_server.py` | ZIM content server for offline browsing |
| `offline_tools/source_manager.py` | Unified source creation interface |
| `offline_tools/packager.py` | Metadata and index generation |
| `offline_tools/indexer.py` | Indexers for HTML, ZIM, PDF |
| `offline_tools/zim_utils.py` | ZIM inspection and metadata utilities |
| `offline_tools/backup/` | HTML and Substack backup tools |
| `offline_tools/scraper/` | Web scrapers (MediaWiki, Fandom, static sites) |
| `offline_tools/vectordb/` | Vector store implementations |
| `offline_tools/cloud/r2.py` | Cloudflare R2 client |

---

## AI Service Architecture

The AI service (`admin/ai_service.py`) provides a unified interface for search and response generation across all connection modes.

### Connection Modes

| Mode | Search | Response | Pinging |
|------|--------|----------|---------|
| `online_only` | Semantic (embedding API) | Cloud LLM | Yes (warn on disconnect) |
| `hybrid` | Semantic with keyword fallback | Cloud LLM with Ollama fallback | Yes (detect recovery) |
| `offline_only` | Keyword only | Ollama or simple response | No |

### Using the AI Service

```python
from admin.ai_service import get_ai_service

# Get the singleton service
ai = get_ai_service()

# Search (automatically uses correct method based on mode)
result = ai.search("how to filter water", n_results=10)
print(f"Found {len(result.articles)} articles via {result.method}")

# Generate response
response = ai.generate_response(query, context, history)
print(f"Response via {response.method}: {response.text}")

# Streaming response
for chunk in ai.generate_response_stream(query, context, history):
    print(chunk, end="", flush=True)
```

### Connection Manager

The connection manager (`admin/connection_manager.py`) handles smart connectivity detection:

```python
from admin.connection_manager import get_connection_manager

conn = get_connection_manager()

# Check if online
if conn.should_try_online():
    # Try online API

# Report success/failure
conn.on_api_success()  # Resets ping timer
conn.on_api_failure()  # Triggers immediate ping check

# Get status for frontend
status = conn.get_status()
# Returns: {mode, is_online, temporarily_offline, effective_mode, ...}
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/chat` | POST | Standard chat (waits for full response) |
| `/api/v1/chat/stream` | POST | Streaming chat (SSE) |
| `/api/v1/connection-status` | GET | Get current connection status |
| `/api/v1/ping` | POST | Trigger connectivity check |
| `/sources` | GET | List available sources with document counts |
| `/welcome` | GET | Get welcome message and stats |

### Source Filtering in Chat

Users can filter search results by source using the "Select Sources" dropdown in the chat interface:

- **Select All**: Search all indexed sources (default)
- **Select None**: Disable search (useful for testing)
- **Individual sources**: Check/uncheck specific sources to include

The selection is persisted to localStorage (`clippy_selected_sources`) so it survives page refreshes.

**API Usage:**
```javascript
// Filter to specific sources
fetch('/api/v1/chat/stream', {
    method: 'POST',
    body: JSON.stringify({
        message: "how to filter water",
        sources: ["appropedia", "wikihow"]  // Only search these sources
    })
});
```

### Chat Link Behavior

Links in chat responses and the articles sidebar open in new tabs to preserve chat history:

- **ZIM article links**: Open in new tab with `target="_blank"`
- **External URLs**: Open in new tab with `rel="noopener noreferrer"`
- **Markdown links**: Parsed and converted to clickable links that open in new tabs

This prevents users from losing their conversation when clicking to read an article.

### Connection Status Display

The `/api/v1/connection-status` endpoint returns unified status data used by all UI pages:

```json
{
  "mode": "hybrid",
  "state": "online",
  "state_label": "Online",
  "state_color": "green",
  "state_icon": "check",
  "message": "Connected to cloud services",
  "is_online": true,
  "temporarily_offline": false,
  "effective_mode": "hybrid_online"
}
```

**Connection States:**

| State | Color | Description |
|-------|-------|-------------|
| `online` | green | Securely connected, recent successful API call |
| `checking` | blue | Currently verifying connection |
| `unstable` | yellow | Hybrid mode with intermittent issues |
| `disconnected` | red | Online mode but connection lost |
| `offline` | gray | User intentionally in offline mode |
| `recovering` | blue | Was offline, now detecting recovery |

**UI Implementations:**

- **Dashboard** (`admin/templates/dashboard.html`): Shows "Connection State" card with label, color, and message
- **Settings** (`admin/templates/settings.html`): Status bar with colored dot and state text
- **Chat** (`templates/index.html` + `static/chat.js`): Header indicator with colored dot, auto-refreshes every 30 seconds

All three pages fetch from the same `/api/v1/connection-status` endpoint for consistency.

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

### Language Filtering for Multi-Language ZIMs

Multi-language ZIM files (like Appropedia) contain articles in many languages. Use the language filter during indexing to index only articles in your preferred language:

**In Source Tools (Step 3):**
- Select a language from the "Language Filter" dropdown
- Only articles detected as that language will be indexed
- Use "Force Re-index" to clear existing documents and re-index with the new filter

**Supported Languages (30+):**
- Common: English, Spanish, French, German, Portuguese, Italian, Russian, Chinese, Japanese, Korean, Arabic, Hindi
- Additional: Vietnamese, Thai, Indonesian, Malay, Tagalog, Swahili, Haitian Creole, Bengali, Nepali, Urdu, Persian, Turkish, Polish, Dutch, Ukrainian, Romanian, Greek, Hebrew, Amharic, Sinhala, Tamil, Telugu, Burmese, Khmer, Lao

**Detection Methods:**
- URL path segments (e.g., `/en/`, `/es/`, `/french/`)
- Title suffixes (e.g., `(Spanish)`, `(Chinese)`, `(Haitian Creole)`)
- Language keywords after separators (e.g., `Solar cooker - Vietnamese`)

Note: Articles with no detectable language marker are included by default (they may be in the target language without explicit marking).

### ZIM Offline Browsing

The ZIM server (`admin/zim_server.py`) enables seamless offline browsing of ZIM archive content. When users click on a search result from a ZIM source, they browse the archived content locally.

**Routes:**

| Endpoint | Purpose |
|----------|---------|
| `/zim/{source_id}/{path}` | Serve article content from ZIM |
| `/zim/{source_id}` | Serve ZIM main page or index |
| `/zim/api/sources` | List all available ZIM sources |

**Features:**

- **URL Index Caching**: Builds an in-memory index on first access for O(1) article lookups
- **Internal Link Rewriting**: Converts relative and absolute links to `/zim/` URLs
- **Dead Site Handling**: Rewrites absolute URLs matching `base_url` to local ZIM URLs
- **Navigation Button**: Minimal floating "Back to Search" button in bottom-right corner
- **MIME Type Detection**: Serves HTML, images, CSS, and other content with correct types

**Link Rewriting Examples:**

```
href="/wiki/Page"              -> href="/zim/{source_id}/wiki/Page"
href="../Page"                 -> href="/zim/{source_id}/Page"
src="/images/foo.png"          -> src="/zim/{source_id}/images/foo.png"
href="https://deadsite.com/p"  -> href="/zim/{source_id}/p" (if base_url matches)
```

**Dead Site Handling:**

When a ZIM is a backup of a site that no longer exists, the server rewrites absolute URLs that match the `base_url` in the source's `_manifest.json`. This handles variations:
- http vs https
- www vs non-www

**KNOWN ISSUE - Dead Site Detection:**

Dead site detection is NOT automatic. The system relies on `base_url` being set in the source's `_manifest.json`. There is currently no intelligent detection of whether a site is dead.

Future enhancement options:
- Add `prefer_local` toggle in source config to force local URLs
- Automatic dead site detection via HTTP checks
- User prompt when external links fail

### ZIM Inspection Tools

ZIM files vary in content type (websites, videos, PDFs) and text density. Use the inspection tools before indexing:

**CLI Tool:**
```bash
python cli/zim_inspect.py /path/to/file.zim
python cli/zim_inspect.py /path/to/file.zim --scan-limit 10000 --min-text 30 -v
python cli/zim_inspect.py /path/to/file.zim --json
```

**Admin API Endpoints (`/useradmin/api/zim/...`):**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/zim/list` | GET | Find all ZIM files in backup folder |
| `/zim/inspect` | POST | Run full diagnostic analysis |
| `/zim/metadata/{source_id}` | GET | Quick metadata extraction |
| `/zim/index` | POST | Index ZIM to ChromaDB |

**Inspection Output:**

- Header metadata (title, description, creator, license, etc.)
- Content type detection (website, video, pdf, mixed)
- Mimetype and namespace distribution
- Text length analysis (indexable vs skipped articles)
- Sample articles with text previews
- Recommendations for indexing parameters

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

# Inspect a ZIM file before indexing
python cli/zim_inspect.py /path/to/file.zim
python cli/zim_inspect.py /path/to/file.zim --scan-limit 10000 -v --json
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
| Inspect ZIM file | `python cli/zim_inspect.py /path/to/file.zim` |
| Browse ZIM content | localhost:8000/zim/{source_id}/ |

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

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Project overview and quick start |
| [CONTEXT.md](CONTEXT.md) | Architecture and design decisions (AI onboarding) |
| [SUMMARY.md](SUMMARY.md) | Executive summary (non-technical) |
| [ROADMAP.md](ROADMAP.md) | Future plans, testing, and feature development |

---

## R2 Cloud Storage (Advanced)

### Two-Bucket Architecture

The R2 storage system uses two separate buckets for security:

| Bucket | Purpose | Railway Access | Global Admin |
|--------|---------|----------------|--------------|
| `disaster-clippy-backups` | Official content | Read only | Read/Write |
| `disaster-clippy-submissions` | User submissions | Write only | Read/Delete |

### R2 Storage Functions (`offline_tools/cloud/r2.py`)

**Bucket Getters:**
- `get_backups_storage()` - Uses `R2_BACKUPS_BUCKET` (reads official content)
- `get_submissions_storage()` - Uses `R2_SUBMISSIONS_BUCKET` (writes user submissions)
- `get_r2_storage()` - Legacy single-bucket mode (backward compatible)

**Server-Side Copy Methods:**
- `copy_to_bucket(source_key, dest_bucket, dest_key)` - Cross-bucket copy
- `move_to_bucket(source_key, dest_bucket, dest_key)` - Copy + delete

**Helper Functions:**
- `approve_submission(submission_key, dest_source_id, dest_filename)` - Server-side approve
- `reject_submission(submission_key, reason)` - Move to rejected folder

### R2 Environment Variables

**Railway Deployment** (limited access tokens):

```bash
R2_ACCESS_KEY_ID=<your-key-id>
R2_SECRET_ACCESS_KEY=<your-secret>
R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
R2_BACKUPS_BUCKET=disaster-clippy-backups
R2_SUBMISSIONS_BUCKET=disaster-clippy-submissions
```

**Global Admin** (full access token):

```bash
R2_ACCESS_KEY_ID=<admin-key-id>
R2_SECRET_ACCESS_KEY=<admin-secret>
R2_ENDPOINT_URL=https://<account>.r2.cloudflarestorage.com
R2_BACKUPS_BUCKET=disaster-clippy-backups
R2_SUBMISSIONS_BUCKET=disaster-clippy-submissions
```

### Cloudflare R2 Setup

1. Create bucket: `disaster-clippy-backups`
2. Create bucket: `disaster-clippy-submissions`
3. Create API token for Railway: Read on backups, Write on submissions
4. Create API token for Global Admin: Full access on both

The code is backward compatible - if you only set `R2_BUCKET_NAME`, it uses single-bucket mode.

---

## Vector Database Configuration

`VECTOR_DB_MODE` controls where vectors are stored:

| Mode | Storage | Use Case |
|------|---------|----------|
| `local` (default) | ChromaDB in `BACKUP_PATH/chroma/` | Local admin, offline use |
| `pinecone` | Pinecone cloud service | Railway deployment, global admin |

### Pinecone Variables

Only needed if `VECTOR_DB_MODE=pinecone`:

| Variable | Description |
|----------|-------------|
| `PINECONE_API_KEY` | Your Pinecone API key from console.pinecone.io |
| `PINECONE_ENVIRONMENT` | Region (e.g., us-east-1, gcp-starter) |
| `PINECONE_INDEX_NAME` | Index name (default: disaster-clippy) |

### Deployment Requirements

| Deployment | VECTOR_DB_MODE | Pinecone Keys |
|------------|----------------|---------------|
| Local admin | `local` | Not needed |
| Railway public | `pinecone` | Yes - in Railway env vars |
| Global admin | `pinecone` | Yes - for syncing to cloud |

---

## URL Handling (Local vs Cloud)

The indexer stores both online and local URLs for each document:

| Source Type | `url` Field | `local_url` Field |
|-------------|-------------|-------------------|
| ZIM | Online URL (e.g., `https://en.bitcoin.it/wiki/Mining`) | `/zim/{source_id}/{article_path}` |
| HTML | `base_url` + relative path | `/backup/{source_id}/{filename}` |
| PDF | `/api/pdf/{source_id}/{filename}` | `file://{local_path}` |

### Context-Aware Display

The `_get_display_url()` helper in `app.py` selects the appropriate URL:

| Deployment | Vector DB | URL Used | Result |
|------------|-----------|----------|--------|
| Railway (`PUBLIC_MODE=true`) | Pinecone | `url` | Online URLs |
| Local admin | ChromaDB | `local_url` | Local viewer URLs |

---

*Last Updated: December 2025*
