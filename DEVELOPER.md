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
| `admin/backup_server.py` | HTML backup content server for offline browsing |
| `offline_tools/source_manager.py` | Unified source creation interface |
| `offline_tools/packager.py` | Metadata and index generation |
| `offline_tools/indexer.py` | Indexers for HTML, ZIM, PDF |
| `offline_tools/zim_utils.py` | ZIM inspection and metadata utilities |
| `offline_tools/backup/html.py` | HTML website scraper with BFS crawling |
| `offline_tools/backup/substack.py` | Substack newsletter backup |
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

High-level interface for source creation workflow with **unified dispatch pattern**.

```python
from offline_tools.source_manager import SourceManager

manager = SourceManager()  # Uses BACKUP_PATH automatically

# Create an index (auto-detects source type)
result = manager.create_index("my_wiki")

# Validate source before distribution
validation = manager.validate_source("my_wiki", source_config)
# Returns: has_backup, has_index, has_license, detected_license, suggested_tags
```

### Unified Pipeline Architecture

All source types (ZIM, HTML, PDF) go through the same 4-step pipeline via SourceManager dispatch methods:

| Step | Entry Point | Dispatches To | Output |
|------|-------------|---------------|--------|
| 1. Backup | `scan_backup()` | `_scan_zim_backup()`, `_scan_html_backup()`, `_scan_pdf_backup()` | `backup_manifest.json` |
| 2. Metadata | `generate_metadata()` | `_generate_zim_metadata()`, `_generate_html_metadata()`, `_generate_pdf_metadata()` | `_metadata.json` |
| 3. Index | `create_index()` | `_index_zim()`, `_index_html()`, `_index_pdf()` | `_index.json`, `_vectors.json`, ChromaDB |

**Auto-detection:** `_detect_source_type()` determines the source type based on:
- `.zim` file present -> ZIM
- `pages/` folder present -> HTML
- `.pdf` files present -> PDF

**API Endpoints (all use unified dispatch):**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/useradmin/api/scan-backup` | POST | Scan backup files |
| `/useradmin/api/generate-metadata` | POST | Generate metadata with language filter |
| `/useradmin/api/create-index` | POST | Create index and embeddings |

**DEPRECATED:** `/useradmin/api/zim/index` - Use `/create-index` instead (auto-detects source type)

### Adding New Source Types

To add a new source type (e.g., EPUB), implement these methods in `source_manager.py`:

```python
# 1. Update _detect_source_type() to recognize the new type
def _detect_source_type(self, source_id: str) -> str:
    source_path = self.backup_path / source_id
    if list(source_path.glob("*.epub")):
        return "epub"
    # ... existing detection logic

# 2. Add dispatch case in scan_backup()
def scan_backup(self, source_id: str, ...):
    source_type = self._detect_source_type(source_id)
    if source_type == "epub":
        return self._scan_epub_backup(source_id, ...)
    # ... existing dispatch

# 3. Add dispatch case in generate_metadata()
def generate_metadata(self, source_id: str, ...):
    if source_type == "epub":
        return self._generate_epub_metadata(source_id, ...)

# 4. Add dispatch case in create_index()
def create_index(self, source_id: str, ...):
    if source_type == "epub":
        return self._index_epub(source_id, ...)

# 5. Implement the type-specific methods
def _scan_epub_backup(self, source_id, ...): ...
def _generate_epub_metadata(self, source_id, ...): ...
def _index_epub(self, source_id, ...): ...
```

The unified architecture ensures:
- Consistent progress callbacks and cancellation support
- Automatic checkpoint/resume for long jobs
- Same UI workflow across all source types

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
- BCP 47 language codes with region (e.g., `/pt-br/`, `/zh-hans/`, `/zh-hant/`) - base language extracted
- Title suffixes (e.g., `(Spanish)`, `(Chinese)`, `(Haitian Creole)`)
- Language keywords after separators (e.g., `Solar cooker - Vietnamese`)

Note: Articles with no detectable language marker are included by default (they may be in the target language without explicit marking).

### HTML Backup Offline Browsing

The HTML backup server (`admin/backup_server.py`) enables offline browsing of HTML backup content. When users click on a search result from an HTML source, they browse the archived content locally.

**Routes:**

| Endpoint | Purpose |
|----------|---------|
| `/backup/{source_id}/{path}` | Serve HTML page from backup |
| `/backup/{source_id}` | Serve source index page or listing |
| `/backup/api/sources` | List all available HTML backup sources |

**Features:**

- **Internal Link Rewriting**: Converts relative and absolute links to `/backup/` URLs
- **Dead Site Handling**: Rewrites absolute URLs matching `base_url` to local backup URLs
- **Navigation Button**: Floating "Back to Search" button in bottom-right corner
- **MIME Type Detection**: Serves HTML, images, CSS with correct content types
- **Auto-Index Generation**: Creates a listing page if no index.html exists

**Link Rewriting Examples:**

```
href="/path/page.html"           -> href="/backup/{source_id}/path/page.html"
href="../page.html"              -> href="/backup/{source_id}/page.html"
src="/images/foo.png"            -> src="/backup/{source_id}/images/foo.png"
href="https://example.com/page"  -> href="/backup/{source_id}/page" (if base_url matches)
```

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

**Pinecone Sync API Endpoints (`/useradmin/api/pinecone-...`):**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/pinecone-status` | GET | Get Pinecone connection status and stats |
| `/pinecone-check-source/{source_id}` | GET | Check if source exists in Pinecone (returns vector count) |
| `/pinecone-source/{source_id}` | DELETE | Delete all vectors for a source (global admin only) |
| `/pinecone-compare` | POST | Compare local ChromaDB with Pinecone |
| `/pinecone-sync` | POST | Sync local vectors to Pinecone |

**Local Source Check API (`/useradmin/api/local-source-check/{source_id}`):**

Returns info about existing local vectors for a source before install/index operations.

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

#### Novel Tags Discovery (Global Admin)

When users choose tags for their sources (including novel terms from content analysis), those tags are saved with the source. Global admins can scan all sources to discover tags that users have chosen but which don't exist in `TOPIC_KEYWORDS`.

**UI Location:** Submissions page (Global mode only) -> "Novel Tags Discovery" section

**API Endpoint:** `GET /useradmin/api/discover-novel-tags` (requires Global Admin mode)

**Backend Function:** `SourceManager.discover_novel_used_tags(source_ids=None)`

```python
from offline_tools.source_manager import SourceManager

manager = SourceManager()

# Scan all sources in backup folder
result = manager.discover_novel_used_tags()

# Or scan specific sources
result = manager.discover_novel_used_tags(source_ids=["bitcoin", "ethereum"])

# Result structure:
{
    "novel_tags": {"blockchain": ["bitcoin", "ethereum"], ...},  # tag -> sources
    "known_tags": {"energy": ["solar-guide"], ...},
    "sources_scanned": 15,
    "sources_with_tags": 12,
    "report": "Novel Tags Not in TOPIC_KEYWORDS:\n...",  # formatted string
    "errors": []
}
```

**Workflow:**
1. Local users create sources and choose tags (Step 4 of Source Tools)
2. Tags are saved in `_manifest.json`
3. Global admin scans sources from Submissions page
4. Novel tags displayed with usage counts
5. Global admin manually adds valuable tags to `TOPIC_KEYWORDS` in `source_manager.py`

### Source Validation

A source is ready for distribution when it has 5 status boxes green:

1. **Config** - `_manifest.json` exists with name, license, base_url
2. **Backup** - HTML pages, ZIM file, or PDF files exist
3. **Metadata** - `_metadata.json` exists
4. **Embeddings** - `_vectors.json` created for search
5. **License** - Specified and verified

### Smart Sync (Update vs Replace)

When installing, indexing, or publishing a source that already exists, the system shows a modal asking how to handle the existing vectors:

| Action | Update Mode | Replace Mode |
|--------|-------------|--------------|
| **Install from Cloud** (sources.html) | Add new vectors, keep existing | Delete old vectors first, then install fresh |
| **Local Indexing** (source_tools.html) | Skip already-indexed documents | Force re-index all documents |
| **Publish to Pinecone** (cloud_upload.html) | Upsert new/changed vectors | Delete all source vectors, then push |

**When to use each mode:**

- **Update** (default): Faster, preserves existing work. Use when adding new content or fixing a few documents.
- **Replace**: Clean slate. Use when the source structure changed significantly, or to fix indexing issues.

**Technical implementation:**

1. Before action, system calls check endpoint to get existing vector count:
   - Local: `/useradmin/api/local-source-check/{source_id}`
   - Pinecone: `/useradmin/api/pinecone-check-source/{source_id}`

2. If vectors exist, modal displays count and offers Update/Replace choice

3. For Replace mode, `delete_by_source()` is called before adding new vectors:
   ```python
   # VectorStore and PineconeStore both implement this
   result = store.delete_by_source(source_id)
   # Returns: {"deleted_count": 1234, "batches": 2}
   ```

### Orphaned Source Detection

When a source folder is manually deleted but the source still exists in `_master.json` and/or ChromaDB, it appears as an **orphaned source** in the Source Tools dropdown.

**How it works:**

1. Source list reads from BOTH `_master.json` AND folder scan
2. Sources in master but missing folder are flagged `is_orphaned: true`
3. UI shows `[ORPHANED]` prefix with red color
4. Selecting orphaned source shows warning panel with cleanup button

**Purpose:** Allows proper cleanup of ChromaDB vectors and `_master.json` even when the source folder was deleted outside the admin panel.

**Implementation:**

```python
# source_tools.py - get_local_sources()
# 1. Load sources from _master.json (catches orphaned entries)
for source_id, source_info in master_data.get("sources", {}).items():
    sources_config[source_id] = {
        "name": source_id,
        "tags": source_info.get("topics", []),
        "from_master": True,  # Flag for orphan detection
    }

# 2. Scan folders to supplement with actual data
for source_folder in backup_path.iterdir():
    # ... updates sources_config with real folder data

# 3. Mark orphaned if in master but folder missing
is_orphaned = config_data.get("from_master", False) and not folder_exists
```

### Internal Links (for Visualization)

During metadata generation, internal links between documents are extracted and stored in `_metadata.json`. These are used by the 3D visualization to draw connection lines between related documents.

**Extracted during:**
- `generate_metadata()` for HTML sources (via `extract_internal_links_from_html()`)
- `_generate_zim_metadata()` for ZIM sources

**Storage format in `_metadata.json`:**

```json
{
  "documents": {
    "doc_id_1": {
      "title": "Solar Water Heater",
      "url": "/wiki/Solar_Water_Heater",
      "internal_links": [
        "/wiki/Solar_Energy",
        "/wiki/Water_Heating",
        "/wiki/DIY_Projects"
      ]
    }
  }
}
```

**Visualization usage (`admin/routes/visualise.py`):**

The `build_link_edges()` function reads `internal_links` from `_metadata.json` and creates edges between points in the 3D scatter plot:

```python
def build_link_edges(points, sources, progress_callback=None):
    # Builds URL-to-point-index mapping
    # Reads internal_links from each source's _metadata.json
    # Returns: [{"from": idx1, "to": idx2}, ...]
```

**Note:** Links are only stored if present (to save space). Empty `internal_links` arrays are omitted.

### File Structure in BACKUP_PATH

```
BACKUP_PATH/
|-- _master.json                         # Master source index (optional)
|-- _visualisation.json                  # 3D visualization data (generated)
|-- my_wiki/                             # Source folder
|   |-- _manifest.json                   # Source config (identity + distribution)
|   |-- _metadata.json                   # Document metadata (includes internal_links)
|   |-- _index.json                      # Full content for display
|   |-- _vectors.json                    # Vector embeddings
|   |-- backup_manifest.json             # URL to file mapping
|   |-- pages/                           # HTML backup content
|-- bitcoin.zim                          # ZIM files at backup root
+-- chroma/                              # ChromaDB data
```

---

## Knowledge Map Visualization

The Knowledge Map is a 3D scatter plot visualization of all indexed documents, showing semantic relationships and hyperlink connections.

### How It Works

1. **PCA Reduction**: 1536-dimensional embeddings are reduced to 3D using Principal Component Analysis
2. **Point Coloring**: Documents colored by source for visual grouping
3. **Link Edges**: Connection lines between documents based on internal hyperlinks
4. **Interactive**: Pan, zoom, rotate the 3D space; click points to view document details

### API Endpoints (`/useradmin/api/visualise/...`)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/visualise/status` | GET | Check if visualization data exists, get stats |
| `/visualise/data` | GET | Get full visualization JSON for rendering |
| `/visualise/generate` | POST | Start background job to generate visualization |
| `/visualise/data` | DELETE | Delete visualization data |

### Data Structure (`_visualisation.json`)

```json
{
  "generated_at": "2025-12-09T12:00:00",
  "algorithm": "pca",
  "point_count": 22000,
  "edge_count": 15000,
  "sources": ["appropedia", "bitcoin", "ready_gov"],
  "source_counts": {"appropedia": 13522, "bitcoin": 1812, ...},
  "variance_explained": [0.15, 0.08, 0.05],
  "total_variance": 0.28,
  "points": [
    {"x": 1.2, "y": -0.5, "z": 0.8, "id": "doc_id", "title": "...", "source": "..."}
  ],
  "edges": [
    {"from": 0, "to": 42}
  ]
}
```

### Generation Process

1. Fetch all embeddings from ChromaDB
2. Run PCA to reduce to 3D coordinates
3. Build point list with metadata (title, source, URLs)
4. Build edges from `internal_links` in `_metadata.json`
5. Save to `_visualisation.json` in backup folder

---

## Local Admin Panel

Access at: `http://localhost:8000/useradmin/`

### Features

- **Dashboard** - System status and statistics
- **Sources** - Browse all sources with status boxes, install cloud sources
- **Source Tools** - 5-step wizard for creating/editing sources
- **Visualization** - 3D knowledge map of all indexed documents
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
  "backup_folder": "/path/to/your/backups",
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

**Option 1: Admin Panel Scraper (Recommended)**

The Source Tools page includes a built-in web scraper with link following. This is the easiest way to backup static websites.

1. Go to `/useradmin/` -> Source Tools
2. Select or create a source with `backup_type: html`
3. Enter the site's base URL or sitemap URL
4. Configure scrape settings (see below)
5. Click "Start Scrape"

**Option 2: Browser Save**
- Visit any page and use "Save As" -> "Webpage, Complete"
- Organize saved pages into folders by source

**Option 3: CLI Scrapers**
```bash
python cli/ingest.py scrape mediawiki --url https://wiki.example.org --output ./backups/mywiki
```

**Option 4: HTTrack / wget**
- Use [HTTrack](https://www.httrack.com/) to mirror websites
- Or use wget: `wget -r -l 2 -p https://example.com`

### HTML Backup Scraper (Admin Panel)

The built-in scraper (`offline_tools/backup/html.py`) supports intelligent crawling of static websites.

#### Scrape Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Page Limit** | 100 | Maximum pages to download |
| **Max Depth** | 3 | How many link levels deep to crawl (0 = only starting pages) |
| **Delay (sec)** | 0.5 | Wait time between requests (be nice to servers) |
| **Follow Links** | Yes | Discover new pages by following links on downloaded pages |
| **Include Assets** | No | Download images/CSS (slower, uses more space) |

#### URL Discovery Methods

The scraper tries multiple methods to find pages:

1. **XML Sitemap** (`/sitemap.xml`) - Standard XML format with `<url><loc>` tags
2. **Sitemap Index** - Handles multi-sitemap sites automatically
3. **HTML Sitemap Pages** - Extracts links from HTML pages (e.g., `/SiteMap.htm`)
4. **Link Following** - Discovers new pages by parsing links from downloaded content

**Tip:** Use "Check Sitemap" button to preview page count before starting a full scrape.

#### Breadth-First Crawling

The scraper uses breadth-first search (BFS):
- Downloads all depth-0 pages first (from sitemap/start page)
- Then all depth-1 pages (links found on depth-0 pages)
- Then depth-2, depth-3, etc.

This ensures broad coverage before going deep into any single branch.

#### Resume Support

The scraper tracks what's already backed up:
- Checks `backup_manifest.json` for existing pages
- Skips already-downloaded pages automatically
- Shows "Existing manifest has X pages backed up" on start

To continue a partial scrape, just run again with the same source - it picks up where it left off.

#### Completion Summary

When a scrape finishes, you'll see:
```
--- Scrape Complete ---
New pages saved: 500
Total pages backed up: 1000
Assets downloaded: 715
Errors: 3
URLs still in queue: 402 (set limit to 1402 to get all)
```

The "URLs still in queue" tells you how many discovered pages weren't downloaded due to the page limit. Use this to set the limit for your next run.

#### API Usage

```python
from offline_tools.backup.html import run_backup

result = run_backup(
    backup_path="/path/to/your/backups",  # Set via Settings page or BACKUP_PATH env
    source_id="mysite",
    base_url="https://example.com",
    scraper_type="static",  # or "mediawiki"
    page_limit=500,
    include_assets=False,
    follow_links=True,
    max_depth=3,
    request_delay=0.5,
    sitemap_url="https://example.com/sitemap.xml"  # optional
)

# Result includes:
# - pages_saved: New pages downloaded this run
# - total_backed_up: Total pages in manifest
# - queued_remaining: URLs discovered but not downloaded
# - errors: List of failed URLs
```

#### Output Structure

```
BACKUP_PATH/mysite/
|-- backup_manifest.json    # URL-to-file mapping
|-- index.html              # Auto-generated browse page
|-- pages/                  # Downloaded HTML files
|   |-- Projects_Solar.html
|   |-- Projects_Water.html
|   +-- ...
+-- assets/                 # Images, CSS (if enabled)
    |-- logo.png
    +-- style.css
```

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

### Large ZIM Indexing Issues

When indexing large ZIM files (400k+ articles), several issues can occur:

**Issue: Only a small number of documents indexed despite large source**

Symptoms:
- Metadata generation works (e.g., 490k docs in 35 min)
- Indexing completes quickly (3 min) with only ~430 docs
- All JSON files show the reduced count

Root causes identified:
1. **Limit defaulting to 1000**: The `indexLimit` input field has `max="50000"`. If the "Index All" button's `dataset.docCount` isn't read correctly, limit defaults to 1000.
2. **Metadata overwrite bug** (FIXED): After indexing, `save_all_outputs()` was overwriting `_metadata.json` with only the indexed documents, destroying the full document list from "Generate Metadata".

Fix applied:
- Added `skip_metadata_save` parameter to `save_all_outputs()` in [indexer.py](offline_tools/indexer.py)
- When `use_metadata=True`, the existing metadata is preserved

To recover:
1. Re-run "Generate Metadata" to restore full document list
2. Re-run "Index All" - metadata will now be preserved

**Planned improvements:**
- Job checkpointing for resume after interruption
- Batch processing with intermediate saves
- Better debug logging for limit values

---

## Legacy Migration Code

Most legacy fallback code has been removed. The following locations still contain legacy handling that may be useful:

**Remaining legacy code (intentionally kept):**

| File | Lines (approx) | Description | Reason to Keep |
|------|----------------|-------------|----------------|
| `offline_tools/backup/html.py` | 61-88 | Auto-migrates old manifest formats | Handles old cloud downloads |
| `offline_tools/backup/html.py` | 730-733 | Backup manifest path check | Used by resume detection |
| `offline_tools/source_manager.py` | 673-771 | `cleanup_redundant_files()` | Actively cleans legacy files |
| `offline_tools/source_manager.py` | 1735-1739 | Manifest path in get_sync_status | May handle edge cases |
| `offline_tools/vectordb/pinecone_store.py` | 441 | Legacy _master.json path | Cloud compatibility |

**Cleaned up (removed 2024-12):**
- `source_manager.py`: `suggest_tags()`, `analyze_metadata_for_tags()`, `discover_novel_used_tags()`, `detect_license()`
- `packager.py`: `load_metadata()`, `get_source_sync_status()`, ZIM root folder check
- `indexer.py`: HTMLBackupIndexer manifest candidates
- `vectordb/metadata.py`: `_get_source_file()` legacy path

**Current file naming (v3 schema):**
- `_manifest.json` - source identity and config
- `_metadata.json` - document metadata
- `_index.json` - full content index
- `_vectors.json` - embeddings
- `backup_manifest.json` - URL to file mapping

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

## Job Cancellation and Resume

The Jobs page includes a "Stop" button to cancel running jobs. This is a **soft cancel** that saves progress for later resumption.

**Current behavior:**
- Job status updates to "cancelled" immediately in the UI
- The background thread checks for cancellation periodically
- Progress is saved to checkpoint file before exiting
- Interrupted jobs appear in the "Interrupted Jobs" section on the Jobs page
- Users can click "Resume" to continue from where they left off

**Jobs that support graceful cancellation with resume:**
- Generate Metadata (ZIM) - checkpoints every 60 seconds or 2000 articles
- Create Index (ZIM) - checkpoints every 60 seconds or 500 documents
- HTML/PDF indexing uses incremental mode (automatically skips already-indexed docs)

**Implementation:**
```python
# Inside the indexing loop
if cancel_checker and cancel_checker():
    # Save checkpoint before exiting
    save_checkpoint(checkpoint)
    with open(partial_path, 'w') as f:
        json.dump({"indexed_doc_ids": list(indexed_doc_ids), ...}, f)
    return {"success": False, "cancelled": True, "indexed_count": indexed}
```

---

## Source Processing Pipeline

Sources go through an 8-step processing pipeline before they're ready for distribution. The wizard UI groups these into 5 visible steps.

### Pipeline Steps

| Step | Substep | Action | Output File(s) | Completion Marker |
|------|---------|--------|----------------|-------------------|
| 1 | 1.1 | Get backup files (folder/download/scrape) | `pages/` or `.zim` | Files exist |
| 1 | 1.2 | Scan backup / Generate backup manifest | `backup_manifest.json` | `pages` dict populated |
| 2 | 2.1 | Configure source (name, description, license) | `_manifest.json` | `source_id`, `name`, `license` set |
| 3 | 3.1 | Generate metadata (with language filter) | `_metadata.json` | `documents` dict, `document_count > 0` |
| 3 | 3.2 | Set base URL | `_manifest.json` | `base_url` set |
| 4 | 4.1 | Run sample URL tests | *(no file)* | Visual check only |
| 5 | 5.1 | Create full index + embeddings | `_index.json`, `_vectors.json` | `vectors` dict, `document_count > 0` |
| 6 | 6.1 | Suggest tags from index content | *(in memory)* | Tags suggested |
| 6 | 6.2 | Save tags | `_manifest.json` | `tags` array populated |
| 7 | 7.1 | Final URL test | *(no file)* | Visual check |
| 8 | 8.1 | Final validation | *(reads all files)* | `is_valid = True` |

### File Schema Summary

| File | Created By | Key Fields | Required |
|------|------------|------------|----------|
| `pages/` or `.zim` | User/Scraper | Backup content | YES |
| `backup_manifest.json` | Scan Backup | `pages`, `assets`, `base_url` | No |
| `_manifest.json` | Configure | `source_id`, `name`, `license`, `tags`, `base_url` | YES |
| `_metadata.json` | Generate Metadata | `documents`, `document_count`, `language_filter` | YES |
| `_index.json` | Create Index | `documents` (full content), `source_metadata_hash` | YES |
| `_vectors.json` | Create Index | `vectors`, `document_count`, `embedding_model`, `dimensions` | YES |

### Metadata vs Index Separation

**Important:** `_metadata.json` and `_index.json` serve different purposes:

- `_metadata.json`: Document listing from backup scan (with language filter). Created by "Generate Metadata".
- `_index.json`: Full content for indexed documents. Created by "Create Index".

To prevent overwriting issues, `_index.json` includes a `source_metadata_hash` field that references the `_metadata.json` it was built from. Validation compares these to detect mismatches.

```json
// _index.json header
{
  "schema_version": 3,
  "source_id": "my_source",
  "source_metadata_hash": "abc123...",  // Hash of _metadata.json used
  "document_count": 5269,
  "created_at": "2025-12-07T00:00:00"
}
```

### Validation Requirements

```
is_valid = True requires:
  - has_backup = True
  - has_manifest = True
  - has_metadata_file = True (document_count > 0)
  - has_vectors_file = True (document_count > 0)
  - has_license = True (license != "Unknown")
  - coverage >= 50% (vectors_doc_count / metadata_doc_count)

production_ready = is_valid AND:
  - has_tags = True
  - base_url is set
```

**Note:** `has_tags = False` only blocks `production_ready`, not `is_valid`. Users can work locally without tags.

---

## Job Checkpoint System

**STATUS: IMPLEMENTED** (Generate Metadata + Create Index for ZIM)

Long-running jobs save periodic checkpoints to allow resuming after interruption.

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

**Analysis Notes:**

- **Download from Cloud** already has resume built-in: checks if local file exists with matching size before downloading
- **Upload to Cloud** uses atomic per-file operations (ZIM = 1 file, HTML = zip then upload)
- **Create Index** uses Incremental Indexing - processes in batches of 100, persists after each batch, skips already-indexed docs on resume

**Incremental Indexing (Create Index):**
- All indexers (ZIM, HTML, PDF) use `VectorStore.add_documents_incremental()`
- Queries existing doc IDs, skips already-indexed, processes in batches
- If interrupted, just re-run - previously indexed docs are skipped automatically
- No checkpoint files needed - ChromaDB is the checkpoint

### Checkpoint Storage

```
BACKUP_PATH/_jobs/
  {source_id}_{job_type}.checkpoint.json    # Checkpoint state
  {source_id}_{job_type}.partial.json       # Partial work file (full documents)
```

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

### Partial File Structure (Full Document Backup)

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

### Checkpoint Behavior

- **Save frequency:** Every 60 seconds OR every 2000 articles (whichever first)
- **On success:** Delete checkpoint file and partial file
- **On failure/interruption:** Keep files (allows resume)
- **Stale checkpoints:** Manual cleanup via Jobs page, or auto-delete after 7 days
- **Atomic writes:** Write to temp file, then rename (prevents corruption)

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

## Global Admin Review Process

When submissions arrive for review, the global admin performs these checks before publishing to Pinecone/R2:

### Review Checklist

| Check | Action | Notes |
|-------|--------|-------|
| Embedding dimensions | Re-index with OpenAI 1536-dim if needed | Submissions may use 384-dim local |
| Tags | Verify/add appropriate tags | Required for discovery |
| URLs | Verify base_url works | Test sample links |
| License | Verify license accuracy | Check source site |
| Content quality | Spot check articles | Look for spam, duplicates |

### Re-indexing Workflow

```
Submission arrives (384-dim local vectors)
    |
    v
Admin opens in Source Tools
    |
    v
Check _vectors.json dimensions
    |
    +-- 384-dim --> Click "Force Re-index" with OpenAI
    |               (creates new 1536-dim vectors)
    |
    +-- 1536-dim --> Skip re-indexing
    |
    v
Review tags, license, URLs
    |
    v
Run final validation
    |
    v
Approve --> Publish to Pinecone + R2
```

### Admin-Only Checks (Future)

These will be added to a dedicated Admin Review page:

- [ ] Dimension check with one-click re-index
- [ ] Tag suggestion review
- [ ] License verification workflow
- [ ] URL batch testing
- [ ] Content sampling
- [ ] Publish approval queue

---

## Known Issues and Fixes

### Metadata/Index Count Mismatch

**Problem:** `_metadata.json` shows 18k docs, `_vectors.json` shows 5k. Validation fails with "coverage 28%".

**Cause:** Different text extraction methods:
- Metadata generation uses simple regex (extracts more text, more docs pass filter)
- Indexing uses BeautifulSoup (extracts main content only, fewer docs pass filter)

**Fix:** Unify extraction to use "lenient BeautifulSoup" for both:
```python
def extract_text_lenient(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    # Remove junk tags but don't require main/article container
    for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
        tag.decompose()
    text = soup.get_text(separator=' ', strip=True)
    return ' '.join(line.strip() for line in text.splitlines() if line.strip())
```

### Token Limit Errors During Indexing (FIXED Dec 2025)

**Problem:** OpenAI embedding fails with "8581 tokens requested, 8192 max".

**Cause:** Text truncation (20k chars) can still exceed 8192 tokens for dense text.

**Fix:** Added intelligent chunking in `EmbeddingService._embed_with_chunking()`:
1. Try to embed full text (up to 32k chars)
2. If token limit error, split text in half at sentence boundary
3. Recursively embed each half (up to 8 chunks max)
4. Average the embeddings to produce final vector

All long articles now get proper embeddings instead of zero vectors.

### Pinecone Sync "Pushed: 0 documents" (FIXED Dec 2025)

**Problem:** After indexing, Pinecone sync shows "Pushed: 0 documents" despite finding thousands to push.

**Cause:** Document ID mismatch between metadata generation and indexer:
- Metadata used sequential IDs like `zim_0`, `zim_1`
- Indexer used hash IDs like `md5(source_id:url)`
- Sync tried to fetch `zim_0` from ChromaDB but ChromaDB had hash IDs

**Fix:** Updated metadata generation to use same hash ID format as indexer:
```python
# source_manager.py and packager.py now use:
doc_id = hashlib.md5(f"{source_id}:{url}".encode()).hexdigest()
```

**Recovery:** Regenerate metadata for affected sources, then retry Pinecone sync.

---

## Search Result Diversity

Search results are re-ranked to ensure diversity across sources, preventing any single source from dominating results.

### How It Works

1. **Search Phase**: Retrieves 15 candidate results from vector DB
2. **Doc Type Prioritization**: Boosts guides over articles (configurable)
3. **Source Diversity**: Limits to 2 results per source, then backfills
4. **Final Output**: Returns top 5 diverse results to user

### Implementation

```python
# app.py - ensure_source_diversity()
def ensure_source_diversity(articles, max_per_source=2, total_results=5):
    # Group by source
    # Round-robin: take up to max_per_source from each source
    # Backfill remaining slots with highest-scored unused articles
```

**Example behavior:**
- Input: `[wiki1, wiki2, wiki3, wiki4, ready1, ready2, appro1]`
- Output: `[wiki1, ready1, appro1, wiki2, ready2]`

### User Override

Users can still filter to a single source using the source filter in chat UI. When only one source is selected, all 5 results come from that source.

---

## Content Filtering

### Minimum Content Length

Articles with less than 100 characters of extracted text are filtered out during metadata generation. This removes stub pages, redirects, and other low-value content.

**Configured in:**
- `source_manager.py:1434` - Filter during ZIM metadata generation
- `zim_utils.py:106` - Default parameter for ZIM inspection

**Previous value:** 50 characters
**Current value:** 100 characters

---

*Last Updated: December 9, 2025*
