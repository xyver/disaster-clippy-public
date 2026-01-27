# Source Tools

This document covers working with sources: creation, indexing, scraping, and the unified pipeline.

---

## Table of Contents

1. [SourceManager](#sourcemanager)
2. [Unified Pipeline Architecture](#unified-pipeline-architecture)
3. [Indexers](#indexers)
4. [ZIM Tools](#zim-tools)
5. [HTML Backup Scraper](#html-backup-scraper)
6. [PDF Processing](#pdf-processing)
7. [Tag System](#tag-system)
8. [Source Processing Pipeline](#source-processing-pipeline)
9. [File Structure](#file-structure)

---

## SourceManager

High-level interface for source creation workflow with **unified dispatch pattern**.

**File:** `offline_tools/source_manager.py`

```python
from offline_tools.source_manager import SourceManager

manager = SourceManager()  # Uses BACKUP_PATH automatically

# Create an index (auto-detects source type)
result = manager.create_index("my_wiki")

# Validate source before distribution
validation = manager.validate_source("my_wiki", source_config)
# Returns: has_backup, has_index, has_license, detected_license, suggested_tags
```

---

## Unified Pipeline Architecture

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

---

## Indexers

**File:** `offline_tools/indexer.py`

| Class | Source Type | Input |
|-------|-------------|-------|
| `HTMLBackupIndexer` | HTML websites | Backup folder with pages/ |
| `ZIMIndexer` | ZIM archives | .zim file |
| `PDFIndexer` | PDF documents | PDF file or folder |

### Incremental Indexing

All indexers use `VectorStore.add_documents_incremental()`:
- Queries existing doc IDs, skips already-indexed
- Processes in batches of 100
- Persists after each batch
- If interrupted, just re-run - previously indexed docs are skipped automatically
- No checkpoint files needed - ChromaDB is the checkpoint

---

## ZIM Tools

### ZIM Metadata Extraction

When indexing ZIM files, metadata is automatically extracted from the ZIM header fields:

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

Note: Articles with no detectable language marker are included by default.

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

**Inspection Output:**
- Header metadata (title, description, creator, license, etc.)
- Content type detection (website, video, pdf, mixed)
- Mimetype and namespace distribution
- Text length analysis (indexable vs skipped articles)
- Sample articles with text previews
- Recommendations for indexing parameters

### ZIM Offline Browsing

The ZIM server (`admin/zim_server.py`) enables seamless offline browsing of ZIM archive content.

**Routes:**

| Endpoint | Purpose |
|----------|---------|
| `/zim/{source_id}/{path}` | Serve article content from ZIM |
| `/zim/{source_id}` | Serve ZIM main page or index |
| `/zim/api/sources` | List all available ZIM sources |

**Features:**
- URL Index Caching: Builds an in-memory index on first access for O(1) article lookups
- Internal Link Rewriting: Converts relative and absolute links to `/zim/` URLs
- Dead Site Handling: Rewrites absolute URLs matching `base_url` to local ZIM URLs
- Navigation Button: Minimal floating "Back to Search" button

---

## HTML Backup Scraper

The built-in scraper (`offline_tools/backup/html.py`) supports intelligent crawling of static websites.

### Scrape Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Page Limit** | 100 | Maximum pages to download |
| **Max Depth** | 3 | How many link levels deep to crawl (0 = only starting pages) |
| **Delay (sec)** | 0.5 | Wait time between requests (be nice to servers) |
| **Follow Links** | Yes | Discover new pages by following links on downloaded pages |
| **Include Assets** | No | Download images/CSS (slower, uses more space) |

### URL Discovery Methods

The scraper tries multiple methods to find pages:

1. **XML Sitemap** (`/sitemap.xml`) - Standard XML format with `<url><loc>` tags
2. **Sitemap Index** - Handles multi-sitemap sites automatically
3. **HTML Sitemap Pages** - Extracts links from HTML pages (e.g., `/SiteMap.htm`)
4. **Link Following** - Discovers new pages by parsing links from downloaded content

### Breadth-First Crawling

The scraper uses breadth-first search (BFS):
- Downloads all depth-0 pages first (from sitemap/start page)
- Then all depth-1 pages (links found on depth-0 pages)
- Then depth-2, depth-3, etc.

This ensures broad coverage before going deep into any single branch.

### Resume Support

The scraper tracks what's already backed up:
- Checks `backup_manifest.json` for existing pages
- Skips already-downloaded pages automatically
- Shows "Existing manifest has X pages backed up" on start

To continue a partial scrape, just run again with the same source.

### Completion Summary

When a scrape finishes:
```
--- Scrape Complete ---
New pages saved: 500
Total pages backed up: 1000
Assets downloaded: 715
Errors: 3
URLs still in queue: 402 (set limit to 1402 to get all)
```

### HTML Backup Offline Browsing

The HTML backup server (`admin/backup_server.py`) enables offline browsing of HTML backup content.

**Routes:**

| Endpoint | Purpose |
|----------|---------|
| `/backup/{source_id}/{path}` | Serve HTML page from backup |
| `/backup/{source_id}` | Serve source index page or listing |
| `/backup/api/sources` | List all available HTML backup sources |

**Features:**
- Internal Link Rewriting: Converts relative and absolute links to `/backup/` URLs
- Dead Site Handling: Rewrites absolute URLs matching `base_url` to local backup URLs
- Navigation Button: Floating "Back to Search" button in bottom-right corner
- MIME Type Detection: Serves HTML, images, CSS with correct content types
- Auto-Index Generation: Creates a listing page if no index.html exists

---

## PDF Processing

The PDF pipeline supports building codes, technical manuals, and other structured documents.

**File:** `offline_tools/scraper/pdf.py`

### Processing Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **sectioned** | Preserves section headers and page references | Building codes, manuals with clear sections |
| **chunked** | Splits by character count with overlap | Long documents without clear structure |
| **full** | Keeps entire document as one chunk | Short documents, reference sheets |

### Sectioned Extraction

The `sectioned` mode is optimized for building codes and technical standards:

- **Header Detection:** Identifies section headers (numbered patterns like "1.2.3", "Section 5", "Chapter 3")
- **Page Numbers:** Each chunk includes page range for citations (e.g., "pp. 17-18")
- **Section Preservation:** Content grouped by logical sections, not arbitrary character limits
- **Title Generation:** Auto-generates titles from section headers

**Example output:**
```
Section: 3.2.1 Roof Deck Attachment
Pages: 17-18
Content: Roof deck attachment shall follow...
```

### Page Range Filtering

Skip front matter (TOC, title pages) and back matter (appendices, index):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `start_page` | First page to process | 1 |
| `end_page` | Last page to process (0 = all) | 0 |

**Admin UI:** Set in Step 2 "PDF Import" job parameters

**CLI:**
```bash
python cli/ingest.py pdf process my-source --sectioned --start-page 5 --end-page 100
```

### R2 Public URL Integration

PDF citations need accessible URLs for online users. The system auto-constructs R2 public URLs:

**Offline URL:** `/backup/source-id/document.pdf#page=17`
**Online URL:** `https://pub-xxx.r2.dev/backups/source-id/document.pdf#page=17`

The `#page=N` fragment enables direct navigation to the cited page in the browser.

**Configuration:** Set `R2_PUBLIC_URL` in `.env`:
```
R2_PUBLIC_URL=https://pub-xxx.r2.dev
```

### PDF Offline Browsing

PDFs are served from the backup folder for offline access:

| Endpoint | Purpose |
|----------|---------|
| `/backup/{source_id}/{filename}.pdf` | Serve PDF file |
| `/backup/{source_id}/{filename}.pdf#page=N` | Direct to specific page |

---

## Tag System

Sources can be tagged for categorization and search filtering. Tags are either:
- **Auto-suggested** during indexing based on content keywords
- **Manual** - set by user in `_manifest.json`

### Available Tag Categories

Defined in `SourceManager.TOPIC_KEYWORDS`:

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

### Novel Tags Discovery (Global Admin)

When users choose tags for their sources (including novel terms from content analysis), those tags are saved with the source. Global admins can scan all sources to discover tags that users have chosen but which don't exist in `TOPIC_KEYWORDS`.

**UI Location:** Submissions page (Global mode only) -> "Novel Tags Discovery" section

**API Endpoint:** `GET /useradmin/api/discover-novel-tags` (requires Global Admin mode)

---

## Source Processing Pipeline

Sources go through a multi-step processing pipeline before they're ready for distribution.

### Pipeline Steps

| Step | Action | Output File(s) | Completion Marker |
|------|--------|----------------|-------------------|
| 1.1 | Get backup files (folder/download/scrape) | `pages/` or `.zim` | Files exist |
| 1.2 | Scan backup / Generate backup manifest | `backup_manifest.json` | `pages` dict populated |
| 2.1 | Configure source (name, description, license) | `_manifest.json` | `source_id`, `name`, `license` set |
| 3.1 | Generate metadata (with language filter) | `_metadata.json` | `documents` dict, `document_count > 0` |
| 3.2 | Set base URL | `_manifest.json` | `base_url` set |
| 4.1 | Detect language | `_manifest.json` | `language` field set |
| 5.1 | Create full index + embeddings | `_index.json`, `_vectors.json` | `vectors` dict, `document_count > 0` |
| 6.1 | Suggest tags from index content | *(in memory)* | Tags suggested |
| 6.2 | Save tags | `_manifest.json` | `tags` array populated |
| 7.1 | Human verification | `_manifest.json` | `license_verified`, `links_verified_*` = true |
| 8.1 | Final validation | *(reads all files)* | `can_submit = True` |

### Metadata vs Index Separation

**Important:** `_metadata.json` and `_index.json` serve different purposes:

- `_metadata.json`: Document listing from backup scan (with language filter). Created by "Generate Metadata".
- `_index.json`: Full content for indexed documents. Created by "Create Index".

To prevent overwriting issues, `_index.json` includes a `source_metadata_hash` field that references the `_metadata.json` it was built from.

---

## File Structure

### In BACKUP_PATH

```
BACKUP_PATH/
|-- _master.json                         # Master source index (optional)
|-- _visualisation.json                  # 3D visualization data (generated)
|-- my_wiki/                             # Source folder (HTML)
|   |-- _manifest.json                   # Source config (identity + distribution)
|   |-- _metadata.json                   # Document metadata (includes internal_links)
|   |-- _index.json                      # Full content for display
|   |-- _vectors.json                    # Vector embeddings (1536-dim)
|   |-- _vectors_768.json                # Vector embeddings (offline)
|   |-- _validation_status.json          # Cached validation result
|   |-- backup_manifest.json             # URL to file mapping
|   |-- pages/                           # HTML backup content
|-- fortified-2025/                      # Source folder (PDF)
|   |-- _manifest.json                   # Source config (backup_type: "pdf")
|   |-- _metadata.json                   # Sections/chunks extracted from PDFs
|   |-- _index.json                      # Full section content
|   |-- _vectors.json                    # Vector embeddings
|   |-- 2025-FORTIFIED-Home-Standard.pdf # Original PDF file(s)
|-- bitcoin.zim                          # ZIM files at backup root
+-- chroma/                              # ChromaDB data
```

### File Schema Summary

| File | Created By | Key Fields | Required |
|------|------------|------------|----------|
| `pages/` or `.zim` | User/Scraper | Backup content | YES |
| `backup_manifest.json` | Scan Backup | `pages`, `assets`, `base_url` | No |
| `_manifest.json` | Configure | `source_id`, `name`, `license`, `tags`, `base_url` | YES |
| `_metadata.json` | Generate Metadata | `documents`, `document_count`, `language_filter` | YES |
| `_index.json` | Create Index | `documents` (full content), `source_metadata_hash` | YES |
| `_vectors.json` | Create Index | `vectors`, `document_count`, `embedding_model`, `dimensions` | YES |

### Internal Links (for Visualization)

During metadata generation, internal links between documents are extracted and stored in `_metadata.json`. These are used by the 3D visualization to draw connection lines between related documents.

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

---

## Related Documentation

- [Validation System](validation.md) - Source validation gates and checks
- [Jobs System](jobs.md) - Background job processing for indexing
- [Architecture](architecture.md) - System design and data flow

---

*Last Updated: January 2026*
