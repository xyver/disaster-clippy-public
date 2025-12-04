# Disaster Clippy Refactoring Plan v3

**Created:** December 3, 2025
**Goal:** Merge two repos into one unified codebase

---

## Development Philosophy

**No backward compatibility.** This is a first-version codebase in active development. Old data formats, file names, and import paths do not need to be supported. If refactoring breaks something, restore from backup and fix forward. Do not waste time on migration scripts or fallback logic for earlier experiments.

---

## Executive Summary

**Problem:** Two repos (`disaster-clippy` private, `disaster-clippy-public`) evolved separately with duplicate code, different admin UIs (Streamlit vs FastAPI), and diverging features.

**Solution:** Use `disaster-clippy-public` as the base (it has more complete code), port missing features from private repo, archive the private repo.

**Outcome:** Single repo with mode switching (`ADMIN_MODE=local|global`) controlling feature access.

---

## Repository Relationship

### Private Repo (disaster-clippy) = Global Admin
- Hosts the main public-facing chat app (Railway)
- Connects to official R2 backups folder (read/write)
- Connects to Pinecone (the global vector DB for chat users)
- Receives submissions from local admins via R2 submissions folder
- Has final authority on what becomes an "official source"
- No localadmin UI - just the chat interface for end users

### Public Repo (disaster-clippy-public) = Local Admin
- Users run their own chat app locally
- Full localadmin settings panel
- Can create/edit their own sources locally
- Can download sources from global R2 backups (read-only)
- Can submit sources to R2 submissions folder (write-only)
- Can optionally set up their own cloud storage (future: other providers beyond R2)

---

## Data Flow

```
LOCAL ADMIN                         GLOBAL ADMIN
-----------                         ------------

[Local WIP]                         [Global WIP]
    |                                    ^
    v                                    |
[Validation] -----> [Submissions] ------>+
    |                (R2 folder)         |
    v                                    v
[Personal Cloud]                    [Validation]
(optional backup)                        |
                                         v
                                    [Official Cloud]
                                    (R2 backups/ + Pinecone)
```

Submissions folder = Local admin's "PR" to Global admin's "WIP"

---

## R2 Bucket Structure

```
r2-bucket/
  backups/
    chromadb/              # Pinecone backup (just in case)
    _master.json           # Global source index (auto-updated)
    {source_id}/           # Individual source packages (flat structure)
      _manifest.json       # Source identity (tiny, never grows)
      _metadata.json       # Document lookup table (small, for diffing)
      _index.json          # Full content for scanning/display
      _vectors.json        # Embeddings for ChromaDB/Pinecone
      backup_manifest.json # URL to file mapping (HTML sources only)
      pages/ or *.zim or *.pdf

  submissions/
    {source_id}/           # Pending packages (same structure as above)
```

**NOTE on _index.json vs _vectors.json:**
- `_index.json` contains full document content for quick scanning and display
- `_vectors.json` contains only the embedding vectors for ChromaDB/Pinecone search
- This separation allows efficient vector operations without loading full content

For hierarchical sources (large collections like Appropedia) - DEFERRED:
```
r2-bucket/
  backups/
    {collection_id}/
      _manifest.json       # Collection identity
      _master.json         # Index of child sources
      backup_manifest.json # SHARED - all URL->file mappings
      pages/               # SHARED - all backup files
      chroma/              # SHARED - combined vector DB (optional)

      {child_source}/      # Child source (flat structure)
        _manifest.json
        _metadata.json
        _index.json
        _vectors.json
```

---

## Cloud Permissions

| Action | Global Admin | Local Admin |
|--------|-------------|-------------|
| Read from R2 backups | Yes | Yes |
| Write to R2 backups | Yes | No |
| Read from R2 submissions | Yes | No |
| Write to R2 submissions | No | Yes |
| Own cloud storage | N/A | Future feature |

---

## Unified Admin UI Structure

Both admins share the same codebase with mode switching for extra features.

### Entry Points

```bash
# Local admin (default)
python app.py

# Global admin
ADMIN_MODE=global python app.py
```

### Page Structure

```
SHARED PAGES:
  /dashboard          - Local: backup folder stats | Global: R2 + Pinecone stats
  /settings           - Backup path, cloud config
  /sources            - Source Packs view with tabs/filters
  /sources/tools      - Indexing, packaging, validation tools
  /sources/create     - Unified backup creation wizard
  /jobs               - Job manager

LOCAL ADMIN ONLY:
  /cloud              - Personal cloud + Submit to global (combined)

GLOBAL ADMIN ONLY:
  /submissions        - Review queue (incoming from local admins)
  /pinecone           - Pinecone settings and sync operations
```

### Dashboard Content

| Local Admin Dashboard | Global Admin Dashboard |
|----------------------|------------------------|
| Backup folder path | R2 connection status |
| Total sources | R2 storage stats |
| Total documents | Pinecone connection status |
| Storage used | Pinecone vector count |
| Last indexed | Pending submissions count |
| Connection mode | Recent activity log |

---

## Validation System

Validation is implemented in `sourcepacks/source_manager.py` with two levels:

### Basic Validation (`validate_source`)

Checks for distribution readiness:
- Has backup files (HTML/ZIM/PDF)
- Has metadata file (document list)
- Has embeddings file (for offline search)
- Has license specified (not "Unknown")
- Has tags for discoverability

Auto-detection features:
- License detection from page content (CC-BY-SA, MIT, etc.)
- Tag suggestion from headings and content keywords

### Production Validation (`validate_for_production`)

Stricter checks for R2/Pinecone upload:
- All basic validation must pass
- Embeddings file must have actual content (not empty)
- License must be specified (blocks upload if "Unknown")

### Validation Result Fields

```python
@dataclass
class ValidationResult:
    is_valid: bool              # Overall pass/fail
    source_id: str
    has_backup: bool            # HTML/ZIM/PDF files exist
    has_index: bool             # Can be searched
    has_metadata: bool          # Document list exists
    has_embeddings: bool        # Vector file exists
    has_license: bool           # License specified
    license_verified: bool      # Manually verified
    has_tags: bool              # Tags for discovery
    production_ready: bool      # Ready for R2/Pinecone
    issues: List[str]           # Blocking problems
    warnings: List[str]         # Non-blocking suggestions
    detected_license: str       # Auto-detected license
    suggested_tags: List[str]   # Auto-suggested tags
    schema_version: int         # Schema version
```

---

## Submission Workflow

### Local Admin Submitting

1. Create/edit source locally
2. Run validation (must pass all checks)
3. Upload to R2 submissions folder
4. Wait for global admin review

### Global Admin Reviewing

1. See new submission in /submissions page
2. Download and inspect
3. Run validation
4. Optionally edit/improve (becomes WIP)
5. Approve -> runs final validation -> copies to R2 backups
6. Manual sync to Pinecone
7. Rejection sends message back to submitter

---

## Phase 0: Data Integrity Fixes (Before Restructuring)

These fixes ensure the scraper -> indexer -> search pipeline works reliably. Do these BEFORE moving files around.

### 0.0 Search Result Diversity (Source Balancing)

**Problem:** Large sources (e.g., 135 PDF chunks) dominate search results, drowning out smaller but potentially more relevant sources (e.g., 8 Substack articles). When searching "the barracks", all results come from PDF chunks even though we have a Substack source about that topic.

**Current behavior:**
- `search_articles()` returns top N results by similarity score only
- `prioritize_results_by_doc_type()` adjusts for doc_type (guide vs article) but NOT source diversity
- No balancing between sources

**Desired behavior:**
- Initial search should return results from multiple sources to give user a broad overview
- As conversation continues, user can dive deeper into specific sources
- Exact matches (title contains query) should be boosted regardless of source size

**Fix options:**

1. **Source-aware re-ranking (recommended):**
```python
def diversify_results(articles: List[dict], max_per_source: int = 2) -> List[dict]:
    """
    Re-rank results to ensure diversity across sources.
    Takes top N results but limits each source to max_per_source items.
    """
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

2. **Title/exact match boosting:**
```python
def boost_exact_matches(articles: List[dict], query: str) -> List[dict]:
    """Boost articles where query appears in title."""
    query_lower = query.lower()
    for article in articles:
        title = article.get("metadata", {}).get("title", "").lower()
        if query_lower in title:
            article["score"] += 0.3  # Significant boost for title match
    return sorted(articles, key=lambda x: x["score"], reverse=True)
```

3. **Fetch more, then diversify:**
```python
# In chat endpoint, fetch more results then filter
articles = search_articles(message, n_results=25)  # Was 10
articles = boost_exact_matches(articles, message)
articles = diversify_results(articles, max_per_source=2)
articles = articles[:5]  # Return top 5 after diversification
```

**Files to modify:**
- `app.py:560-564` - Add diversification step after search
- `app.py:1245` - Update `prioritize_results_by_doc_type()` or create new function

**Testing:**
- Search "the barracks" should return Substack results (exact match) alongside PDFs
- Search "solar cooker" should return results from multiple wiki sources

**Future Investigation: Source vs Chunk Granularity**

The current issue highlights a deeper question about how we define "source" for weighting purposes:

| Content Type | Current Behavior | Question |
|--------------|------------------|----------|
| PDF collection | 135 chunks, each competes equally | Should whole PDF be 1 "source" for diversity? |
| Substack | 8 articles, each competes equally | Should each article be its own "source"? |
| Wiki | Many pages, each competes equally | Pages are naturally separate topics |
| Large PDF (multi-topic) | All chunks = 1 source | Should it be split into topic-based sources? |

The right answer likely depends on content structure:
- A PDF handbook covering 10 topics might deserve 10x the representation of a single blog post
- But a PDF with 100 pages on ONE topic shouldn't drown out other sources
- Blog posts are naturally topic-separated (1 post = 1 topic)
- Wiki pages are naturally topic-separated (1 page = 1 topic)

**Possible approaches (for future):**
1. Weight by unique topics covered, not chunk count
2. Cluster chunks by topic before diversity filtering
3. Add "topic breadth" metadata to sources during indexing
4. Let users configure source importance/weights

This needs more thought - parking for later investigation.

### 0.1 Normalize URLs to Full Paths

**Problem:** URLs stored inconsistently - some relative (`/wiki/Page`), some absolute (`https://...`).

**Fix in `indexer.py`:** During indexing, always construct full URLs:
```python
# Get base_url from source metadata
base_url = source_metadata.get("base_url", "").rstrip("/")

# Normalize URL
if url.startswith("/"):
    url = base_url + url
elif not url.startswith("http"):
    url = base_url + "/" + url
```

**Files:** `offline_tools/indexer.py:749-761`

### 0.2 Require base_url in Source Metadata

**Problem:** `_source.json` doesn't always have `base_url`; code falls back to hardcoded mappings in `app.py:1174-1186`.

**Fix:**
1. Make `base_url` required in `SourceMetadata` schema (no default empty string)
2. Scraper must populate `base_url` when creating source
3. Remove hardcoded fallback mappings from `app.py`

**Files:**
- `offline_tools/schemas.py:84` - remove default
- `offline_tools/html_backup.py` - ensure base_url written
- `offline_tools/substack_backup.py` - ensure base_url written
- `app.py:1174-1186` - remove hardcoded mappings after sources fixed

### 0.3 Validate URLs During Indexing

**Problem:** Malformed/empty URLs pass through silently.

**Fix in `indexer.py`:** Add validation before adding to documents:
```python
if not url or url == "":
    logger.warning(f"Empty URL for document {doc_id}, skipping")
    continue
if not url.startswith(("http://", "https://", "zim://")):
    logger.warning(f"Malformed URL: {url}")
```

**Files:** `offline_tools/indexer.py:749-761`

### 0.4 Live Site Indexing (Already Exists - Port from Private Repo)

**Status:** ALREADY BUILT in private repo. Just needs to be ported in Phase 1.

**Problem:** Public repo indexer only works on local backups. Live site scraping exists in private repo.

**What Exists in `disaster-clippy/scraper/` (9 files, ~5,000 lines):**

| File | Class | Purpose |
|------|-------|---------|
| `base.py` | `BaseScraper`, `ScrapedPage`, `RateLimitMixin` | Abstract base with rate limiting |
| `mediawiki.py` | `MediaWikiScraper` | Full MediaWiki API - `get_all_pages()`, `search_pages()`, `get_category_pages()` |
| `fandom.py` | `FandomScraper` | Fandom wiki scraper |
| `static_site.py` | `StaticSiteScraper`, `BuildItSolarScraper` | Generic HTML site scraper |
| `substack.py` | `SubstackScraper` | Newsletter scraper |
| `pdf.py` | `PDFScraper` | PDF document scraper |
| `pdf_collections.py` | `PDFCollectionManager` | PDF collection management |
| `appropedia.py` | `ApropediaScraper` | Appropedia-specific scraper |
| `__init__.py` | `get_scraper()`, `SCRAPER_REGISTRY` | Factory function for all scrapers |

**What Exists in `disaster-clippy/ingest.py` (969 lines):**

CLI that connects scrapers to vector store:
```bash
# Scrape ANY MediaWiki site
python ingest.py scrape mediawiki --url https://www.appropedia.org --all --limit 1000

# Search specific topics
python ingest.py scrape mediawiki --url https://solarcooking.fandom.com --search "parabolic"

# Get pages from a category
python ingest.py scrape appropedia --category "Water" --limit 100

# Sync all configured sources from config file
python ingest.py sync
```

**Data Flow (already working):**
```
Live Site -> scraper.scrape_page() -> ScrapedPage -> .to_dict() -> ingest_documents() -> VectorStore
```

**URL Discovery Methods (already implemented in MediaWikiScraper):**
- `get_all_pages(limit)` - Uses MediaWiki API `action=query&list=allpages`
- `search_pages(query, limit)` - Uses MediaWiki API search
- `get_category_pages(category, limit)` - Uses MediaWiki API category members
- `list_categories()` - Lists all categories on wiki

**Key Functions to Preserve:**
- `disaster-clippy/scraper/mediawiki.py:108-155` - `get_all_pages()` with pagination
- `disaster-clippy/scraper/mediawiki.py:157-200` - `search_pages()`
- `disaster-clippy/scraper/mediawiki.py:292-373` - `scrape_page_by_title()` using API
- `disaster-clippy/scraper/base.py:112-136` - `scrape_all()` with progress callback
- `disaster-clippy/ingest.py:52-87` - `ingest_documents()` with deduplication
- `disaster-clippy/ingest.py:162-225` - `_handle_web_scrape()` orchestration

**Action:** Copy `scraper/` and `ingest.py` in Phase 1.1 (already in plan)

### 0.5 Auto-Generate Tags During Indexing

**Problem:** Tags only suggested during validation, not auto-applied. Limited to 5 tags and small keyword vocabulary.

**What Exists (source_manager.py:148-157, 1431-1487):**
- `TOPIC_KEYWORDS` dict with 8 topics (water, solar, energy, food, shelter, health, emergency, diy)
- `suggest_tags()` function scans titles and source_id for keyword matches
- Limited to 5 suggestions

**Improvements Needed:**

1. **Expand TOPIC_KEYWORDS** - Add more disaster-relevant topics:
```python
TOPIC_KEYWORDS = {
    # Existing
    "water": ["water", "filtration", "purification", "well", "pump", "irrigation", "rainwater", "desalination"],
    "solar": ["solar", "photovoltaic", "pv", "sun", "renewable", "panel"],
    "energy": ["energy", "power", "electricity", "generator", "battery", "wind", "hydro", "off-grid"],
    "food": ["food", "agriculture", "farming", "garden", "crop", "cooking", "preservation", "canning", "ferment"],
    "shelter": ["shelter", "housing", "building", "construction", "roof", "insulation", "earthbag", "cob"],
    "health": ["health", "medical", "medicine", "first aid", "sanitation", "hygiene", "wound"],
    "emergency": ["emergency", "disaster", "survival", "preparedness", "crisis", "evacuation", "rescue"],
    "diy": ["diy", "homemade", "build", "make", "construct", "repair", "fix"],
    # New topics
    "communication": ["radio", "ham", "antenna", "signal", "communication", "mesh", "network"],
    "transportation": ["vehicle", "bicycle", "boat", "fuel", "engine", "transport"],
    "tools": ["tool", "workshop", "forge", "welding", "machining", "woodworking"],
    "sanitation": ["toilet", "composting", "sewage", "waste", "latrine", "greywater"],
    "heating": ["heating", "stove", "furnace", "insulation", "thermal", "firewood", "rocket stove"],
    "cooling": ["cooling", "ventilation", "evaporative", "shade", "passive cooling"],
    "lighting": ["lighting", "lamp", "led", "candle", "lantern"],
    "storage": ["storage", "preservation", "root cellar", "stockpile", "cache"],
}
```

2. **Increase tag limit from 5 to 10-12:**
```python
# In suggest_tags():
return list(suggested)[:12]  # Was [:5]
```

3. **Auto-apply tags during indexing:**
```python
# In indexer.py after creating source:
from sourcepacks.source_manager import SourceManager
manager = SourceManager()
suggested = manager.suggest_tags(source_id)
if suggested and not source_meta.get("tags"):
    source_meta["tags"] = suggested
    # Save updated source metadata
```

4. **Scan full content, not just titles:**
```python
# Full content is already being read for embeddings - keyword scan is essentially free
content_text = doc.get("content", "").lower()
all_text += " " + content_text
```

5. **Future: Learn new tags from content (not in Phase 0):**
   - Track frequently-occurring terms that don't match existing tags
   - Surface candidates in admin UI for human approval
   - Options: TF-IDF extraction, LLM suggestion, or embedding clustering
   - Processing cost is negligible since content is already being read for embeddings

**Files:**
- `sourcepacks/source_manager.py:148-157` - Expand TOPIC_KEYWORDS
- `sourcepacks/source_manager.py:1487` - Change limit from 5 to 12
- `offline_tools/indexer.py` - Add auto-apply after indexing

### 0.6 ZIM File Improvements

**Problem:** ZIM files contain metadata and search indexes that we're not using. Also, "offline browsing" currently requires internet (converts `zim://` to live web URLs).

#### 0.6.1 Extract ZIM Metadata During Indexing

**What ZIM files have that we ignore:**
- **Categories** - Currently set to `[]` (line 517), but ZIM stores article categories
- **Global metadata** - Creator, description, language, tags - could auto-populate `_source.json`
- **Full-text search index** - Xapian index built into ZIM (not currently used)

**Fix in ZIMIndexer:**
```python
# Extract categories from article (if available)
categories = []
# Many ZIM files store categories in article metadata or special pages
# Check article namespace and Category: links in content

# Extract global ZIM metadata for _source.json
zim_metadata = {
    "name": zim.metadata.get("Title", source_id),
    "description": zim.metadata.get("Description", ""),
    "language": zim.metadata.get("Language", "en"),
    "creator": zim.metadata.get("Creator", ""),
    "base_url": zim.metadata.get("Source", ""),  # Original website URL
}
```

**Files:** `offline_tools/indexer.py:511-523` - Extract categories, `indexer.py:367-400` - Populate source metadata from ZIM

#### 0.6.2 True Offline Browsing

**Current Problem:**
```
User clicks ZIM result -> zim://solarcooking/A/Solar_panel
  -> _convert_zim_url() -> https://solarcooking.fandom.com/wiki/Solar_panel
  -> REQUIRES INTERNET (defeats purpose of offline!)
```

**Solution: Add ZIM content serving endpoint:**
```python
# New endpoint in app.py
@app.get("/zim/{source_id}/{path:path}")
def serve_zim_article(source_id: str, path: str):
    """Serve ZIM article content directly - true offline browsing"""
    zim_path = get_zim_path_for_source(source_id)
    zim = ZIMFile(zim_path)
    article = zim.get_article_by_url(path)
    return HTMLResponse(article.data, media_type="text/html")
```

**URL generation based on connection mode:**
```python
def get_article_url(zim_url: str, source_id: str) -> str:
    mode = config.get_offline_mode()  # "online_only", "offline_only", "hybrid"

    if mode == "online_only":
        # Convert to live web URL
        return _convert_zim_url(zim_url, source_id)

    elif mode == "offline_only":
        # Serve from local ZIM
        # zim://solarcooking/A/Page -> /zim/solarcooking/A/Page
        return zim_url.replace("zim://", "/zim/")

    else:  # hybrid
        # Auto-detect: ping check or try online first
        if _is_online():
            return _convert_zim_url(zim_url, source_id)
        else:
            return zim_url.replace("zim://", "/zim/")

def _is_online() -> bool:
    """Quick connectivity check (cached for performance)"""
    try:
        requests.head("https://1.1.1.1", timeout=1)
        return True
    except:
        return False
```

**User experience by mode:**

| Mode | URL Result | Requires Internet |
|------|------------|-------------------|
| `online_only` | `https://solarcooking.fandom.com/wiki/Solar_panel` | Yes |
| `offline_only` | `/zim/solarcooking/A/Solar_panel` (served locally) | No |
| `hybrid` | Auto-detect via ping, fallback to local | Graceful degradation |

**Files:**
- `app.py` - Add `/zim/{source_id}/{path:path}` endpoint
- `app.py:1208-1242` - Update `_convert_zim_url()` to respect connection mode
- `useradmin/local_config.py` - Connection mode already exists, reuse it

#### 0.6.3 Future: Hybrid Search with ZIM Index

ZIM files include a Xapian full-text search index. Could potentially:
- Use ZIM search for keyword matching (fast, no embedding cost)
- Use vector search for semantic matching
- Combine results for hybrid search

Not in Phase 0 - requires research into zimply-core Xapian access.

### 0.7 Auto-Detect Licenses During Indexing

**Problem:** License defaults to "Unknown" and requires manual entry. But many sources have detectable licenses.

**What Already Exists (source_manager.py:104-145, 1335-1430):**
- `LICENSE_PATTERNS` dict with regex for CC-BY-SA, CC-BY, CC0, Public Domain, MIT, GPL, etc.
- `detect_license()` scans metadata files, document content, LICENSE files, backup manifest
- `_match_license_patterns()` does regex matching

**Improvements Needed:**

1. **Known domain licenses (hardcode):**
```python
KNOWN_LICENSES = {
    # Wikis with known licenses
    "fandom.com": "CC-BY-SA",
    "wikipedia.org": "CC-BY-SA",
    "appropedia.org": "CC-BY-SA",
    "wikimedia.org": "CC-BY-SA",
    "wikibooks.org": "CC-BY-SA",
    "solarcooking.fandom.com": "CC-BY-SA",
    # Other known sources
    "builditsolar.com": "Open Access",  # Verify
    "kiwix.org": "Varies by ZIM",
}

def detect_license_from_url(base_url: str) -> Optional[str]:
    """Check if domain has known license"""
    for domain, license in KNOWN_LICENSES.items():
        if domain in base_url:
            return license
    return None
```

2. **Extract from ZIM metadata:**
```python
# In ZIMIndexer - ZIM files often have license metadata
license = zim.metadata.get("License", None)
if license:
    source_meta["license"] = license
    source_meta["license_verified"] = True  # From official ZIM metadata
```

3. **Scrape MediaWiki copyright page during backup:**
```python
# In html_backup.py or mediawiki scraper
def get_wiki_license(api_url: str) -> Optional[str]:
    """Fetch license from MediaWiki API"""
    response = requests.get(f"{api_url}?action=query&meta=siteinfo&siprop=rightsinfo&format=json")
    rights = response.json().get("query", {}).get("rightsinfo", {})
    return rights.get("text")  # e.g., "Creative Commons Attribution-ShareAlike"
```

4. **Auto-apply during indexing:**
```python
# In indexer.py after creating source
if not source_meta.get("license") or source_meta["license"] == "Unknown":
    # Try domain lookup first
    detected = detect_license_from_url(base_url)
    if not detected:
        # Fall back to content scanning
        detected = manager.detect_license(source_id)
    if detected:
        source_meta["license"] = detected
        source_meta["license_verified"] = False  # Auto-detected, needs human verification
        source_meta["license_detection_method"] = "auto"
```

5. **Track detection confidence:**
```python
# Add to _source.json
{
    "license": "CC-BY-SA",
    "license_verified": false,
    "license_detection_method": "domain_match",  # or "content_scan", "zim_metadata", "manual"
    "license_source_url": "https://wiki.example.com/Project:Copyrights"
}
```

**What can be fully automated:**

| Source Type | Detection Method | Confidence |
|-------------|-----------------|------------|
| Fandom wikis | Domain match | High (Fandom ToS requires CC-BY-SA) |
| Wikipedia/Wikimedia | Domain match | High |
| Appropedia | Domain match | High |
| ZIM files | ZIM metadata | High |
| MediaWiki sites | API rightsinfo | Medium-High |
| HTML backups | Content regex scan | Medium |
| PDFs | PDF metadata + content | Low-Medium |
| Substack | Cannot auto-detect | Manual only |

**Files:**
- `sourcepacks/source_manager.py:104-145` - Add KNOWN_LICENSES dict
- `offline_tools/indexer.py` - Auto-apply license during indexing
- `scraper/mediawiki.py` - Add `get_wiki_license()` to fetch from API
- `offline_tools/html_backup.py` - Store detected license in manifest

---

## What Each Repo Has

### disaster-clippy-public (THE BASE)
- FastAPI + Jinja2 admin UI (cleaner)
- More complete `source_manager.py` (1,567 lines vs 968)
  - Extra: `normalize_filenames()`, `scan_backup()`, `cleanup_redundant_files()`
  - Extra: Windows ZIM handle cleanup
  - Extra: `ValidationResult.redundant_files`, `has_cleanup_needed`
- `useradmin/` with job manager, ollama manager
- Local admin workflows
- **Backup tools (identical in both repos):**
  - `offline_tools/html_backup.py` (756 lines) - `HTMLBackupScraper` downloads websites to local HTML
  - `offline_tools/substack_backup.py` (410 lines) - Substack newsletter backup
  - `offline_tools/schemas.py` (417 lines) - Identical

### File Comparison: offline_tools/

| File | Private | Public | Status |
|------|---------|--------|--------|
| `html_backup.py` | 756 | 756 | IDENTICAL |
| `substack_backup.py` | 410 | 410 | IDENTICAL |
| `schemas.py` | 417 | 417 | IDENTICAL |
| `indexer.py` | 1331 | 1374 | PUBLIC +43 lines (more complete) |
| `migrate.py` | 379 | MISSING | **SKIP - backward compat tool** |

### File Comparison: sourcepacks/

| File | Private | Public | Status |
|------|---------|--------|--------|
| `__init__.py` | 11 | 11 | IDENTICAL |
| `api.py` | 253 | 253 | IDENTICAL |
| `pack_tools.py` | 901 | 901 | IDENTICAL |
| `registry.py` | 278 | 314 | PUBLIC +36 lines (more complete) |
| `source_manager.py` | 968 | 1567 | PUBLIC +599 lines (much more complete) |

### disaster-clippy (TO PORT FROM)
- `scraper/` module (9 files, ~5,000 lines) - **NEED THIS**
- `offline_tools/migrate.py` (379 lines) - **SKIP** (backward compat tool)
- `admin/app.py` Streamlit (5,047 lines) - **Pull functionality as needed, use as logic inspiration**
- `vectordb/pinecone_store.py` - **NEED THIS** (global admin only)
- `ingest.py`, `sync.py` CLI tools - **NEED THESE**
- Pinecone sync logic - **NEED THIS** (global admin only)
- Global admin features - **NEED THESE**

---

## Phase 1: Port Missing Code from Private Repo

### 1.1 Copy Scraper Module

**From:** `disaster-clippy/scraper/`
**To:** `disaster-clippy-public/scraper/`

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | ~50 | Factory: `get_scraper()`, `SCRAPER_REGISTRY` |
| `base.py` | ~100 | `BaseScraper`, `ScrapedPage`, `RateLimitMixin` |
| `mediawiki.py` | 491 | Generic MediaWiki API |
| `appropedia.py` | 251 | Appropedia-specific |
| `fandom.py` | ~200 | Fandom wiki |
| `static_site.py` | 357 | Static HTML sites |
| `pdf.py` | 779 | PDF extraction |
| `pdf_collections.py` | 731 | PDF collection management |
| `substack.py` | 329 | Substack newsletters |

**Action:** Direct copy, no modifications needed.

### 1.2 migrate.py - SKIP

**Location:** `disaster-clippy/offline_tools/migrate.py` (379 lines)

**What it does:** One-time migration script to convert sources from schema v1 to v2.

**Decision: DO NOT COPY.** This is a backward compatibility tool. Per our "no backward compatibility" philosophy, we will re-index sources from scratch using the new schema rather than migrating old data formats. Any sources with old schemas should be re-indexed fresh.

### 1.3 Copy Pinecone Store (Global Admin Only)

**From:** `disaster-clippy/vectordb/pinecone_store.py` (429 lines)
**To:** `disaster-clippy-public/admin/services/pinecone.py`

**NOTE:** This is GLOBAL ADMIN ONLY functionality. Local admins use ChromaDB exclusively.

**What it does:**
- Connects to Pinecone cloud vector database API
- Uploads vectors (embeddings) to the cloud index
- Searches vectors in the cloud
- Syncs local sources to the global searchable database

Key methods to verify exist:
- `PineconeStore.__init__()`
- `search()`
- `add_documents()`
- `delete()`
- `get_stats()`

**Action:** Copy to `admin/services/pinecone.py`, only loaded when `ADMIN_MODE=global`.

### 1.4 Copy CLI Tools

**From private, copy to public:**

| File | Lines | Purpose |
|------|-------|---------|
| `ingest.py` | 969 | Content ingestion CLI |
| `sync.py` | 342 | Pinecone sync CLI (global admin only) |

These depend on scraper module, so copy after 1.1.

### 1.5 Copy Global Admin Logic

**From:** `disaster-clippy/admin/app.py` (Streamlit, 5,047 lines)

Don't copy the file - extract the business logic:

| Streamlit Section | Extract To | Purpose |
|-------------------|-----------|---------|
| Lines 162-200 | `admin/services/pinecone.py` | `check_pinecone_status()` |
| Sync operations | `admin/services/sync.py` | Pinecone sync logic |
| Submissions review | `admin/services/submissions.py` | Review workflow |
| Source curation | Already in `source_manager.py` | Validation, licensing |

---

## Phase 2: Unify Admin UI

### 2.1 Current State

```
disaster-clippy-public/
  useradmin/
    app.py              # 2,097 lines - FastAPI routes
    templates/          # Jinja2 templates
    local_config.py
    job_manager.py
    ollama_manager.py
    cloud_upload.py
```

### 2.2 Target State

```
disaster-clippy-public/
  admin/                          # Renamed from useradmin/
    __init__.py
    app.py                        # FastAPI router factory
    config.py                     # Settings management

    routes/
      __init__.py
      dashboard.py                # Stats, health
      sources.py                  # Source CRUD
      backups.py                  # Backup management
      indexing.py                 # Index operations
      settings.py                 # User settings
      cloud.py                    # R2 upload/download
      # Global-only routes (loaded conditionally):
      pinecone.py                 # Pinecone settings and sync
      submissions.py              # Review queue

    services/
      __init__.py
      pinecone.py                 # Pinecone operations (global admin only)
      sync.py                     # Sync logic (global admin only)
      submissions.py              # Submission workflow

    templates/
      base.html
      dashboard.html
      sources/
        list.html
        detail.html
        create.html
      backups/
        ...

    static/
      admin.css
      admin.js

    job_manager.py                # Keep
    ollama_manager.py             # Keep
    local_config.py               # Keep
```

### 2.3 Mode Switching

**In `admin/app.py`:**
```python
import os
from fastapi import APIRouter

ADMIN_MODE = os.getenv("ADMIN_MODE", "local")  # "local" or "global"

def get_admin_router() -> APIRouter:
    from .routes import dashboard, sources, backups, indexing, settings, cloud

    router = APIRouter(prefix="/admin", tags=["Admin"])

    # Always included
    router.include_router(dashboard.router)
    router.include_router(sources.router)
    router.include_router(backups.router)
    router.include_router(indexing.router)
    router.include_router(settings.router)
    router.include_router(cloud.router)

    # Global admin only
    if ADMIN_MODE == "global":
        from .routes import pinecone, submissions
        router.include_router(pinecone.router)
        router.include_router(submissions.router)

    return router
```

### 2.4 What to Extract from useradmin/app.py

Current `useradmin/app.py` (2,097 lines) should be split:

| Lines (approx) | Current Function | Move To |
|----------------|------------------|---------|
| 1-200 | Imports, setup, ZIM analysis | `admin/routes/__init__.py`, `admin/services/zim.py` |
| 200-500 | Settings endpoints | `admin/routes/settings.py` |
| 500-900 | Source listing/sync | `admin/routes/sources.py` |
| 900-1300 | Backup operations | `admin/routes/backups.py` |
| 1300-1600 | Indexing operations | `admin/routes/indexing.py` |
| 1600-1900 | Validation, licensing | `admin/routes/sources.py` or `services/` |
| 1900-2097 | Cloud operations | `admin/routes/cloud.py` |

---

## Phase 3: Consolidate Folder Structure

### 3.1 Current Structure (Public Repo)

```
disaster-clippy-public/
  app.py
  local_cli.py
  vectordb/
  offline_tools/
  sourcepacks/
  storage/
  useradmin/
  static/
  templates/
```

### 3.2 Target Structure

```
disaster-clippy-public/
  app.py                          # Main FastAPI app
  local_cli.py                    # User CLI (keep)

  # --- NEW: Scrapers (from private) ---
  scraper/
    __init__.py
    base.py
    mediawiki.py
    appropedia.py
    fandom.py
    static_site.py
    pdf.py
    pdf_collections.py
    substack.py

  # --- CONSOLIDATED: Core tools ---
  offline_tools/
    __init__.py
    schemas.py                    # Data structures
    embeddings.py                 # MOVE from vectordb/
    indexer.py                    # Keep
    source_manager.py             # MOVE from sourcepacks/
    packager.py                   # MOVE from sourcepacks/pack_tools.py

    backup/                       # Reorganize
      __init__.py
      html.py                     # MOVE from html_backup.py
      substack.py                 # MOVE from substack_backup.py

    vectordb/                     # MOVE from vectordb/
      __init__.py
      store.py
      metadata.py
      factory.py
      sync.py

    cloud/                        # MOVE from storage/
      __init__.py
      r2.py

  # --- UNIFIED: Admin panel ---
  admin/                          # RENAME from useradmin/
    __init__.py
    app.py
    config.py
    routes/
    services/
      pinecone.py                 # Global admin only - Pinecone operations
      sync.py                     # Global admin only - Pinecone sync
      submissions.py              # Submission workflow
    templates/
    static/
    job_manager.py
    ollama_manager.py
    local_config.py

  # --- NEW: CLI tools (from private) ---
  cli/
    __init__.py
    ingest.py                     # MOVE from private
    sync.py                       # MOVE from private (global admin only)

  # --- KEEP: Main app assets ---
  static/
    chat.js
    embed-widget.html
  templates/
    index.html

  # --- DELETE after migration ---
  # vectordb/          -> offline_tools/vectordb/
  # sourcepacks/       -> offline_tools/ + admin/routes/
  # storage/           -> offline_tools/cloud/
  # useradmin/         -> admin/
```

### 3.3 Folders to Delete (After Migration)

| Folder | Replaced By |
|--------|-------------|
| `vectordb/` | `offline_tools/vectordb/` |
| `sourcepacks/` | `offline_tools/source_manager.py`, `offline_tools/packager.py`, `admin/routes/sources.py` |
| `storage/` | `offline_tools/cloud/` |
| `useradmin/` | `admin/` |

---

## Phase 4: Update Imports

### 4.1 Old -> New Import Mapping

| Old Import | New Import |
|------------|-----------|
| `from vectordb import get_vector_store` | `from offline_tools.vectordb import get_vector_store` |
| `from vectordb.store import VectorStore` | `from offline_tools.vectordb.store import VectorStore` |
| `from vectordb.embeddings import EmbeddingService` | `from offline_tools.embeddings import EmbeddingService` |
| `from vectordb.metadata import MetadataIndex` | `from offline_tools.vectordb.metadata import MetadataIndex` |
| `from storage.r2 import get_r2_storage` | `from offline_tools.cloud.r2 import get_r2_storage` |
| `from sourcepacks.pack_tools import ...` | `from offline_tools.packager import ...` |
| `from sourcepacks.source_manager import SourceManager` | `from offline_tools.source_manager import SourceManager` |
| `from sourcepacks.api import ...` | `from admin.routes.sources import ...` |
| `from useradmin.app import ...` | `from admin.app import ...` |

### 4.2 Files Needing Import Updates

**High impact (many imports):**
- `app.py` - Main app
- `local_cli.py` - CLI tool
- `offline_tools/indexer.py` - Imports vectordb, sourcepacks
- `admin/` (all files) - Imports everything

**Run after restructure:**
```bash
# Find all old imports
grep -r "from vectordb" --include="*.py"
grep -r "from sourcepacks" --include="*.py"
grep -r "from storage" --include="*.py"
grep -r "from useradmin" --include="*.py"
```

---

## Phase 5: File Naming Migration

### 5.1 Schema File Names

**Update in `offline_tools/schemas.py` lines 29-52:**

| Current Function | Current Return | New Return |
|------------------|----------------|------------|
| `get_source_file()` | `{source_id}_source.json` | `_manifest.json` |
| `get_documents_file()` | `{source_id}_documents.json` | `_metadata.json` |
| `get_embeddings_file()` | `{source_id}_embeddings.json` | `_vectors.json` |
| `get_backup_manifest_file()` | `{source_id}_backup_manifest.json` | `backup_manifest.json` |
| `get_distribution_manifest_file()` | `{source_id}_manifest.json` | (merge into `_manifest.json`) |
| (new) | N/A | `_index.json` |

**New file structure per source:**

| File | Purpose | Contents |
|------|---------|----------|
| `_manifest.json` | Source identity | name, license, base_url, tags, stats |
| `_metadata.json` | Document lookup | title, url, hash per document |
| `_index.json` | Full content | document content for scanning/display |
| `_vectors.json` | Embeddings | vectors for ChromaDB/Pinecone search |
| `backup_manifest.json` | URL mappings | URL to local file mapping (HTML only) |

### 5.2 Re-index Old Sources

Per "no backward compatibility" philosophy, sources with old schema formats should be re-indexed from scratch rather than migrated. The indexer will output the new file format directly.

---

## Phase 6: Cleanup

### 6.1 Archive Private Repo

1. Push final state of `disaster-clippy` to a branch called `archive/pre-merge`
2. Update README: "This repo is archived. See disaster-clippy-public for active development."
3. Make repo read-only or private

### 6.2 Rename Public Repo (Optional)

Consider renaming `disaster-clippy-public` to just `disaster-clippy` since it's now the main repo.

### 6.3 Update Documentation

- Update all README files
- Update DEVELOPER.md with new structure
- Delete FUTURE_CONTEXT.md (consolidated into this plan)
- Update CONTEXT.md with final architecture

---

## Implementation Strategy

**Approach:** Reorganize folder structure FIRST to match target state (Section 2.2), fix imports, verify everything works, THEN add new features. This way new code goes directly to its final location.

**Why this approach:**
- New code (scraper/, pinecone, etc.) goes directly to final location
- No "copy here now, move later" - touch files once
- Imports are fixed once, not twice
- Cleaner git history
- Data can be rebuilt if needed (Pinecone backup exists, sources can be re-indexed)

---

## Implementation Order

### Phase 1: Folder Reorganization (Structure First) - COMPLETED Dec 3, 2025

**Goal:** Get folder structure to match Section 2.2 target state, then verify app still works.

**Status: COMPLETE**

1. [x] Rename `useradmin/` to `admin/`
2. [x] Create `admin/routes/` folder structure
3. [x] Create `admin/services/` folder structure
4. [x] Move `vectordb/` contents to `offline_tools/vectordb/`
5. [x] Move `storage/` contents to `offline_tools/cloud/`
6. [x] Move `sourcepacks/` contents to `offline_tools/`
7. [x] Reorganize `offline_tools/` backup files into `offline_tools/backup/`
8. [x] Update ALL imports throughout codebase
9. [x] Test: `python app.py` starts, admin panel works, search works

**Files moved:**
- `vectordb/embeddings.py` -> `offline_tools/embeddings.py`
- `vectordb/store.py` -> `offline_tools/vectordb/store.py`
- `vectordb/metadata.py` -> `offline_tools/vectordb/metadata.py`
- `vectordb/factory.py` -> `offline_tools/vectordb/factory.py`
- `vectordb/sync.py` -> `offline_tools/vectordb/sync.py`
- `vectordb/pinecone_store.py` -> `offline_tools/vectordb/pinecone_store.py`
- `storage/r2.py` -> `offline_tools/cloud/r2.py`
- `sourcepacks/source_manager.py` -> `offline_tools/source_manager.py`
- `sourcepacks/pack_tools.py` -> `offline_tools/packager.py`
- `sourcepacks/registry.py` -> `offline_tools/registry.py`
- `sourcepacks/api.py` -> `admin/routes/sources.py`
- `offline_tools/html_backup.py` -> `offline_tools/backup/html.py`
- `offline_tools/substack_backup.py` -> `offline_tools/backup/substack.py`

**Old folders archived to `_archive/`:**
- `vectordb/`
- `sourcepacks/`
- `storage/`
- `useradmin/`

**Other cleanup:**
- Deleted empty `data/` folder
- Moved `config/local_settings.json` to `_archive/`
- Deleted empty `config/` folder

### Phase 2: Add New Features (To Clean Structure) - COMPLETED

**Goal:** Port missing capabilities from private repo, placing files directly in final locations.

10. [x] Copy `scraper/` module from private to public (new folder)
11. [x] Copy `ingest.py` to `cli/ingest.py`
12. [x] Update `pinecone_store.py` with R2 sources feature (already in `offline_tools/vectordb/`)
13. [x] Copy `sync.py` to `cli/sync.py`
14. [x] Update imports in copied files to match new structure
15. [x] Test: scrapers work, Pinecone connects (when API key set)

**Completed:** December 3, 2025

Files added:
- `scraper/` module (8 files: __init__.py, base.py, mediawiki.py, appropedia.py, fandom.py, static_site.py, pdf.py, pdf_collections.py, substack.py)
- `cli/__init__.py`
- `cli/ingest.py`
- `cli/sync.py`

Note: `pinecone_store.py` was already at `offline_tools/vectordb/pinecone_store.py` - updated with R2 sources feature.

### Phase 3: Add ADMIN_MODE Gating

**Goal:** Gate global-only features so local admins don't see them.

16. Add `ADMIN_MODE` environment variable check to `admin/app.py`
17. Create `/admin/pinecone` page - only visible when `ADMIN_MODE=global`
18. Create `/admin/submissions` page - only visible when `ADMIN_MODE=global`
19. Gate R2 backups/ write access behind `ADMIN_MODE=global`
20. Test: Local mode hides global features, global mode shows all

### Phase 4: Schema Updates

**Goal:** Update file naming conventions for cleaner source packages.

21. Update `schemas.py` with new file naming (`_manifest.json`, `_index.json`, `_vectors.json`)
22. Update `indexer.py` to output new format
23. Re-index sources as needed (no migration, just rebuild)

### Phase 5: Cleanup

24. Delete old empty folders (`vectordb/`, `sourcepacks/`, `storage/`, `useradmin/`)
25. Archive private repo
26. Update documentation (README, DEVELOPER.md, CONTEXT.md)

---

## Private Repo Tools Reference

These tools exist in `disaster-clippy` (private) and can be ported when needed.

### Scraper Module (scraper/)

| File | Lines | Purpose | Port When |
|------|-------|---------|-----------|
| `__init__.py` | ~50 | Factory: `get_scraper()`, `SCRAPER_REGISTRY` | Phase 1 |
| `base.py` | ~100 | `BaseScraper`, `ScrapedPage`, `RateLimitMixin` | Phase 1 |
| `mediawiki.py` | 491 | Generic MediaWiki API scraper | Phase 1 |
| `appropedia.py` | 251 | Appropedia-specific scraper | Phase 1 |
| `fandom.py` | ~200 | Fandom wiki scraper | Phase 1 |
| `static_site.py` | 357 | Static HTML sites (BuildItSolar) | Phase 1 |
| `pdf.py` | 779 | PDF document extraction | Phase 1 |
| `pdf_collections.py` | 731 | PDF collection management | Phase 1 |
| `substack.py` | 329 | Substack newsletter scraper | Phase 1 |

### CLI Tools

| File | Lines | Purpose | Port When |
|------|-------|---------|-----------|
| `ingest.py` | 969 | Content ingestion CLI, connects scrapers to vector store | Phase 1 |
| `sync.py` | 342 | Pinecone sync CLI | Phase 2 |

### Pinecone Integration

| File | Lines | Purpose | Port When |
|------|-------|---------|-----------|
| `vectordb/pinecone_store.py` | 429 | Pinecone API wrapper | Phase 2 |

### Streamlit Admin (admin/app.py - 5,047 lines)

Features to extract for FastAPI admin:

| Lines (approx) | Feature | Port When |
|----------------|---------|-----------|
| 162-200 | Pinecone status check | Phase 3 |
| 200-400 | Pinecone sync operations | Phase 3 |
| 400-600 | Submissions review queue | Phase 3 |
| 600-900 | Source curation tools | Already in public |
| 900-1200 | Backup management | Already in public |

### Other Private Files (reference only)

| File | Purpose | Notes |
|------|---------|-------|
| `ingest_config.json` | Scraper configuration | May need for Phase 1 |
| `test_pinecone.py` | Pinecone connection test | Useful for Phase 2 testing |
| `posts.csv` | Substack posts data | Data file, not code |

---

## Files to Move (Phase 1)

Complete list of file moves for Phase 1 folder reorganization:

| Source | Destination |
|--------|-------------|
| `vectordb/embeddings.py` | `offline_tools/embeddings.py` |
| `vectordb/store.py` | `offline_tools/vectordb/store.py` |
| `vectordb/metadata.py` | `offline_tools/vectordb/metadata.py` |
| `vectordb/factory.py` | `offline_tools/vectordb/factory.py` |
| `vectordb/sync.py` | `offline_tools/vectordb/sync.py` |
| `storage/r2.py` | `offline_tools/cloud/r2.py` |
| `sourcepacks/source_manager.py` | `offline_tools/source_manager.py` |
| `sourcepacks/pack_tools.py` | `offline_tools/packager.py` |
| `sourcepacks/api.py` | `admin/routes/sources.py` |
| `sourcepacks/registry.py` | `offline_tools/registry.py` |
| `offline_tools/html_backup.py` | `offline_tools/backup/html.py` |
| `offline_tools/substack_backup.py` | `offline_tools/backup/substack.py` |

### Files to Delete (End of Phase 1)

| File/Folder | Reason |
|-------------|--------|
| `vectordb/` (root) | Moved to `offline_tools/vectordb/` |
| `sourcepacks/` | Moved to `offline_tools/` and `admin/routes/` |
| `storage/` | Moved to `offline_tools/cloud/` |
| `useradmin/` | Renamed to `admin/` |
| `FUTURE_CONTEXT.md` | Consolidated into this plan |

---

## Decisions Made

| Topic | Decision |
|-------|----------|
| Backward compatibility | None - no fallbacks, fix forward from backups |
| `_index.json` vs `_vectors.json` | Split: `_index.json` = content, `_vectors.json` = embeddings |
| Pinecone location | `admin/services/pinecone.py` (global admin only) |
| Hierarchical sources | Deferred - not in scope for this refactoring |
| `base.py` cloud abstraction | Not needed - use `r2.py` directly |
| migrate.py | Skip - re-index sources instead of migrating old schemas |

---

## Decisions Deferred

| Topic | Decision | Revisit When |
|-------|----------|--------------|
| Repo rename (`-public` suffix) | TBD | After merge complete |
| Hierarchical source support | TBD | After basic refactoring complete |
| Additional cloud providers | TBD | When user requests it |

---

## Testing Checklist

After each phase:
- [ ] `python app.py` starts without errors
- [ ] Admin panel loads at `/admin/`
- [ ] Source listing works
- [ ] Backup creation works (HTML, ZIM, PDF)
- [ ] Indexing works
- [ ] Validation works
- [ ] Search returns results
- [ ] R2 upload/download works
- [ ] (Global mode) Pinecone settings page loads
- [ ] (Global mode) Pinecone sync operations work

---

## Security Notes

**What stays private (env vars, never in code):**
- `PINECONE_API_KEY` - Only global admin has this
- `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` - Write access credentials
- `ADMIN_API_KEY` - Protects global admin routes

**What's safe to be public:**
- All code (scrapers, admin UI, sync logic)
- Local admins can run everything locally
- They just can't write to YOUR Pinecone/R2 without credentials

---
