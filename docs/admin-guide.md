# Admin Guide

This document covers administering Disaster Clippy: the admin panel, CLI tools, getting content, and troubleshooting.

---

## Table of Contents

1. [Local Admin Panel](#local-admin-panel)
2. [CLI Tools](#cli-tools)
3. [Getting ZIM Files](#getting-zim-files)
4. [Creating HTML Backups](#creating-html-backups)
5. [Global Admin Review Process](#global-admin-review-process)
6. [Smart Sync](#smart-sync)
7. [Orphaned Source Detection](#orphaned-source-detection)
8. [Knowledge Map Visualization](#knowledge-map-visualization)
9. [Troubleshooting](#troubleshooting)
10. [Known Issues and Fixes](#known-issues-and-fixes)

---

## Local Admin Panel

Access at: `http://localhost:8000/useradmin/`

### Features

- **Dashboard** - System status and statistics
- **Sources** - Browse all sources with status boxes, install cloud sources
- **Source Tools** - Multi-step wizard for creating/editing sources
- **Jobs** - Background job queue with progress tracking
- **Visualization** - 3D knowledge map of all indexed documents
- **Settings** - Configure backup paths, connection modes

### Status Boxes (6 items)

Each source displays 6 status indicators:

| Box | Shows | Green | Yellow | Red |
|-----|-------|-------|--------|-----|
| Config | `has_manifest` | Has manifest | - | Missing |
| Backup | `has_backup` | Has content | - | Missing |
| Metadata | `has_metadata` | Has docs | - | Missing |
| 1536 | `has_vectors_1536` | Has vectors | Missing | - |
| 768 | `has_vectors_768` | Has vectors | Missing | - |
| License | License status | Verified | Exists | Unknown |

### Status Badges

| Condition | Badge Text | Color |
|-----------|------------|-------|
| Missing requirements | "Incomplete" | Orange |
| `can_submit` passed | "Ready to Submit" | Blue |
| `can_publish` passed | "Production Ready" | Green |
| Already published | "Published" | Green (dim) |

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

## Getting ZIM Files

ZIM files are compressed offline archives. Download from:

1. **Kiwix Library**: https://library.kiwix.org/
   - Wikipedia (by topic: medicine, technology, etc.)
   - Wikihow
   - StackExchange sites

2. **Direct Downloads**:
   - Search for `[topic] kiwix zim download`
   - Files range from 50MB to several GB

**Recommended starter ZIMs:**
- `wikipedia_en_medicine` (~500MB) - Medical reference
- `wikihow_en_all` (~2GB) - How-to guides
- `wikibooks_en_all` (~1GB) - Technical books

---

## Creating HTML Backups

### Option 1: Admin Panel Scraper (Recommended)

The Source Tools page includes a built-in web scraper with link following.

1. Go to `/useradmin/` -> Source Tools
2. Select or create a source with `backup_type: html`
3. Enter the site's base URL or sitemap URL
4. Configure scrape settings
5. Click "Start Scrape"

### Option 2: Browser Save

- Visit any page and use "Save As" -> "Webpage, Complete"
- Organize saved pages into folders by source

### Option 3: CLI Scrapers

```bash
python cli/ingest.py scrape mediawiki --url https://wiki.example.org --output ./backups/mywiki
```

### Option 4: HTTrack / wget

- Use [HTTrack](https://www.httrack.com/) to mirror websites
- Or use wget: `wget -r -l 2 -p https://example.com`

---

## Global Admin Review Process

When submissions arrive for review, the global admin performs these checks before publishing:

### Review Checklist

| Check | Action | Notes |
|-------|--------|-------|
| Embedding dimensions | Re-index with OpenAI 1536-dim if needed | Submissions may use local embeddings |
| Tags | Verify/add appropriate tags | Required for discovery |
| URLs | Verify base_url works | Test sample links |
| License | Verify license accuracy | Check source site |
| Content quality | Spot check articles | Look for spam, duplicates |
| Language | Confirm content is English | Required for global index |

### Re-indexing Workflow

```
Submission arrives (local vectors only)
    |
    v
Admin opens in Source Tools
    |
    v
Check _vectors.json dimensions
    |
    +-- Missing 1536 --> Click "Create Online Index"
    |                    (creates 1536-dim vectors)
    |
    +-- Missing 768 --> Click "Create Offline Index"
    |                   (creates 768-dim vectors)
    |
    v
Review tags, license, URLs
    |
    v
Run final validation (deep)
    |
    v
Approve --> Publish to Pinecone + R2
```

---

## Smart Sync

When installing, indexing, or publishing a source that already exists, the system shows a modal asking how to handle existing vectors:

| Action | Update Mode | Replace Mode |
|--------|-------------|--------------|
| **Install from Cloud** | Add new vectors, keep existing | Delete old vectors first, then install fresh |
| **Local Indexing** | Skip already-indexed documents | Force re-index all documents |
| **Publish to Pinecone** | Upsert new/changed vectors | Delete all source vectors, then push |

**When to use each mode:**

- **Update** (default): Faster, preserves existing work. Use when adding new content or fixing a few documents.
- **Replace**: Clean slate. Use when the source structure changed significantly, or to fix indexing issues.

**Technical implementation:**

Before action, system calls check endpoint to get existing vector count:
- Local: `/useradmin/api/local-source-check/{source_id}`
- Pinecone: `/useradmin/api/pinecone-check-source/{source_id}`

---

## Orphaned Source Detection

When a source folder is manually deleted but the source still exists in `_master.json` and/or ChromaDB, it appears as an **orphaned source** in the Source Tools dropdown.

**How it works:**
1. Source list reads from BOTH `_master.json` AND folder scan
2. Sources in master but missing folder are flagged `is_orphaned: true`
3. UI shows `[ORPHANED]` prefix with red color
4. Selecting orphaned source shows warning panel with cleanup button

**Purpose:** Allows proper cleanup of ChromaDB vectors and `_master.json` even when the source folder was deleted outside the admin panel.

---

## Knowledge Map Visualization

Interactive 3D visualization of the document network for admin users.

**Purpose:**
- Find gaps in coverage (sparse areas = missing topics)
- Spot duplicate/redundant content (dense clusters)
- Explore document relationships

**Access:** `/useradmin/visualise` (admin-only)

**Technical Implementation:**
- Uses PCA to reduce embedding vectors to 3D coordinates
- Plotly.js for interactive 3D scatter plot
- Edge building from internal links in `_metadata.json`
- Per-source lazy loading for performance

**File Structure:**
```
BACKUP_PATH/visualisation/
|-- _visualisation.json              # Core data (points, coordinates)
|-- _visualisation_urls_{source}.json   # URLs per source (lazy loaded)
|-- _visualisation_edges_{source}.json  # Link edges per source (lazy loaded)
```

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

## Known Issues and Fixes

### Metadata/Index Count Mismatch

**Problem:** `_metadata.json` shows 18k docs, `_vectors.json` shows 5k. Validation fails with "coverage 28%".

**Cause:** Different text extraction methods:
- Metadata generation uses simple regex (extracts more text, more docs pass filter)
- Indexing uses BeautifulSoup (extracts main content only, fewer docs pass filter)

**Fix:** Unify extraction to use "lenient BeautifulSoup" for both.

### Token Limit Errors During Indexing (FIXED Dec 2025)

**Problem:** OpenAI embedding fails with "8581 tokens requested, 8192 max".

**Cause:** Text truncation (20k chars) can still exceed 8192 tokens for dense text.

**Fix:** Added intelligent chunking in `EmbeddingService._embed_with_chunking()`:
1. Try to embed full text (up to 32k chars)
2. If token limit error, split text in half at sentence boundary
3. Recursively embed each half (up to 8 chunks max)
4. Average the embeddings to produce final vector

### Pinecone Sync "Pushed: 0 documents" (FIXED Dec 2025)

**Problem:** After indexing, Pinecone sync shows "Pushed: 0 documents" despite finding thousands to push.

**Cause:** Document ID mismatch between metadata generation and indexer:
- Metadata used sequential IDs like `zim_0`, `zim_1`
- Indexer used hash IDs like `md5(source_id:url)`
- Sync tried to fetch `zim_0` from ChromaDB but ChromaDB had hash IDs

**Fix:** Updated metadata generation to use same hash ID format as indexer.

**Recovery:** Regenerate metadata for affected sources, then retry Pinecone sync.

### Large ZIM Indexing Issues

When indexing large ZIM files (400k+ articles), several issues can occur:

**Issue: Only a small number of documents indexed despite large source**

Symptoms:
- Metadata generation works (e.g., 490k docs in 35 min)
- Indexing completes quickly (3 min) with only ~430 docs

Root causes:
1. **Limit defaulting to 1000**: If the "Index All" button's `dataset.docCount` isn't read correctly
2. **Metadata overwrite bug** (FIXED): `save_all_outputs()` was overwriting `_metadata.json`

**Fix applied:** Added `skip_metadata_save` parameter to `save_all_outputs()` in indexer.py

**To recover:**
1. Re-run "Generate Metadata" to restore full document list
2. Re-run "Index All" - metadata will now be preserved

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

## Related Documentation

- [Source Tools](source-tools.md) - Creating and managing sources
- [Validation System](validation.md) - Source validation gates
- [Jobs System](jobs.md) - Background job processing
- [Deployment](deployment.md) - Setting up different environments

---

*Last Updated: December 2025*
