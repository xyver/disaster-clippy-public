# Disaster Clippy - Developer Documentation

Welcome to the Disaster Clippy developer docs. Choose the guide that matches your role:

---

## Repository Structure (Public vs Private)

This project uses a two-layer architecture. The **public GitHub repository** contains everything needed for local users, while **private components** stay on the maintainer's machine.

### PUBLIC GITHUB (What Users Get)

```
disaster-clippy/
|-- app.py                    # FastAPI chat interface
|-- local_cli.py              # CLI for local admins
|-- useradmin/                # Local admin panel (/useradmin/)
|-- sourcepacks/              # Pack tools (shared utilities)
|   |-- pack_tools.py         # Metadata/indexing functions
|   |-- source_manager.py     # Unified source creation interface
|-- vectordb/                 # ChromaDB local storage
|-- offline_tools/            # Tools to index local backups
|   |-- indexer.py            # HTMLBackupIndexer, ZIMIndexer, PDFIndexer
|   |-- html_backup.py        # HTML website backup/scraping
|   |-- substack_backup.py    # Substack newsletter backup
|-- storage/                  # R2 client (limited access)
|-- config/
|   |-- sources.json          # Empty template (users populate via packs)
|-- templates/                # Web UI templates
|-- static/                   # CSS/JS assets
|-- DEVELOPER-LOCAL.md        # User documentation
```

### PRIVATE (Maintainer Only - Not in Public Repo)

```
disaster-clippy/
|-- admin/                    # Streamlit admin dashboard
|-- scraper/                  # Web scrapers (rate-limited, API access)
|-- ingest.py                 # CLI for scraping content
|-- sync.py                   # Pinecone sync tools
|-- .env                      # API keys (Pinecone write, full R2)
|-- DEVELOPER-PARENT.md       # Maintainer documentation
```

### Access Levels

| Role | Vector DB | R2 Storage | What They Use |
|------|-----------|------------|---------------|
| End User | ChromaDB (local) | Read backups/ only | useradmin panel |
| Local Admin | ChromaDB (local) | Write submissions/, Read backups/ | useradmin + pack tools |
| Global Admin | Pinecone (write) | Full access | Streamlit admin + scrapers |

### Admin Dashboard Comparison

The two admin interfaces serve different purposes and have different features:

| Feature | Global Admin (admin/app.py) | Local User Admin (useradmin/app.py) |
|---------|-------------------------------|---------------------------------------|
| Framework | Streamlit | FastAPI |
| Purpose | Maintainer managing sources | End users managing local setup |
| Vector DB | Pinecone (cloud) | ChromaDB (local) |
| Source management | Full source editing, scraping | Download/install packs |
| Connection modes | N/A (always online) | online_only / hybrid / offline_only |
| Local LLM (Ollama) | N/A | Yes (for offline responses) |
| R2 Cloud Storage | Upload to cloud | Download from cloud |

The offline tools (Ollama, connection modes) are only in useradmin/ because they are end-user features. The global admin runs online with cloud resources and doesn't need offline capabilities.

Shared dependencies used by both:
- `sourcepacks/source_manager.py` - Unified source creation interface
- `sourcepacks/pack_tools.py` - Metadata and index generation
- `offline_tools/indexer.py` - Indexers for HTML, ZIM, PDF
- `offline_tools/html_backup.py` - HTML website backup
- `vectordb/` - Vector store implementations
- `storage/r2.py` - R2 access (upload in admin, download in useradmin)

### Data Flow Architecture

All work is done in the BACKUP_PATH folder to keep sources self-contained:

```
                    GLOBAL ADMIN                          LOCAL USER
                    (Maintainer)                          (End User)
                         |                                     |
    +--------------------v--------------------+                |
    |           BACKUP_PATH                   |                |
    |  - Edit/create sources                  |                |
    |  - Scrape content                       |                |
    |  - Generate metadata & indexes          |                |
    +--------------------+--------------------+                |
                         |                                     |
                         | Upload                              |
                         v                                     |
    +-----------------------------------------------------+    |
    |                 R2 CLOUD STORAGE                    |    |
    |  backups/        - Official packs (read by users)   |<---+ Download
    |  submissions/    - User contributions (review queue)|    |
    |  sources.json    - Source registry                  |    |
    +-----------------------------------------------------+    |
                         |                                     |
                         | Sync                                |
                         v                                     |
    +-----------------------------------------------------+    |
    |               PINECONE (Cloud Vector DB)            |    |
    |  - Global semantic search                           |    |
    |  - Used by Railway app                              |    |
    +-----------------------------------------------------+    |
                         |                                     |
                         v                                     v
    +-----------------------------------------------------+----+
    |              RAILWAY APP (Production)               |    |
    |  - Reads from Pinecone for semantic search          |    |
    |  - Reads from R2 for backup content                 |    |
    +-----------------------------------------------------+    |
                                                               |
                    +------------------------------------------+
                    |
                    v
    +--------------------+--------------------+
    |           LOCAL USER SETUP              |
    |  BACKUP_PATH:                           |
    |  - Downloaded packs from R2             |
    |  - ChromaDB for local semantic search   |
    |  - Can work fully offline               |
    |                                         |
    |  Submissions:                           |
    |  - Upload custom sources to R2          |
    |  - Global admin reviews & approves      |
    +--------------------+--------------------+
```

**Key Points:**
- Both sides use shared `pack_tools.py` functions for source management
- BACKUP_PATH is the working directory for all source data
- Global admin uploads finished sources to Pinecone + R2 (source of truth)
- Railway app reads from Pinecone + R2 (production)
- Local users download from R2 to their BACKUP_PATH
- Local admins can submit sources to `submissions/` for review

### Unified Source Tools

Both dashboards share the same tools for creating and managing sources. These are kept in sync between repos.

#### SourceManager (`sourcepacks/source_manager.py`)

High-level interface for source creation workflow:

```python
from sourcepacks.source_manager import SourceManager

manager = SourceManager()  # Uses BACKUP_PATH automatically

# Create a backup (HTML, ZIM, PDF, Substack)
result = manager.create_backup(
    source_id="my_wiki",
    source_type="html",
    base_url="https://example.wiki.org"
)

# Create an index (auto-detects source type)
result = manager.create_index("my_wiki")

# Validate source before distribution
validation = manager.validate_source("my_wiki", source_config)
# Returns: has_backup, has_index, has_license, detected_license, suggested_tags

# Create distribution pack
pack = manager.create_pack("my_wiki", source_config)
```

#### Indexers (`offline_tools/indexer.py`)

Three indexer classes for different source types:

| Class | Source Type | Input |
|-------|-------------|-------|
| `HTMLBackupIndexer` | HTML websites | Backup folder with pages/ |
| `ZIMIndexer` | ZIM archives | .zim file |
| `PDFIndexer` | PDF documents | PDF file or folder |

Convenience functions:
- `index_html_backup(path, source_id)` - Index HTML backup
- `index_zim_file(path, source_id)` - Index ZIM file
- `index_pdf_folder(path, source_id)` - Index PDF folder

#### Backup Tools (`offline_tools/html_backup.py`)

```python
from offline_tools.html_backup import run_backup

result = run_backup(
    backup_path="D:/backups/my_wiki",
    source_id="my_wiki",
    base_url="https://example.wiki.org",
    scraper_type="mediawiki",  # or: static, fandom
    limit=1000
)
```

#### Source Validation

A source is considered "clean" and ready for distribution when it has:

1. **Backup** - HTML pages, ZIM file, or PDF files exist
2. **Index** - Metadata file exists (`{source_id}_metadata.json`)
3. **License** - Specified and preferably verified
4. **Tags** - Categorized for search (optional but recommended)

The `SourceManager.validate_source()` method checks all of these and provides:
- Auto-detected license from content scanning
- Suggested tags based on content keywords
- List of issues blocking distribution

#### File Structure in BACKUP_PATH

Each source creates a self-contained folder:

```
BACKUP_PATH/
|-- sources.json                    # Source registry
|-- _master.json                    # Master metadata
|-- my_wiki/                        # Source folder
|   |-- pages/                      # HTML backup content
|   |-- my_wiki_backup_manifest.json
|   |-- my_wiki_metadata.json       # Index metadata
|   |-- my_wiki_index.json          # Embeddings (optional)
|   +-- my_wiki_manifest.json       # Distribution manifest
|-- bitcoin.zim                     # ZIM files at root
|-- pdf_collection/                 # PDF folder
|   |-- doc1.pdf
|   |-- doc2.pdf
|   +-- pdf_collection_metadata.json
+-- chroma/                         # ChromaDB data
```

---

## For End Users (Local System)

**[DEVELOPER-LOCAL.md](DEVELOPER-LOCAL.md)** - Setting up your own offline system

If you want to:
- Run Disaster Clippy on your own computer
- Add personal sources and PDFs
- Set up offline backups (ZIM, HTML, PDF)
- Configure connection modes (online/hybrid/offline)
- Use the Local Admin Panel at `/useradmin/`

---

## For Maintainers (Parent System)

**[DEVELOPER-PARENT.md](DEVELOPER-PARENT.md)** - Managing the global infrastructure

If you want to:
- Manage the central Pinecone database
- Deploy to Railway
- Add and curate official sources
- Run the Streamlit admin dashboard
- Sync local changes to production
- Create new scrapers

---

## Quick Reference

| Task | Guide | Repo | Key Command/URL |
|------|-------|------|-----------------|
| Run chat locally | Local | Public | `python app.py` -> localhost:8000 |
| Configure backups | Local | Public | localhost:8000/useradmin/ |
| Index local backups | Local | Public | `python local_cli.py index-html ...` |
| Submit source pack | Local | Public | useradmin -> Cloud Upload |
| Manage global sources | Parent | Private | `streamlit run admin/app.py` |
| Add scraped content | Parent | Private | `python ingest.py scrape ...` |
| Sync to Pinecone | Parent | Private | `python sync.py --remote pinecone push` |

### Source Creation Quick Reference

| Task | Code |
|------|------|
| Backup HTML site | `SourceManager().create_backup("id", "html", base_url="...")` |
| Index any source | `SourceManager().create_index("id")` |
| Validate source | `SourceManager().validate_source("id", config)` |
| Create pack | `SourceManager().create_pack("id", config)` |
| Index HTML directly | `index_html_backup(path, source_id)` |
| Index ZIM directly | `index_zim_file(path, source_id)` |
| Index PDFs directly | `index_pdf_folder(path, source_id)` |

---

## Other Documentation

- [SUMMARY.md](SUMMARY.md) - Executive summary (non-technical overview)
- [README.md](README.md) - Project overview
- [CONTEXT.md](CONTEXT.md) - Design decisions and rationale
- [ROADMAP.md](ROADMAP.md) - Future plans

---

*Last Updated: December 2025*
