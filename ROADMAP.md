# Disaster Clippy - Roadmap

Future plans and feature development priorities.

---

## Current Status: v1.0

### Completed Features

**Core Functionality:**
- Conversational search with 1000+ documents
- Vector embeddings (OpenAI or local sentence-transformers)
- Source attribution with clickable links
- Document classification (Guide, Article, Research, Product)
- External API for embedding on other sites
- Metadata index for fast sync

**Admin Panel:**
- FastAPI admin dashboard at /useradmin/
- 5-step Source Tools wizard
- Source validation with status boxes (Config, Backup, Metadata, Embeddings, License)
- Install/download cloud source packs
- Auto-discovery of indexed sources
- License compliance tracking

**Content Tools:**
- HTML backup system for offline browsing
- PDF ingestion with intelligent chunking
- Substack scraper with paid content support
- Web scrapers (MediaWiki, Fandom, static sites)
- ZIM file indexing

**Infrastructure:**
- Unified codebase (merged private/public repos)
- CLI tools for local admin, ingestion, and sync
- Cloudflare R2 cloud storage integration
- Pinecone sync functionality
- Cloud sync (Pinecone)

**Refactoring Completed (Dec 2025):**
- Folder reorganization (`admin/`, `offline_tools/`, `cli/`)
- Scrapers ported from private repo to `offline_tools/scraper/`
- CLI tools consolidated into `cli/` folder
- Route refactoring (extracted API into modular route files)
- Removed all legacy/backwards compatibility code
- Deleted empty folders and duplicate configs

---

## In Progress (v1.5)

### ADMIN_MODE Gating

Gate global-only features so local admins don't see them.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- `ADMIN_MODE` environment variable (defaults to "local")
- `require_global_admin()` dependency blocks cloud write endpoints
- `/api/admin-mode` endpoint for frontend feature detection
- Protected endpoints: `/api/upload-backup`, `/api/pinecone-push`, `/api/pinecone-namespace`
- Frontend banners and disabled buttons when not in global mode
- API key security remains primary layer (R2/Pinecone credentials required)

---

### Schema Standardization

Unified file naming and schema version across all tools.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- All file creation uses `schemas.py` getter functions
- Removed "v3 schema" references - current schema is the only schema
- Legacy file patterns documented in DEVELOPER.md for cleanup
- TESTING_CHECKLIST.md created for pipeline validation

---

### ZIM Metadata Extraction

Auto-populate source metadata from ZIM header_fields.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- ZIMIndexer extracts metadata from header_fields (title, description, license, creator, language, tags)
- Auto-populates _manifest.json with ZIM metadata during indexing
- User edits preserved (existing values not overwritten on re-index)
- Stores ZIM-specific fields: language, publisher, zim_date

---

### Tag Taxonomy Expansion

Expanded topic keywords for better automatic tag suggestions.

**Status:** COMPLETED (Dec 2025)

**Implemented:**
- Expanded from 8 to 28 tag categories in TOPIC_KEYWORDS
- Categories: water, sanitation, solar, energy, wind, biogas, fuel, food, agriculture, livestock, aquaculture, foraging, shelter, construction, tools, medical, herbal, mental-health, nutrition, emergency, fire, earthquake, flood, hurricane, nuclear, pandemic, navigation, communication, security, knots, appropriate-tech, electronics, vehicles, reference, how-to
- Increased suggestion limit from 5 to 10 tags
- Helps categorize content for cross-source search

---

### Pipeline Testing

Validate the 5-step wizard and file creation tools work correctly.

**Status:** In Progress

**Test Areas:**
- Fresh source creation (HTML backup)
- ZIM file indexing
- PDF collection workflow
- Source rename
- Cleanup redundant files
- Cloud download

---

## Near Term

### PDF Collection System

Structured ingestion and management for PDF documents.

**Status:** Planning

**Features:**
- Collection-based organization (group PDFs by topic/author)
- Two-level metadata (collection + document)
- DOI detection and CrossRef citation lookup
- License classification (public domain, open access, restricted)
- R2 hosting for public domain PDFs
- Smart chunking with section header detection

**Input Methods:**
- Single PDF file
- Folder of PDFs
- ZIP archive
- URL download

**Collection Structure:**
```
pdf_inbox/
  flu_preparedness/
    _collection.json    # Collection metadata
    FluSCIM_Guide.pdf
    CDC_Plan.pdf
  uncategorized/        # Default landing spot
```

**CLI Commands:**
```bash
python cli/ingest.py pdf add <file_or_folder> --collection <name>
python cli/ingest.py pdf list
python cli/ingest.py pdf create-collection <name> --license <type>
```

---

### Search Result Diversity

Balance results across sources so large collections don't dominate.

**Status:** Planning

**Problem:** Large sources (e.g., 135 PDF chunks) dominate search results, drowning out smaller but potentially more relevant sources.

**Solution:**
1. Source-aware re-ranking with max results per source
2. Title/exact match boosting
3. Fetch more results, then diversify

**Implementation:**
```python
def diversify_results(articles: List[dict], max_per_source: int = 2) -> List[dict]:
    """Re-rank results to ensure diversity across sources."""
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

---

### Export Formats

Support multiple export formats for different use cases.

**Status:** Planning

**Formats:**

| Format | Use Case | Add/Remove | Best For |
|--------|----------|------------|----------|
| Folder + manifest | Working copy | Easy | Active development |
| ZIP (.dcpack) | Distribution | Rebuild | Sharing collections |
| ZIM | Offline browsing | Rebuild | HTML website snapshots |

**ZIM Conversion (Future):**
- Convert HTML backups (builditsolar, solarcooking) to ZIM
- Requires zimwriterfs tool
- Best for stable, rarely-updated content
- Enables Kiwix offline browsing

---

### Knowledge Map Visualization

Interactive graph showing document relationships based on embedding similarity.

**Status:** Planning

**Features:**
- UMAP projection of embeddings to 2D
- Plotly scatter plot colored by source/doc_type
- Hover for document details
- Outlier detection (low neighbor similarity)
- Community detection for topic clusters

**Purpose:**
- Visualize source overlap
- Find topic gaps
- Identify misclassified content
- Detect redundancy

---

### Government Scrapers

Add FEMA, Cal Fire, EPA sources.

**Status:** Partial (PDF ingestion ready, need specific scrapers)

**Targets:**
- FEMA preparedness guides (Public Domain)
- Cal Fire wildfire resources (Public Domain)
- EPA emergency response (Public Domain)
- CDC health emergency guides

**Approach:**
- PDF scraper for downloadable reports
- Static site scraper for HTML guides
- Auto-detect Public Domain license

---

## Medium Term (v2.0)

### Location-Aware Search

Prioritize locally-relevant content based on user location.

**Features:**
- Location metadata on documents (state, county, region)
- Extract location from user queries
- Boost location-specific results
- Display location badges in results

**Implementation:**
- Add location fields to ScrapedPage
- NER or regex for location extraction
- Location filtering in vector search
- Session-based location storage

---

### ZIM as Foundation Layer

Use Kiwix ZIM format as the primary backup/distribution format.

**Why ZIM:**
- Single file distribution (not thousands of HTML files)
- Built-in full-text search
- Works in Kiwix readers on all platforms
- Wikipedia uses it (proven at scale)
- Easy to torrent/IPFS distribute

**Layer Architecture:**
```
Layer 3: AI/RAG (Optional, requires API keys)
Layer 2: Vector Index (Optional, semantic search)
Layer 1: Kiwix/ZIM (Foundation, always works offline)
```

**User Entry Points:**
- Emergency user: Just the ZIM, browse offline
- Power user: ZIM + local vectors
- Full experience: ZIM + vectors + AI

**Distribution:**
```
disaster-clippy-core.zim     # 500MB - Essential survival
disaster-clippy-medical.zim  # 200MB - Medical deep dive
disaster-clippy-solar.zim    # 150MB - Solar/energy
```

---

### Offline AI Assistant

Provide AI capabilities without internet.

**Option A: Pre-trained Small Model (2-4 GB)**
- Phi-3, Llama-3.2, or Mistral 7B quantized
- Runs on CPU (8GB RAM minimum)
- Fine-tuned system prompt for disaster prep domain
- Via Ollama or llama.cpp

**Option B: Cached Response Database (50-200 MB)**
- Pre-computed answers to 10,000+ common questions
- Fuzzy matching to find similar questions
- No inference needed, instant responses
- Works on any device including phones

**Option C: Hybrid (Recommended)**
- Check cached answers first (instant)
- If no match + local model: run inference
- If no match + no model: show relevant ZIM pages
- Always offer "browse in Kiwix" fallback

**Hardware Tiers:**

| Tier | RAM | Capability |
|------|-----|------------|
| Phone/RPi | 512MB-2GB | Cached answers + ZIM browse |
| Old Laptop | 4-8GB | + Phi-3 Mini (slow) |
| Modern Laptop | 8-16GB | + Llama-3.2/Mistral-7B |
| Desktop GPU | 16GB+ | + Larger models, fast |

**Offline Download Packages:**
- Minimal (500 MB): ZIM only, browse and search
- Standard (700 MB): + cached AI answers
- Full (3 GB): + local LLM model
- Power User (5 GB): + larger model, dev tools

---

## Long Term (v3.0+)

### Source Packs & Marketplace

Enable users to create, share, and install curated content packs.

**Concept:**
A "Source Pack" is a curated, shareable bundle containing indexed content that users can install into their own Clippy instance.

**Pack Contents:**
```
solar-cooking-pack/
  manifest.json        # Pack metadata, version, author
  sources.json         # Source definitions, scraper configs, licenses
  vectors.parquet      # Pre-computed embeddings (optional)
  metadata/            # Document metadata (titles, URLs, hashes)
```

**Quality Tiers:**

| Tier | Requirements | Visibility |
|------|--------------|------------|
| Personal | Any state, messy OK | Private only |
| Community | 80%+ indexed, licenses reported | Public, peer reviewed |
| Official | 100% indexed, verified licenses, backups complete | Featured, maintainer reviewed |

**Pack Health Checklist:**
- All sources have verified licenses
- No license conflicts
- 100% of pages indexed successfully
- HTML/ZIM backups complete for offline use
- No broken source URLs
- Metadata complete

**Download Options (Hybrid Model):**
- Quick Start: Vectors only (small, fast queries)
- Standard: Vectors + metadata (can re-scrape)
- Full Offline: Vectors + ZIM backups (works like Kiwix)
- Source Configs Only: Just scraper configs (user runs scrapers)

---

### Multi-User Platform

Transform from single-user admin to multi-user platform.

**User Journey:**

```
Stage 1: Anonymous Browser
  - Uses main app, queries all official sources
  - No account needed

Stage 2: Registered User
  - Creates account
  - Customizes source selection (checkboxes)
  - Can upload private docs
  - Preferences saved to profile

Stage 3: Power User
  - Exports config + downloads packs
  - Sets up local offline system
  - Can sync preferences with main app

Stage 4: Self-Hoster
  - Full clone of system
  - Own Pinecone + own backups
  - Can contribute back to official
```

**Source Selection UI:**
```
My Source Preferences:
  [x] Solar Cooking (176 docs)
  [x] Medical Emergency (500 docs)
  [ ] Homesteading (1200 docs)
  [x] My Uploads (23 docs)

  [Save] [Export for Offline]
```

---

### Account Tiers

**Free Tier:**
- Query all official sources
- Save preferences
- 10 private doc uploads
- Rate limited
- Export: vectors only

**Pro Tier ($5/mo):**
- Unlimited queries
- 100 private uploads
- Priority queue
- Export: full backups included
- Community pack submissions

**Self-Hosted (free, DIY):**
- Everything runs on their hardware
- No ongoing cost to project
- Can sync preferences with main app
- Contributor status

---

## Version Targets

| Version | Focus | Key Features |
|---------|-------|--------------|
| v0.5 | Unified codebase | Merged repos, admin dashboard, scrapers (COMPLETE) |
| v0.75 | Production polish | ADMIN_MODE gating, schema updates, testing (IN PROGRESS) |
| v1.0 | Offline | ZIM distribution, local AI, source packs |
| v2.0 | Platform | User accounts, marketplace, federated queries |

---

## Technical Debt

### Known Bugs

**EMBEDDING_MODE=local not respected**
- Location: `offline_tools/embeddings.py`
- Issue: EmbeddingService always tries to initialize OpenAI client even when EMBEDDING_MODE=local
- Impact: Crashes without OPENAI_API_KEY even if user wants local embeddings
- Fix: Check EMBEDDING_MODE before initializing OpenAI client

### Testing
- Unit tests for scrapers (mock HTTP)
- Integration tests for search flow
- Load tests for concurrent sessions

### Code Quality
- Type hints (Python 3.10+)
- Docstrings (Google style)
- Linting (ruff)
- CI/CD with GitHub Actions

### Documentation
- API documentation (OpenAPI/Swagger)
- Deployment runbook
- Scraper development guide
- Pack creation guide

---

## Cost Estimates

**Project Infrastructure:**
- Pinecone Serverless: ~$3-10/mo (1-10M vectors)
- Cloudflare R2: ~$1-5/mo (100GB, free egress)
- Railway: ~$5-20/mo (backend hosting)
- Total: ~$10-35/mo at moderate scale

**User Costs (self-hosted):**
- Pinecone free tier: 100K vectors ($0)
- Local storage: their drives ($0)
- Optional cloud backup: ~$5/mo

---

*Last Updated: December 2025*
