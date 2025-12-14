# Disaster Clippy - Developer Documentation

This guide covers development setup and provides links to detailed documentation.

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

## Repository Structure

```
disaster-clippy/
|-- app.py                    # FastAPI chat interface
|-- local_settings.json       # User configuration (single source of truth)
|
|-- cli/                      # Command-line tools
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
|   |-- templates/            # Admin UI templates
|   |-- static/               # Admin CSS/JS
|
|-- offline_tools/            # Core business logic
|   |-- schemas.py            # Data structures
|   |-- embeddings.py         # Embedding service
|   |-- indexer.py            # HTML/ZIM/PDF indexing
|   |-- source_manager.py     # Source CRUD operations
|   |-- validation.py         # Source validation
|   |-- translation.py        # Translation service
|   |-- language_registry.py  # Language pack management
|   |-- backup/               # Backup utilities
|   |-- cloud/                # Cloud storage (R2)
|   |-- scraper/              # Web scrapers
|   |-- vectordb/             # Vector database implementations
|
|-- templates/                # Main app templates
|-- static/                   # Main app static files
|-- docs/                     # Detailed documentation
```

---

## Key Modules

| Module | Purpose |
|--------|---------|
| `admin/ai_service.py` | Unified AI search and response service |
| `admin/connection_manager.py` | Smart connectivity detection and mode switching |
| `admin/job_manager.py` | Background job queue with checkpointing |
| `admin/zim_server.py` | ZIM content server for offline browsing |
| `offline_tools/source_manager.py` | Unified source creation interface |
| `offline_tools/validation.py` | Source validation (can_submit/can_publish gates) |
| `offline_tools/indexer.py` | Indexers for HTML, ZIM, PDF |
| `offline_tools/translation.py` | Translation service (MarianMT/NLLB) |
| `offline_tools/vectordb/` | Vector store implementations |
| `offline_tools/cloud/r2.py` | Cloudflare R2 client |

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
| `BACKUP_PATH` | Path to backup folder | No (defaults to ./backups) |

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

Your database is empty. Index some content:
1. Go to `/useradmin/` -> Sources tab
2. Use Source Tools to create a new source
3. Or install a cloud source pack

### "Unable to connect to OpenAI"

Check your API key in `.env` or switch to local embeddings:
```bash
EMBEDDING_MODE=local
```

### "Port 8000 already in use"

The system automatically tries port 8001. Or kill the process using port 8000.

---

## Documentation Index

### Architecture and Design
- [Architecture](docs/architecture.md) - Modes, security, data flow, offline architecture
- [AI Service](docs/ai-service.md) - Search, chat, connection modes

### Working with Sources
- [Source Tools](docs/source-tools.md) - SourceManager, indexers, scrapers, tags
- [Validation](docs/validation.md) - Permission gates, validation tiers, human verification

### Operations
- [Jobs](docs/jobs.md) - Background jobs, checkpoints, job builder
- [Admin Guide](docs/admin-guide.md) - Admin panel, CLI tools, troubleshooting

### Deployment
- [Deployment](docs/deployment.md) - Scenarios, cloud backup, R2, vector DB

### Features
- [Language Packs](docs/language-packs.md) - Offline translation for non-English users

### Other
- [README.md](README.md) - Project overview and quick start
- [CONTEXT.md](CONTEXT.md) - Architecture and design decisions (AI onboarding)
- [SUMMARY.md](SUMMARY.md) - Executive summary (non-technical)
- [ROADMAP.md](ROADMAP.md) - Future plans, testing, and feature development

---

*Last Updated: December 2025*
