# Disaster Clippy - AI Assistant Context

**Read this first.** This document helps you quickly understand the project and find the right documentation for whatever feature you're working on.

---

## What This Project Does (30 seconds)

Disaster Clippy is an **offline-capable AI search assistant** for emergency preparedness content. Users ask questions in natural language, the system searches curated sources (wikis, guides, PDFs), and returns answers with citations.

**Key concept:** Everything is designed to work **without internet** - on a Raspberry Pi in a disaster scenario.

---

## Quick Orientation

| Term | Meaning |
|------|---------|
| **Source** | A collection of documents (e.g., "appropedia", "wikihow-zim") |
| **BACKUP_PATH** | Local folder containing all sources and ChromaDB |
| **Local Admin** | User running their own instance, creates/manages sources |
| **Global Admin** | Maintainer of official cloud sources (Pinecone + R2) |
| **ZIM** | Compressed offline archive format (Wikipedia, WikiHow, etc.) |

---

## Documentation Map

### For Quick Setup
| Doc | When to Read |
|-----|--------------|
| [README.md](../README.md) | First time setup, API usage |
| [DEVELOPER.md](../DEVELOPER.md) | Dev environment, folder structure, env vars |

### For Understanding Architecture
| Doc | When to Read |
|-----|--------------|
| [architecture.md](architecture.md) | Modes (local/global), security, data flow, offline design |
| [ai-service.md](ai-service.md) | Search pipeline, connection states, LLM integration |

### For Working with Sources
| Doc | When to Read |
|-----|--------------|
| [source-tools.md](source-tools.md) | Creating sources, indexers, scrapers, ZIM tools |
| [validation.md](validation.md) | can_submit/can_publish gates, status boxes, human verification |

### For Background Jobs
| Doc | When to Read |
|-----|--------------|
| [jobs.md](jobs.md) | Job types, checkpoints, resume, job builder UI |

### For Deployment
| Doc | When to Read |
|-----|--------------|
| [deployment.md](deployment.md) | Self-hosted, RPi5, air-gapped, personal cloud backup |
| [admin-guide.md](admin-guide.md) | Admin panel usage, CLI tools, troubleshooting |

### For Features
| Doc | When to Read |
|-----|--------------|
| [language-packs.md](language-packs.md) | Translation system, MarianMT models |

### For Planning
| Doc | When to Read |
|-----|--------------|
| [ROADMAP.md](../ROADMAP.md) | What's done, what's in progress, future plans |

---

## What Are You Working On?

Use this to find the right files and docs:

### Chat / Search / AI Responses
```
Read: docs/ai-service.md
Files: app.py, admin/ai_service.py, admin/connection_manager.py
```

### Creating or Editing Sources
```
Read: docs/source-tools.md, docs/validation.md
Files: offline_tools/source_manager.py, offline_tools/indexer.py
UI: admin/templates/source_tools.html
```

### Background Jobs / Long Operations
```
Read: docs/jobs.md
Files: admin/job_manager.py, admin/routes/source_tools.py
UI: admin/templates/jobs.html
```

### ZIM Files (Wikipedia, WikiHow, etc.)
```
Read: docs/source-tools.md (ZIM Tools section)
Files: offline_tools/indexer.py (ZIMIndexer), admin/zim_server.py
CLI: cli/zim_inspect.py
```

### HTML Scraping / Backups
```
Read: docs/source-tools.md (HTML Backup Scraper section)
Files: offline_tools/backup/html.py, offline_tools/indexer.py (HTMLBackupIndexer)
```

### Validation / Status Boxes
```
Read: docs/validation.md
Files: offline_tools/validation.py
```

### Cloud Storage / R2 / Pinecone
```
Read: docs/deployment.md, docs/architecture.md
Files: offline_tools/cloud/r2.py, offline_tools/vectordb/pinecone_store.py
```

### Offline Mode / Local LLM
```
Read: docs/architecture.md (Offline Architecture section)
Files: admin/ai_service.py, admin/ollama_manager.py
```

### Translation / Language Packs
```
Read: docs/language-packs.md
Files: offline_tools/translation.py, offline_tools/language_registry.py
UI: admin/templates/sources.html (Languages tab)
```

### Admin Panel UI
```
Read: docs/admin-guide.md
Files: admin/app.py, admin/routes/*.py, admin/templates/*.html
```

### 3D Visualization / Knowledge Map
```
Read: docs/admin-guide.md (Knowledge Map section)
Files: admin/routes/visualise.py, admin/templates/visualise.html
```

---

## Key Files (Quick Reference)

| File | Purpose |
|------|---------|
| `app.py` | Main chat API, FastAPI routes |
| `admin/ai_service.py` | Search + response generation |
| `admin/job_manager.py` | Background job queue |
| `admin/connection_manager.py` | Online/offline detection |
| `offline_tools/source_manager.py` | Source creation pipeline |
| `offline_tools/indexer.py` | HTML/ZIM/PDF indexers |
| `offline_tools/validation.py` | Validation gates |
| `offline_tools/vectordb/store.py` | ChromaDB operations |
| `local_settings.json` | User configuration |

---

## Folder Structure (Simplified)

```
disaster-clippy/
|-- app.py                    # Chat API
|-- local_settings.json       # User config
|-- cli/                      # Command-line tools
|-- admin/                    # Admin panel + APIs
|   |-- routes/               # API endpoints
|   |-- templates/            # UI pages
|-- offline_tools/            # Core logic
|   |-- vectordb/             # ChromaDB/Pinecone
|   |-- scraper/              # Web scrapers
|   |-- cloud/                # R2 storage
|-- docs/                     # Documentation
```

---

## Current Status (v0.9)

**Recently completed:** Validation system, job builder, connection states, personal cloud backup, HTML scraper integration, language packs phase 1

**In progress:** Search result diversity tuning, pipeline testing

**See:** [ROADMAP.md](../ROADMAP.md) for full status

---

## Environment Variables (Essential)

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Embeddings + chat (or use EMBEDDING_MODE=local) |
| `VECTOR_DB_MODE` | `local` (default), `pinecone`, or `global` |
| `BACKUP_PATH` | Where sources live (default: ./backups) |

**Full list:** [DEVELOPER.md](../DEVELOPER.md#environment-variables)

---

*Last Updated: December 2025*
