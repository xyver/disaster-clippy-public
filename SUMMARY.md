# Disaster Clippy - Executive Summary

**What it is:** An AI-powered search assistant for disaster preparedness, DIY guides, and humanitarian resources. Users ask questions in plain language and get answers sourced from curated, verified content.

---

## Core Features

### Chat Interface
- Natural language Q&A about disaster preparedness, survival skills, medical guides, and DIY projects
- Answers include source citations with links to original content
- Works online (cloud) or offline (local database)

### Source Library
- Curated collection of open-source guides (Appropedia, Practical Action, medical references)
- Each source is verified for licensing (Creative Commons, public domain)
- Sources can be browsed and filtered by topic

### Offline Capability
- Download source packs (ZIM archives, HTML backups) for offline use
- Run entirely without internet using local vector database
- Hybrid mode: local sources + cloud fallback

### Local Admin Panel
- 5-step Source Tools wizard for creating/editing sources
- Status boxes show validation state (Config, Backup, Metadata, Embeddings, License)
- Configure which sources to search
- Add personal documents (PDFs, saved websites)
- Submit new sources for community review

---

## How It Works

1. **User asks a question** in the chat interface
2. **AI searches** the vector database for relevant content chunks
3. **LLM synthesizes** an answer from the retrieved content
4. **Sources are cited** so users can verify and read more

---

## Architecture Overview

| Layer | Purpose |
|-------|---------|
| Frontend | Web chat interface (FastAPI + Jinja2) |
| Admin | Source management panel at /useradmin/ |
| Search | Vector similarity search (ChromaDB local / Pinecone cloud) |
| AI | Language model for answers (Claude or GPT) |
| Storage | Source backups and metadata (Cloudflare R2) |
| CLI | Command-line tools for indexing, scraping, sync |

### Folder Structure

```
disaster-clippy/
|-- app.py                    # Main chat app
|-- cli/                      # Command-line tools
|-- admin/                    # Admin panel (/useradmin/)
|-- offline_tools/            # Core business logic
|   |-- indexer.py            # HTML/ZIM/PDF indexing
|   |-- source_manager.py     # Source CRUD
|   |-- scraper/              # Web scrapers
|   |-- vectordb/             # ChromaDB/Pinecone
|   |-- cloud/                # R2 storage
```

---

## User Types

| User | What They Do |
|------|--------------|
| End User | Ask questions, get answers |
| Local Admin | Run their own instance, add personal sources |
| Global Admin | Curate official sources, manage cloud infrastructure |

---

## Current State (v1.0)

**Completed:**
- Chat with source citations
- 10+ indexed sources (wikis, guides, newsletters)
- FastAPI admin panel with Source Tools wizard
- Cloud deployment on Railway
- Offline ZIM/HTML/PDF backup support
- Web scrapers (MediaWiki, Fandom, static sites, PDF)
- CLI tools for local admin, scraping, sync
- Cloudflare R2 cloud storage
- Unified codebase (merged private/public repos)
- Source validation with status boxes
- Install/download cloud source packs
- Pinecone sync functionality

**In Progress:**
- ADMIN_MODE gating for global-only features
- Schema file naming updates

---

## Roadmap Highlights

### Near Term (v1.5)
- ADMIN_MODE gating for global admin features
- Schema file naming updates (_manifest.json, _vectors.json)
- PDF collection system
- Knowledge map visualization

### Medium Term (v2.0)
- Community contributions (submit and vote on sources)
- ZIM as primary distribution format
- Offline AI assistant (local LLMs)

### Long Term (v3.0)
- Mobile app with offline-first design
- Source pack marketplace
- Multi-user platform with accounts
- Mesh networking for source sharing

---

## Links

- Public Repository: https://github.com/xyver/disaster-clippy-public
- Live Demo: https://disaster-clippy.up.railway.app/

---

*Last Updated: December 2025*
