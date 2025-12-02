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
| Frontend | Web chat interface (FastAPI) |
| Search | Vector similarity search (ChromaDB local / Pinecone cloud) |
| AI | Language model for answers (Claude or GPT) |
| Storage | Source backups and metadata (Cloudflare R2) |

---

## User Types

| User | What They Do |
|------|--------------|
| End User | Ask questions, get answers |
| Local Admin | Run their own instance, add personal sources |
| Global Admin | Curate official sources, manage cloud infrastructure |

---

## Current State

**Working:**
- Chat with source citations
- 10+ indexed sources (wikis, guides, newsletters)
- Local admin panel for configuration
- Cloud deployment on Railway
- Offline ZIM/HTML backup support

**In Progress:**
- Source pack submission workflow
- Expanded medical/emergency content
- Multi-language support

---

## Roadmap Highlights

- **Community contributions:** Let users submit and vote on sources
- **Mobile app:** Offline-first mobile version
- **Mesh networking:** Share sources between nearby devices
- **Print guides:** Generate printable PDFs from search results

---

## Links

- Public Repository: https://github.com/xyver/disaster-clippy-public
- Live Demo: (Railway deployment URL)

---

*Last Updated: December 2025*
