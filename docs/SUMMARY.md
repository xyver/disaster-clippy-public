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

### Admin Panel Option for Advanced Users
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

## User Types

| User | What They Do |
|------|--------------|
| End User | Ask questions, get answers |
| Local Admin | Run their own instance, add personal sources |
| Global Admin | Curate official sources, manage cloud infrastructure |

---

## Current State (v0.9 Pre-release)

**What Works Now:**
- Chat interface with source citations and filtering
- 10+ curated sources (Appropedia, Wikipedia, medical guides, disaster prep)
- Admin panel with 5-step Source Tools wizard
- Three connection modes: Online Only, Hybrid (recommended), Offline Only
- Knowledge Map 3D visualization showing document relationships
- ZIM/HTML/PDF file support with metadata extraction
- Web scrapers for adding new sources (MediaWiki, static sites, PDF)
- Personal cloud backup (connect your own S3/R2/Backblaze storage)
- Job management with checkpoint/resume for interrupted tasks
- Source validation with status indicators
- Download pre-packaged source packs
- Public cloud deployment (Railway) + local self-hosting options

**Currently:**
- Final testing and polish for v1.0 release

---

## Roadmap Highlights

### Next (v1.0 Official Release)
- Stable release with comprehensive testing
- Expanded source library (more medical, survival, DIY guides)
- Performance optimizations for large ZIM files
- Improved mobile experience

### Future (v2.0+)
- Community contributions and source submissions
- Mobile app with offline-first capability
- Enhanced visualizations and analytics
- Multi-language interface support

---

## Links

- Public Repository: https://github.com/xyver/disaster-clippy-public
- Live Demo: https://disaster-clippy.up.railway.app/

---

## Related Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | Quick start and project overview |
| [DEVELOPER.md](DEVELOPER.md) | Technical details and CLI tools |
| [CONTEXT.md](CONTEXT.md) | Architecture (AI onboarding) |
| [ROADMAP.md](ROADMAP.md) | Future plans and testing |

---

*Last Updated: December 10, 2025*
