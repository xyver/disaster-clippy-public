# Disaster Clippy

**Evidence-based emergency preparedness guidance through conversational AI**

Disaster Clippy helps you find actionable information from trusted sources - educational guides, DIY projects, government reports, and research papers. Ask questions in plain language and get recommendations with source attribution.

**Fully offline-capable** - runs on a Raspberry Pi with no internet required.

---

## What Can I Ask?

- "How do I purify water in an emergency?"
- "What's the best way to build a solar cooker?"
- "Show me guides on food preservation"
- "How do I prepare for a wildfire?"

The system searches thousands of documents from trusted sources and provides answers with links to the original content.

---

## Quick Start

### Option 1: Use the Hosted Version

Try it now: **https://disaster-clippy.up.railway.app/**

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/xyver/disaster-clippy-public.git
cd disaster-clippy-public

# Install dependencies
pip install -r requirements.txt

# Configure (add your API keys)
cp .env.example .env
# Edit .env with your OpenAI API key

# Start the chat interface
python app.py
# Visit http://localhost:8000
```

### Option 3: Run Fully Offline

```bash
# Use local embeddings (no API key needed)
EMBEDDING_MODE=local python app.py

# Install Ollama for offline AI responses
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
```

### Option 4: Use the API

Embed Disaster Clippy on your website:

```bash
curl -X POST "https://disaster-clippy.up.railway.app/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I filter water?"}'
```

---

## Installation Profiles

Different requirements files for different use cases:

| File | Size | Use Case |
|------|------|----------|
| `requirements.txt` | ~2GB+ | Full local admin with offline AI, translation, ZIM browsing |
| `requirements-cloud.txt` | ~200MB | Cloud deployment (Railway) - chat only, no local ML |

### Full Installation (requirements.txt)

Includes everything for local administration and offline operation:
- **ChromaDB** - Local vector database
- **sentence-transformers** - Local embeddings (768-dim, no API needed)
- **transformers + sentencepiece** - Translation models (~300MB per language)
- **scikit-learn** - PCA for knowledge map visualization
- **zimply-core** - ZIM file reading for offline Wikipedia/WikiHow
- **beautifulsoup4 + lxml** - Web scraping

Best for: Local admins, offline deployments, Raspberry Pi

### Cloud Installation (requirements-cloud.txt)

Minimal dependencies for hosted chat interface:
- **Pinecone** - Cloud vector database (no local storage)
- **langchain-openai/anthropic** - API-based embeddings and chat
- **boto3** - R2 cloud storage for submissions

Excludes: ChromaDB, sentence-transformers, transformers, zimply, scikit-learn

Best for: Railway deployment, public chat instance, low-resource servers

### Installing

```bash
# Full installation (local admin)
pip install -r requirements.txt

# Cloud deployment (Railway)
pip install -r requirements-cloud.txt

# Optional: GPU acceleration for translation (CUDA)
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Content Sources

| Source | Topics |
|--------|--------|
| Appropedia | Appropriate technology, water, sanitation, shelter |
| BuildItSolar | DIY solar projects, heating, cooling |
| SolarCooking Wiki | Solar cooking, food safety |
| WikiHow | How-to guides across topics |
| PDF Collections | Pandemic preparedness, emergency management |
| Wikipedia (ZIM) | Medical, technology, and other topic archives |

All content is Creative Commons or Public Domain with full attribution.

---

## Features

### For Users

- **Natural language search** - Ask questions like you would ask a person
- **Source attribution** - Every answer links to the original source
- **Source filtering** - Search specific sources or all at once
- **Multi-turn conversation** - Follow-up questions keep context
- **Offline browsing** - View ZIM archives and HTML backups without internet

### For Local Admins

- **Admin Panel** - Full source management at `/useradmin/`
- **5-Step Source Tools** - Wizard for creating and editing sources
- **6 Status Boxes** - Visual validation (Config, Backup, Metadata, 1536, 768, License)
- **Validation Gates** - `can_submit` and `can_publish` permission system
- **Job System** - Background jobs with checkpoint/resume for long operations
- **Job Builder** - Visual UI for creating custom job chains
- **Web Scrapers** - MediaWiki, static sites, with link following
- **ZIM Support** - Index and browse Wikipedia, WikiHow, and other ZIM archives
- **Language Packs** - Download translation models for non-English users
- **CLI Tools** - Command-line tools for indexing, scraping, sync

### For Offline Deployment

- **Dual Embedding** - 1536-dim (cloud) + 768-dim (local) vectors
- **Local LLM** - Ollama integration for offline AI responses
- **ZIM Browsing** - Offline viewing of Wikipedia and other archives
- **Personal Cloud** - Sync to your own S3-compatible storage
- **RPi5 Ready** - Runs on Raspberry Pi 5 with 8GB RAM

---

## Project Structure

```
disaster-clippy/
|-- app.py                    # FastAPI chat interface
|-- local_settings.json       # User configuration
|
|-- cli/                      # Command-line tools
|-- admin/                    # Admin panel (/useradmin/)
|-- offline_tools/            # Core business logic
|-- templates/                # Main app templates
|-- static/                   # Main app static files
|-- docs/                     # Detailed documentation
```

---

## API Endpoints

### Chat API

**POST /api/v1/chat**

```json
// Request
{"message": "How do I build a solar oven?", "session_id": "optional"}

// Response
{"response": "Here are several approaches...", "session_id": "abc123"}
```

**POST /api/v1/chat/stream** - Server-sent events for streaming responses

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/health` | GET | Health check |
| `/api/v1/connection-status` | GET | Connection state |
| `/sources` | GET | List available sources |

---

## Documentation

| Document | Purpose |
|----------|---------|
| **Getting Started** | |
| [README.md](README.md) | This file - overview and quick start |
| [DEVELOPER.md](DEVELOPER.md) | Setup guide and documentation index |
| **Architecture** | |
| [docs/architecture.md](docs/architecture.md) | Modes, security, data flow, offline design |
| [docs/ai-service.md](docs/ai-service.md) | Search, chat, connection modes |
| **Working with Sources** | |
| [docs/source-tools.md](docs/source-tools.md) | SourceManager, indexers, scrapers |
| [docs/validation.md](docs/validation.md) | Permission gates, validation tiers |
| [docs/jobs.md](docs/jobs.md) | Background jobs, checkpoints |
| **Operations** | |
| [docs/deployment.md](docs/deployment.md) | Deployment scenarios, cloud backup |
| [docs/admin-guide.md](docs/admin-guide.md) | Admin panel, CLI, troubleshooting |
| [docs/language-packs.md](docs/language-packs.md) | Offline translation system |
| **Planning** | |
| [ROADMAP.md](ROADMAP.md) | Future plans and version targets |
| [CONTEXT.md](CONTEXT.md) | Architecture decisions (AI onboarding) |
| [SUMMARY.md](SUMMARY.md) | Executive summary (non-technical) |

---

## Tech Stack

- **Backend**: Python, FastAPI
- **AI**: OpenAI GPT-4o-mini, Claude, or local Ollama
- **Embeddings**: OpenAI API or local sentence-transformers
- **Vector Database**: ChromaDB (local) or Pinecone (cloud)
- **Translation**: MarianMT models for offline translation
- **Cloud Storage**: Cloudflare R2 (S3-compatible)

---

## License

MIT

Content sources retain their original licenses (CC-BY, CC-BY-SA, Public Domain).

---

## Contributing

See [DEVELOPER.md](DEVELOPER.md) for development setup.

Key areas:
- Add new content scrapers
- Improve search result diversity and weighting
- Test language pack translations
- Documentation and testing

---

*Version 0.9 (Pre-release) - December 2025*
