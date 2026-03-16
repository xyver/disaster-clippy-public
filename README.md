# Disaster Clippy

**Preparedness knowledge you can use online, offline, and locally.**

Disaster Clippy is an offline-capable AI search system for emergency preparedness, resilience, survival skills, medical references, and practical DIY knowledge. It combines a hosted app, a local runtime, and advanced local admin/source tools around a shared source-pack model.

It is not just a chat app and not just an admin tool. The project is meant to support four related surfaces:

- hosted public app
- product and catalog site
- downloadable local runtime
- downloadable advanced local admin toolkit

---

## What It Does

Users ask questions in plain language and get grounded answers from curated source packs.

Example questions:

- "How do I purify water in an emergency?"
- "What should I keep in a first aid kit?"
- "Show me practical guides on food preservation"
- "How can I heat a room safely during a winter outage?"

The system is designed so the same general experience can work:

- in a hosted public app
- on a laptop or local server
- on a Raspberry Pi or air-gapped node

---

## Product Model

### Hosted App

The hosted app is the easiest way to use Disaster Clippy.

It is intended for:

- normal end users
- people exploring the system for the first time
- users who want search/chat without running admin tooling

### Local Runtime

The local runtime is for users who want resilience and control.

It is intended for:

- offline setups
- LAN/Raspberry Pi deployments
- organizations or individuals carrying selected source packs locally

### Advanced Local Admin

The advanced admin layer is for power users and maintainers.

It includes:

- source creation and validation
- scraping and indexing
- translation workflows
- video processing workflows
- future OCR support for scanned PDFs

This is intentionally local-first. Disaster Clippy is not currently aiming to turn source creation into a public hosted workflow.

---

## Source Packs

Source packs are the main unit of knowledge distribution.

A source pack may include:

- HTML backups
- ZIM archives
- PDFs
- transcript and translation artifacts
- metadata and vector indexes

This lets users:

- understand what content they are using
- choose which sources to install or sync
- carry the same pack choices between hosted and local use

---

## Quick Start

### 1. Use the Hosted App

Current hosted app:

- `https://disaster-clippy.up.railway.app/`

### 2. Run Locally

```bash
# Clone the repository
git clone https://github.com/xyver/disaster-clippy-public.git
cd disaster-clippy-public

# Install dependencies
pip install -r requirements.txt

# Configure local settings
cp .env.example .env

# Start the app
python app.py
```

Then open:

- `http://localhost:8000`

### 3. Run More Offline

```bash
# Local embeddings
EMBEDDING_MODE=local python app.py

# Optional local LLM via Ollama
ollama pull llama3.2:3b
```

### 4. Use the API

```bash
curl -X POST "https://disaster-clippy.up.railway.app/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I purify water?"}'
```

---

## Installation Profiles

| File | Use Case |
|------|----------|
| `requirements.txt` | Full local runtime and advanced admin tooling |
| `requirements-cloud.txt` | Lean hosted/runtime deployment |

Use the full install when you want:

- local ChromaDB
- source tools
- translation models
- video processing
- ZIM/PDF handling
- offline-oriented workflows

Use the cloud install when you want:

- a smaller hosted deployment
- chat/search without the heavy local processing stack

---

## Core Capabilities

### For Users

- natural language search across curated preparedness knowledge
- grounded responses with source attribution
- source filtering
- multi-turn conversation
- hosted or offline-capable usage paths

### For Local Runtime Users

- local source storage
- offline semantic search
- local model support
- pack-based content installation

### For Advanced Local Admins

- source creation and validation
- background jobs and checkpointing
- translation and language pack workflows
- video transcript acquisition and processing
- future OCR preprocessing for scanned PDFs

---

## Project Structure

```text
disaster-clippy-public/
|-- app.py
|-- admin/
|-- cli/
|-- docs/
|-- offline_tools/
|-- static/
|-- templates/
```

The broader workspace direction is now:

```text
disaster-clippy/
+-- disaster-clippy-public/
+-- disaster-clippy-private/
```

Where:

- `disaster-clippy-public/` is the core runtime and local tooling repo
- `disaster-clippy-private/` is the `.com` site and future control-plane shell

---

## Docs To Read Next

- [`docs/deployment.md`](docs/deployment.md) - runtime model, public/private split, hosted vs local
- [`docs/architecture.md`](docs/architecture.md) - system architecture and runtime layers
- [`docs/source-tools.md`](docs/source-tools.md) - source creation and ingestion tools
- [`docs/language-packs.md`](docs/language-packs.md) - translation system
- [`docs/video_processing.md`](docs/video_processing.md) - video and transcript pipeline
- [`docs/clippy-core-extraction.md`](docs/clippy-core-extraction.md) - portable core architecture target

---

## License

MIT for project code.

Content sources keep their own original licenses.
