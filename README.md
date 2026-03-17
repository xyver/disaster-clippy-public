# Disaster Clippy

**Conversational search over your own data, online, offline, and locally.**

Disaster Clippy is an offline-capable AI search system for any curated body of knowledge. It combines a hosted demo, a local runtime, and advanced local admin/source tools around a shared collection and source-pack model.

It is not just a chat app and not just an admin tool. The project is meant to support four related surfaces:

- hosted demo app
- product and documentation site
- downloadable local runtime
- downloadable advanced local admin toolkit

---

## The Platform Behind It

Disaster preparedness is the first example deployment. The engine underneath is general purpose.

The core idea is simple: most AI systems hide what they know and you have to trust them. This one makes the knowledge layer visible, swappable, and verifiable. Every answer cites its source. You control what goes in. If something looks wrong, you can check it yourself.

That approach works for any curated body of knowledge. The same engine has been used for:

- **Building codes** — searchable local and regional code references for construction and compliance
- **Pandemic preparedness** — flu response protocols and field guidance for organizational planning
- **Humanitarian reference** — offline-capable search across Appropedia, Akvo, and similar open knowledge bases

The source pack model is what makes this portable. On the public site and in the app, people explore collections. Under the hood, a pack is the discrete, versioned bundle of knowledge — HTML backups, ZIM archives, PDFs, embeddings, and metadata — that can be installed locally, carried offline, or swapped out entirely. The runtime does not care what the pack contains.

If you are a developer thinking about building something similar for your own domain, the architecture is designed with that in mind. See [`docs/clippy-core-extraction.md`](docs/clippy-core-extraction.md) for the direction the portable core is heading, and [`docs/template-deployments.md`](docs/template-deployments.md) for the planned plug-and-play deployment path.

---

## What It Does

Users ask questions in plain language and get grounded answers from curated collections.

Example questions:

- "What are the main code requirements for rooftop solar in this jurisdiction?"
- "Show me the strongest references in this internal operations manual"
- "What does this humanitarian guide say about water storage?"
- "Find practical how-to material across this archive"

The system is designed so the same general experience can work:

- in a hosted demo app
- on a laptop or local server
- on a Raspberry Pi or air-gapped node

---

## Product Model

### Hosted Demo

The hosted demo is the easiest way to experience Disaster Clippy.

It is intended for:

- people exploring the system for the first time
- teams validating a collection before self-hosting
- users who want search/chat without running admin tooling

### Local Runtime

The local runtime is for people who want resilience, control, and their own data.

It is intended for:

- offline setups
- LAN/Raspberry Pi deployments
- organizations or individuals carrying selected collections locally

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

On the public site and in the app, these are presented as collections. In the runtime and tooling, packs are the portable unit underneath.

---

## Quick Start

### 1. Use the Hosted Demo

Current hosted demo:

- `https://app.disasterclippy.com`

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
curl -X POST "https://app.disasterclippy.com/api/v1/chat" \
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

- natural language search across curated collections
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
