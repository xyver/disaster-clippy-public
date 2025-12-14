# System Architecture

This document covers the core architecture of Disaster Clippy, including deployment modes, security, data flow, and the offline-capable design.

---

## Table of Contents

1. [Mode Switching](#mode-switching)
2. [Security Architecture](#security-architecture)
3. [Data Flow](#data-flow)
4. [Offline Architecture](#offline-architecture)
5. [URL Handling](#url-handling)
6. [Performance Guidelines](#performance-guidelines)

---

## Mode Switching

The same codebase supports both local and global admin modes:

```bash
# Local admin (default)
python app.py

# Global admin
ADMIN_MODE=global python app.py
```

| Feature | Local Admin | Global Admin |
|---------|-------------|--------------|
| Vector DB | ChromaDB (local) | Pinecone (cloud) |
| R2 backups/ | Read only | Read/Write |
| R2 submissions/ | Write only | Read only |
| Pinecone page | Hidden | Visible |
| Submissions page | Hidden | Visible |

### Access Levels

| Role | Vector DB | R2 Storage | What They Use |
|------|-----------|------------|---------------|
| End User | ChromaDB (local) | Read backups/ only | Chat interface |
| Local Admin | ChromaDB (local) | Write submissions/, Read backups/ | Admin panel + CLI tools |
| Global Admin | Pinecone (write) | Full access | Admin panel (global mode) |

---

## Security Architecture

### Deployment Modes (VECTOR_DB_MODE)

A single environment variable controls your deployment mode:

| Mode | Admin UI | Vector DB | R2 Access |
|------|----------|-----------|-----------|
| `local` (default) | Yes | ChromaDB | Read backups, R/W submissions |
| `pinecone` | No | Pinecone (read) | None (public chat only) |
| `global` | Yes | Pinecone (R/W) | Full access |

### Feature Access by Mode

| Feature | local | pinecone | global |
|---------|-------|----------|--------|
| Chat UI | Yes | Yes | Yes |
| Admin UI (`/useradmin/`) | Yes | No | Yes |
| Local ChromaDB | Yes | No | No |
| Pinecone Read (search) | Via proxy | Yes | Yes |
| Pinecone Write (sync) | No | No | Yes |
| R2 Read (download backups) | Via proxy | No | Yes |
| R2 Write (submissions) | Via proxy | No | Yes |
| R2 Full (official backups) | No | No | Yes |

### Rate Limits (Railway)

| Endpoint | Limit | Purpose |
|----------|-------|---------|
| `/chat`, `/api/v1/chat` | 10/min | Chat requests |
| `/api/cloud/sources` | 30/min | List sources |
| `/api/cloud/download/{id}` | 10/min | List files |
| `/api/cloud/download/{id}/{file}` | 5/min | Download file |
| `/api/cloud/submit` | 5/min | Submit content |

### Railway Proxy for Local Admins

Local admins without API keys can access cloud features through Railway proxy endpoints:

```
GET  /api/cloud/sources          - List available backups
GET  /api/cloud/download/{id}    - Stream backup download
POST /api/cloud/submit           - Submit content for review
```

Configure with `RAILWAY_PROXY_URL` in local settings.

---

## Data Flow

```
LOCAL ADMIN                              GLOBAL ADMIN
-----------                              ------------

[Local Backup Folder]                    [R2 Cloud Storage]
      |                                         ^
      v                                         |
[Validation] -----> [R2 submissions/] --------->+
      |              (review queue)             |
      v                                         v
[ChromaDB]                               [Validation]
(local search)                                  |
                                                v
                                         [R2 backups/]
                                         [Pinecone]
                                         (production)
```

**Key Points:**
- BACKUP_PATH is the working directory for all source data
- Global admin uploads finished sources to Pinecone + R2 (source of truth)
- Local users download from R2 to their BACKUP_PATH
- Local admins can submit sources to R2 `submissions/` for review

---

## Offline Architecture

The system uses a dual embedding architecture to support both online and offline semantic search.

### Dual Embedding Standard

| Context | Dimension | Model | Purpose |
|---------|-----------|-------|---------|
| Online (Pinecone) | 1536 | OpenAI text-embedding-3-small | Cloud search |
| Offline (ChromaDB) | Variable | Local model (384/768/1024) | Local search |

**Why two dimensions?** To do semantic search, the query must be embedded in the SAME dimension as stored vectors. Offline mode has no API access to generate 1536-dim query embeddings, so we use a local model instead.

**Local model examples:**
- `all-MiniLM-L6-v2` - 384-dim
- `all-mpnet-base-v2` - 768-dim (default)
- `bge-large-en-v1.5` - 1024-dim

### User Tiers

| Tier | Hardware | Role | Primary Actions |
|------|----------|------|-----------------|
| **Consumer** | RPi5 / Field device | End user | Download packs, search, browse |
| **Local Admin** | Laptop 8-16GB | Content creator | Create sources, submit to global |
| **Global Admin** | Desktop + API | Curator | Review, re-embed, publish |
| **Super Powered** | Cloud/GPU farm | Heavy processing | Parallel API, mass indexing |

### Tier Capabilities

| Capability | Consumer | Local Admin | Global Admin |
|------------|----------|-------------|--------------|
| Download packs | Yes | Yes | N/A |
| Search (semantic) | Yes (local) | Yes (local) | Yes (both) |
| Create sources | No | Yes | Yes |
| Embed documents | No | Yes (any dim) | Yes (768+1536) |
| Submit to global | No | Yes | N/A |
| Pinecone write | No | No | Yes |
| R2 backups write | No | No | Yes |

### Two-Pack Model System

Users download two separate model packs for full offline capability:

**Pack 1: Embedding Model (Essential)**
- Model: all-mpnet-base-v2 (or configured alternative)
- Size: ~420MB
- Dimensions: 768 (default, configurable)
- Purpose: Enable semantic search offline

**Pack 2: LLM Model (Enhancement)**
- Model: Llama 3.2 3B Q4 (recommended)
- Size: ~2GB
- Purpose: Generate AI responses from search results

| Installed | Search Type | Response Type | Experience |
|-----------|-------------|---------------|------------|
| Neither | Keyword | Raw results | Basic |
| Embedding only | **Semantic** | Raw results | Good |
| LLM only | Keyword | AI response | Mixed |
| **Both** | **Semantic** | **AI response** | **Full offline** |

### LLM Model Options

| Tier | Model | Size | RAM | Speed | Use Case |
|------|-------|------|-----|-------|----------|
| Lite | TinyLlama 1.1B Q4 | 700MB | 2GB | 15-30 tok/s | Very constrained |
| **Standard** | **Llama 3.2 3B Q4** | **2GB** | **4GB** | **8-15 tok/s** | **Recommended** |
| Full | Mistral 7B Q4 | 4GB | 8GB | 5-10 tok/s | Desktop |

### Models Folder Structure

```
BACKUP_PATH/
|-- models/
|   |-- _models.json          # Model registry
|   |-- embeddings/
|   |   |-- all-mpnet-base-v2/
|   |       |-- _manifest.json
|   |       |-- pytorch_model.bin
|   |       |-- config.json
|   |       |-- tokenizer.json
|   |-- llm/
|       |-- llama-3.2-3b-q4/
|           |-- _manifest.json
|           |-- Llama-3.2-3B-Instruct-Q4_K_M.gguf
|-- chroma_db/                # Local vectors
```

### File Structure (Dual Vectors)

```
{source_id}/
    _manifest.json
    _metadata.json
    _index.json
    _vectors_768.json    # Offline (downloaded by users)
    _vectors_1536.json   # Online (Pinecone + backup)
```

Global Admin creates BOTH. Users download offline vectors only.

### RPi5 Consumer Experience

**Memory Budget (8GB RPi5):**
```
OS + ChromaDB:        ~1.5GB
Embedding model:      ~0.5GB
LLM (3B quantized):   ~2.0GB
Buffer:               ~1.0GB
---------------------------------
Total:                ~5GB of 8GB
```

**Typical Response Time:**
```
Query embedding:      200-500ms
ChromaDB search:      50-100ms
Retrieve docs:        50ms
LLM generation:       ~15 seconds (150 tokens @ 10 tok/s)
---------------------------------
Total:                ~15-20 seconds
First query:          +15-30 sec for model loading
```

### GPU Acceleration

| Component | Bottleneck | GPU Benefit |
|-----------|------------|-------------|
| Backups | Network bandwidth | None |
| Metadata generation | Text parsing (CPU) | Minimal |
| Local indexing | Local embedding model | HIGH - matrix math |
| 1536-dim indexing | OpenAI API calls | None |
| Local LLM (chat) | Model inference | HIGH - tensor ops |
| Translation | Local model inference | HIGH |

GPU mainly accelerates offline-capable parts - exactly what you want for disaster scenarios.

---

## URL Handling

The indexer stores both online and local URLs for each document:

| Source Type | `url` Field | `local_url` Field |
|-------------|-------------|-------------------|
| ZIM | Online URL (e.g., `https://en.bitcoin.it/wiki/Mining`) | `/zim/{source_id}/{article_path}` |
| HTML | `base_url` + relative path | `/backup/{source_id}/{filename}` |
| PDF | `/api/pdf/{source_id}/{filename}` | `file://{local_path}` |

### Context-Aware Display

The `_get_display_url()` helper in `app.py` selects the appropriate URL:

| Deployment | Vector DB | URL Used | Result |
|------------|-----------|----------|--------|
| Railway (`PUBLIC_MODE=true`) | Pinecone | `url` | Online URLs |
| Local admin | ChromaDB | `local_url` | Local viewer URLs |

---

## Performance Guidelines

Storage is cheap (assume 1TB drives on most deployments), but user-facing latency is expensive. Optimize for read/scan speed, not storage size.

### Key Principles

1. **Never load full `_vectors.json` or `_index.json` into memory** unless absolutely necessary
   - These files can be 100MB+ for large sources
   - Use `read_json_header_only()` for counts/stats
   - Stream or batch process when you need content

2. **Use header reading for counts/stats**
   - `_master.json` has header fields (`source_count`, `total_documents`, `total_size_bytes`) before the large `sources` dict
   - `read_json_header_only()` in `packager.py` reads first 2KB only

3. **Stream large operations with progress callbacks**
   - All indexing operations use `progress_callback(current, total, message)`
   - Show users what's happening during long operations

4. **Background jobs for anything over ~2 seconds**
   - Use `JobManager.submit()` for delete, index, metadata generation
   - Jobs page shows progress and allows cancellation

5. **Use `read_only=True` when you don't need to write**
   - `VectorStore(read_only=True)` skips loading the embedding model
   - Saves ~2 seconds on startup for read-only operations

### Optimized Operations

| Operation | Before | After |
|-----------|--------|-------|
| Dashboard stats | Load all source folders | Read 2KB header from `_master.json` |
| Delete source IDs | Scan entire ChromaDB | Read keys from `_vectors.json` |
| Vector count check | ChromaDB where query | Read from `_vectors.json` |
| Internet check | 3-sec network call every page | 30-sec cached result |
| R2 source list | Fetch on every request | 1-hour cache |
| VectorStore init | Load embedding model | `read_only=True` skips model |

### Why Duplicate Storage is OK

The system intentionally stores data in multiple formats:

| Storage | Purpose | Optimized For |
|---------|---------|---------------|
| `_vectors.json` | Portable, shareable | Distribution, ID lookup |
| `_index.json` | Full content | Display, content scanning |
| `_metadata.json` | Lightweight metadata | Quick lookups, stats |
| ChromaDB | Vector similarity | Fast semantic search |

Duplicating data with increased storage is a fair trade for faster document scanning and operations.

---

## Related Documentation

- [Deployment Scenarios](deployment.md) - How to deploy in different environments
- [Validation System](validation.md) - Source validation gates and checks
- [Jobs System](jobs.md) - Background job processing
- [Language Packs](language-packs.md) - Offline translation for non-English users

---

*Last Updated: December 2025*
