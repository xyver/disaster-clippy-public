# Offline Upgrade - Portable Model Packs

This document covers the implementation of portable LLM and embedding model packs for offline disaster preparedness. It is self-contained with all relevant code sections, architecture decisions, and implementation details.

**Goal:** Enable users to download and run semantic search + AI conversation completely offline.

---

## Table of Contents
1. [OpenAI Embedding API Considerations](#openai-embedding-api-considerations)
2. [Embedding Check Optimization](#embedding-check-optimization)
3. [Current Codebase State](#current-codebase-state)
4. [Current Embedding Chain (As-Is)](#current-embedding-chain-as-is)
5. [Target Embedding Chain (To-Be)](#target-embedding-chain-to-be)
6. [Two-Pack Model System](#two-pack-model-system)
7. [Folder Structure](#folder-structure)
8. [Model Manifest Schemas](#model-manifest-schemas)
9. [Runtime Decisions](#runtime-decisions)
10. [Local Admin Submission Validation](#local-admin-submission-validation)
11. [UI Design](#ui-design)
12. [Implementation Phases](#implementation-phases)
13. [Dual Embedding Architecture](#dual-embedding-architecture)
14. [User Tier System](#user-tier-system)
15. [RPi5 Consumer Tier Analysis](#rpi5-consumer-tier-analysis)
16. [Processing Time Reality](#processing-time-reality)
17. [Key Decisions Summary](#key-decisions-summary)

---

## OpenAI Embedding API Considerations

### Token Limits

OpenAI's `text-embedding-3-small` model has a limit of **8,191 tokens** per request.
- Average English word = ~1.3 tokens
- Roughly ~6,000 words per request
- Long articles (10,000+ words) will exceed this limit

### Current Chunking Strategy

Located in `offline_tools/embeddings.py`:

```
Original Document (15,000 words)
           |
           v
    [Check length]
           |
    [Exceeds limit?]
           |
     YES   |   NO
           v
    [Split at sentence boundaries]
           |
           v
    [Chunk 1]  [Chunk 2]  [Chunk 3]
        |          |          |
        v          v          v
    [Embed]    [Embed]    [Embed]
        |          |          |
        v          v          v
    [1536-dim] [1536-dim] [1536-dim]
        \          |          /
         \         |         /
          v        v        v
         [Average all embeddings]
                   |
                   v
            [Final 1536-dim vector]
```

**Implementation details:**
- MAX_CHARS = 32,000 (conservative estimate)
- Splits recursively at sentence boundaries
- Each chunk embedded separately
- Final embedding = element-wise average of all chunk embeddings

### Why Averaging Works for This Use Case

For disaster preparedness content:
- Most articles have a clear central topic
- Users search for topics, not specific paragraphs
- Averaged embedding still captures the main subject
- Simpler retrieval (1 doc = 1 vector)

---

## Embedding Check Optimization

### Problem

When re-indexing, we check if existing documents have valid embeddings.
Current implementation: O(n) individual DB calls for n existing documents.

### Solution: Batch + Parallel Approach

Instead of individual calls, batch in chunks:

```python
# Batch in chunks of 1000 to avoid OOM
batch_size = 1000
bad_ids = set()

for offset in range(0, len(existing_ids), batch_size):
    batch_ids = list(existing_ids)[offset:offset + batch_size]
    result = collection.get(ids=batch_ids, include=["embeddings"])

    for doc_id, embedding in zip(result["ids"], result["embeddings"]):
        if embedding is None or all(v == 0 for v in embedding):
            bad_ids.add(doc_id)
```

**Performance:**
- 13,000 docs = 13 batch calls instead of 13,000 individual calls
- Memory bounded at ~40MB per batch (1000 * 1536 floats * 4 bytes)

### Memory Considerations

| Source Size | Embeddings Memory | Safe? |
|-------------|------------------|-------|
| 1,000 docs | ~6 MB | Yes |
| 10,000 docs | ~60 MB | Yes |
| 50,000 docs | ~300 MB | Caution |
| 100,000 docs | ~600 MB | Use batching |

---

## Current Codebase State

### Existing Infrastructure

| Component | Location | Status |
|-----------|----------|--------|
| LLM Models tab | `admin/templates/sources.html` line 180 | Placeholder - "coming soon" |
| Embedding service | `offline_tools/embeddings.py` | Supports OpenAI + sentence-transformers |
| Ollama manager | `admin/ollama_manager.py` | Full integration, portable path support |
| Download system | `admin/routes/packs.py` | Resume-capable, progress tracking |
| Backup folder config | `local_settings.json` | GUI configurable |

### Current Embedding Configuration

Environment variables:
```
EMBEDDING_MODE=local           # or "openai"
EMBEDDING_MODEL=all-mpnet-base-v2  # 768-dim default
```

### Ollama Settings (local_settings.json)

```json
{
  "ollama": {
    "enabled": false,
    "url": "http://localhost:11434",
    "model": "mistral",
    "auto_start": true,
    "portable_path": ""
  }
}
```

---

## Current Embedding Code

**File:** `offline_tools/embeddings.py`

```python
"""
Embedding service for converting text to vectors.
Supports both OpenAI API and local models (sentence-transformers).
"""

from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class EmbeddingService:
    """
    Handles text-to-vector embedding.

    Supports:
    - OpenAI API (default): text-embedding-3-small, text-embedding-3-large
    - Local models (free): all-MiniLM-L6-v2, all-mpnet-base-v2, etc.

    Set EMBEDDING_MODE=local in .env to use local models.
    """

    def __init__(self, model: Optional[str] = None):
        """
        Args:
            model: Model to use. If None, auto-detects based on EMBEDDING_MODE env var.
                   OpenAI: "text-embedding-3-small", "text-embedding-3-large"
                   Local: "all-MiniLM-L6-v2", "all-mpnet-base-v2", etc.
        """
        self.mode = os.getenv("EMBEDDING_MODE", "openai").lower()

        # Check for model override from env
        env_model = os.getenv("EMBEDDING_MODEL", "")

        if self.mode == "local":
            # Use sentence-transformers for local embeddings
            default_local = "all-mpnet-base-v2"
            self._init_local(model or env_model or default_local)
        else:
            # Use OpenAI API
            self._init_openai(model or env_model or "text-embedding-3-small")

    def _init_openai(self, model: str):
        """Initialize OpenAI embedding client"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Either:\n"
                "  1. Set OPENAI_API_KEY in your .env file, or\n"
                "  2. Set EMBEDDING_MODE=local and install sentence-transformers:\n"
                "     pip install sentence-transformers"
            )
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._local_model = None
        print(f"Using OpenAI embeddings: {model}")

    def _init_local(self, model: str):
        """Initialize local sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading local embedding model: {model}...")
            self._local_model = SentenceTransformer(model)
            self.model = model
            self.client = None
            print(f"Local model loaded: {model} (dimension: {self._local_model.get_sentence_embedding_dimension()})")
        except ImportError:
            # Check if we can fall back to OpenAI
            if os.getenv("OPENAI_API_KEY"):
                print("sentence-transformers not installed. Run: pip install sentence-transformers")
                print("Falling back to OpenAI embeddings...")
                self._init_openai("text-embedding-3-small")
                self.mode = "openai"
            else:
                raise ValueError(
                    "EMBEDDING_MODE=local but sentence-transformers is not installed.\n"
                    "Install it with: pip install sentence-transformers\n"
                    "Or set OPENAI_API_KEY to use OpenAI embeddings instead."
                )

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if self.mode == "local" and self._local_model:
            text = text[:50000]
            embedding = self._local_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            return self._embed_with_chunking(text)

    def embed_batch(self, texts: List[str], batch_size: int = 50,
                    progress_callback=None) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently."""
        # ... implementation handles batching and progress
        pass

    def get_dimension(self) -> int:
        """Get the embedding dimension for the current model"""
        if self.mode == "local" and self._local_model:
            return self._local_model.get_sentence_embedding_dimension()
        elif "large" in self.model:
            return 3072
        else:
            return 1536
```

### Current Issues for Offline Use

1. Default mode is "openai" - not offline-friendly
2. No runtime fallback if API fails mid-operation
3. sentence-transformers auto-downloads to `~/.cache/huggingface/` (not portable)
4. No support for loading models from custom portable path
5. No graceful degradation to keyword search if embeddings unavailable

### What Needs to Change

1. Support loading models from `BACKUP_PATH/models/embeddings/`
2. Graceful fallback chain with warnings at each step
3. Auto-detect installed portable models
4. Let user cancel fallback if they prefer to fix their connection

---

## Current Embedding Chain (As-Is)

```
Initialization:
    |
    v
Check EMBEDDING_MODE env var (default: "openai")
    |
    +-- mode = "openai" -----------------> Require OPENAI_API_KEY
    |                                          |
    |                                          +-- Key exists: Use OpenAI API (1536-dim)
    |                                          +-- No key: RAISE ERROR (no fallback)
    |
    +-- mode = "local" ------------------> Try sentence-transformers
                                               |
                                               +-- Installed: Use local model (768-dim)
                                               +-- Not installed:
                                                       |
                                                       +-- OPENAI_API_KEY exists: Fall back to OpenAI
                                                       +-- No key: RAISE ERROR
```

---

## Target Embedding Chain (To-Be)

Respects user's mode preference while providing graceful degradation with warnings.

```
User's offline_mode setting: "online_only" | "hybrid" | "offline_only"
    |
    v
+-- "online_only" or "hybrid" (start with API)
|       |
|       v
|   Try OpenAI API (1536-dim)
|       |
|       +-- Success: Use API embeddings
|       |
|       +-- Failure (no key, network error, rate limit):
|               |
|               v
|           [WARNING] "API unavailable. Fall back to local model?"
|           [Continue] [Cancel & Retry Later]
|               |
|               +-- User cancels: Abort operation
|               +-- User continues: Fall through to local chain
|
+-- "offline_only" (skip API entirely)
        |
        v
    LOCAL FALLBACK CHAIN:
        |
        v
    1. Check BACKUP_PATH/models/embeddings/ for portable model
        |
        +-- Found: Load portable model (768-dim)
        |       |
        |       +-- Success: Use portable embeddings
        |       +-- Failure: Continue to next fallback
        |
        +-- Not found: Continue to next fallback
                |
                v
    2. Check HuggingFace cache (~/.cache/huggingface/)
        |
        +-- Found: Load cached model (768-dim)
        |       |
        |       +-- [WARNING] "Using cached model, not portable install"
        |       +-- Success: Use cached embeddings
        |
        +-- Not found: Continue to next fallback
                |
                v
    3. Try auto-download via sentence-transformers
        |
        +-- Online: Download and cache model
        |       |
        |       +-- [WARNING] "Downloading embedding model (~420MB)"
        |       +-- Success: Use downloaded embeddings
        |
        +-- Offline: Continue to final fallback
                |
                v
    4. FINAL FALLBACK: Keyword search
        |
        +-- [WARNING] "No embedding model available. Using keyword search only."
        +-- [WARNING] "Semantic search disabled. Results may be less relevant."
        +-- Execute keyword search (no vectors used)
```

---

## Fallback Warnings (User-Facing Messages)

| Fallback Level | Warning Message | User Options |
|----------------|-----------------|--------------|
| API -> Local | "Cloud API unavailable. Use local model instead?" | Continue / Cancel |
| Portable -> Cache | "Using cached model (not portable)" | Info only |
| Cache -> Download | "Downloading embedding model (~420MB)..." | Cancel |
| Download -> Keyword | "No embedding model. Using basic keyword search." | Info + suggestion |

**Key principle:** User should always know what's happening and have the option to stop if they'd rather fix the underlying issue.

### Warning Implementation

```python
class EmbeddingFallbackWarning:
    """Warnings shown during embedding fallback"""

    API_UNAVAILABLE = {
        "level": "warning",
        "title": "Cloud API Unavailable",
        "message": "Could not connect to embedding API. Results may differ from online mode.",
        "action": "Fall back to local embedding model?",
        "options": ["Continue with Local", "Cancel"]
    }

    USING_CACHE = {
        "level": "info",
        "title": "Using Cached Model",
        "message": "Loading embedding model from system cache. For portable offline use, download the model pack.",
        "action": None
    }

    DOWNLOADING = {
        "level": "info",
        "title": "Downloading Model",
        "message": "Downloading embedding model (all-mpnet-base-v2, ~420MB). This only happens once.",
        "action": None,
        "cancellable": True
    }

    KEYWORD_FALLBACK = {
        "level": "warning",
        "title": "Semantic Search Unavailable",
        "message": "No embedding model available. Using basic keyword search. Results may be less relevant.",
        "action": "Download embedding model for better search?",
        "options": ["Download Now", "Continue with Keyword Search"]
    }
```

---

## Mode Behavior Summary

| Mode | API Available | Local Available | Behavior |
|------|---------------|-----------------|----------|
| online_only | Yes | - | Use API |
| online_only | No | Yes | Warn, offer local fallback |
| online_only | No | No | Warn, keyword fallback |
| hybrid | Yes | - | Use API |
| hybrid | No | Yes | Auto-fallback to local (with notice) |
| hybrid | No | No | Keyword fallback (with warning) |
| offline_only | - | Yes | Use local directly |
| offline_only | - | No | Keyword fallback (with warning) |

**"hybrid" mode** = graceful auto-fallback (current behavior, enhanced)
**"online_only" mode** = ask before falling back (user may prefer to wait)
**"offline_only" mode** = never try API, use local chain directly

---

## Two-Pack Model System

Users download two separate model packs. Both work with 768-dim vectors.

### Dependency Chain

```
Semantic Search requires: Embedding Model (converts query to 768-dim vector)
Conversation Mode requires: Embedding Model + LLM (search first, then generate)

                         +------------------+
                         | Embedding Model  |  <-- FOUNDATION (download first)
                         | all-mpnet-base-v2|
                         | 420MB, 768-dim   |
                         +--------+---------+
                                  |
                    Enables semantic search
                                  |
                         +--------v---------+
                         |    LLM Model     |  <-- ENHANCEMENT (optional)
                         | Llama 3.2 3B Q4  |
                         | 2GB              |
                         +------------------+
                                  |
                    Enables AI conversation
```

---

### Pack 1: Semantic Search (Essential)

**Purpose:** Enable true semantic search offline

| Field | Value |
|-------|-------|
| Model | all-mpnet-base-v2 |
| Size | ~420MB |
| Dimensions | 768 |
| Runtime | sentence-transformers |
| Priority | HIGH - Download first |

**What it enables:**
- Query embedding for vector similarity search
- Works with 768-dim source packs
- True semantic search (not just keyword matching)

**Without it:**
- Falls back to keyword search only
- Stored 768-dim vectors are unused

---

### Pack 2: AI Conversation (Enhancement)

**Purpose:** Generate intelligent responses from search results

| Field | Value |
|-------|-------|
| Model | Llama 3.2 3B Q4 |
| Size | ~2GB |
| Format | GGUF |
| Runtime | llama.cpp or Ollama |
| Priority | MEDIUM - Download after Pack 1 |

**What it enables:**
- AI-synthesized answers from retrieved documents
- Conversational interface
- Question understanding and reformulation

**Without it:**
- Shows raw search results (still useful!)
- User reads articles directly

---

### User Experience by Configuration

| Installed | Search Type | Response Type | User Experience |
|-----------|-------------|---------------|-----------------|
| Neither | Keyword | Raw results | Basic - like site search |
| Embedding only | **Semantic** | Raw results | Good - finds relevant docs |
| LLM only | Keyword | AI response | Mixed - smart answers, dumb search |
| **Both** | **Semantic** | **AI response** | **Full offline experience** |

**Recommendation:** Always install Embedding Model first. It's smaller (420MB vs 2GB) and provides the foundation for good search.

---

### LLM Model Options (Pack 2 Variants)

| Tier | Model | Size | Hardware | Tokens/sec | Use Case |
|------|-------|------|----------|------------|----------|
| **Lite** | TinyLlama 1.1B Q4 | 700MB | RPi4 4GB | 15-30 | Very constrained |
| **Standard** | Llama 3.2 3B Q4 | 2GB | RPi5 8GB | 8-15 | Recommended |
| **Full** | Mistral 7B Q4 | 4GB | 16GB RAM | 5-10 | Desktop |

All LLM options work with the same 768-dim embedding model and source packs.

---

### Download Priority for Constrained Users

For users with limited storage/bandwidth:

```
Priority 1: Embedding Model (420MB)
    - Enables semantic search immediately
    - Small download, big impact

Priority 2: Source Packs (varies)
    - Pick most relevant sources
    - Each includes 768-dim vectors

Priority 3: LLM Model (2GB+)
    - Adds conversation capability
    - Can skip if storage limited
```

---

## Folder Structure

Models stored relative to backup folder:

```
BACKUP_PATH/
|-- _master.json              # Source index
|-- models/                   # NEW: Model storage
|   |-- _models.json          # Model registry/manifest
|   |-- embeddings/           # Embedding models
|   |   |-- all-MiniLM-L6-v2/
|   |   |   |-- _manifest.json
|   |   |   |-- model.onnx
|   |   |   |-- tokenizer.json
|   |   |   |-- config.json
|   |   |-- all-mpnet-base-v2/
|   |       |-- _manifest.json
|   |       |-- model.onnx
|   |       |-- ...
|   |-- llm/                  # LLM models (GGUF format for llama.cpp)
|       |-- tinyllama-1.1b-q4/
|       |   |-- _manifest.json
|       |   |-- tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
|       |-- llama-3.2-3b-q4/
|       |   |-- _manifest.json
|       |   |-- llama-3.2-3b-instruct.Q4_K_M.gguf
|       |-- mistral-7b-q4/
|           |-- _manifest.json
|           |-- mistral-7b-instruct-v0.2.Q4_K_M.gguf
|-- appropedia/               # Source packs (existing)
|-- chroma/                   # Vector database
```

### Portable sentence-transformers Structure

```
BACKUP_PATH/models/embeddings/all-mpnet-base-v2/
|-- config.json
|-- tokenizer.json
|-- tokenizer_config.json
|-- vocab.txt
|-- modules.json
|-- pytorch_model.bin       # ~420MB weights
|-- sentence_bert_config.json
|-- special_tokens_map.json
```

**Loading portable model:**
```python
from sentence_transformers import SentenceTransformer

# Current (downloads to ~/.cache/huggingface/)
model = SentenceTransformer("all-mpnet-base-v2")

# Portable (loads from backup folder)
model = SentenceTransformer("/path/to/BACKUP_PATH/models/embeddings/all-mpnet-base-v2")
```

---

## Model Manifest Schemas

### _models.json (Registry)

```json
{
  "schema_version": 1,
  "last_updated": "2025-12-11T00:00:00Z",
  "installed_embedding": "all-mpnet-base-v2",
  "installed_llm": "llama-3.2-3b-q4",
  "models": {
    "all-MiniLM-L6-v2": {
      "type": "embedding",
      "installed": false,
      "size_bytes": 83000000
    },
    "all-mpnet-base-v2": {
      "type": "embedding",
      "installed": true,
      "size_bytes": 420000000
    },
    "llama-3.2-3b-q4": {
      "type": "llm",
      "installed": true,
      "size_bytes": 2000000000
    }
  }
}
```

### Individual Model _manifest.json (Embedding)

```json
{
  "schema_version": 1,
  "model_id": "all-mpnet-base-v2",
  "model_type": "embedding",
  "display_name": "MPNet Base v2 (768-dim)",
  "description": "Good balance of quality and speed for offline semantic search",

  "dimensions": 768,
  "format": "onnx",
  "quantization": null,

  "size_bytes": 420000000,
  "files": [
    "model.onnx",
    "tokenizer.json",
    "config.json",
    "vocab.txt"
  ],

  "requirements": {
    "min_ram_gb": 4,
    "recommended_ram_gb": 8,
    "gpu_required": false
  },

  "compatibility": {
    "source_dimensions": [768],
    "runtime": "sentence-transformers"
  },

  "source_url": "https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
  "license": "Apache 2.0",
  "version": "1.0.0",
  "checksum_sha256": "abc123..."
}
```

### LLM Model _manifest.json

```json
{
  "schema_version": 1,
  "model_id": "llama-3.2-3b-q4",
  "model_type": "llm",
  "display_name": "Llama 3.2 3B (Q4 Quantized)",
  "description": "Best balance of quality and speed for RPi5 8GB",

  "parameters": "3B",
  "format": "gguf",
  "quantization": "Q4_K_M",

  "size_bytes": 2000000000,
  "files": [
    "llama-3.2-3b-instruct.Q4_K_M.gguf"
  ],

  "requirements": {
    "min_ram_gb": 4,
    "recommended_ram_gb": 8,
    "gpu_required": false
  },

  "performance": {
    "tokens_per_sec_rpi5": "8-15",
    "tokens_per_sec_laptop": "15-30",
    "context_length": 8192
  },

  "compatibility": {
    "runtime": ["llama.cpp", "ollama"],
    "platforms": ["windows", "linux", "macos", "arm64"]
  },

  "source_url": "https://huggingface.co/...",
  "license": "Llama 3.2 Community License",
  "version": "1.0.0",
  "checksum_sha256": "def456..."
}
```

---

## Cloud Storage Structure (R2)

```
r2://disaster-clippy-backups/
|-- backups/                  # Existing source packs
|   |-- appropedia/
|   |-- ...
|-- models/                   # NEW: Model distribution
    |-- _registry.json        # Available models list
    |-- embeddings/
    |   |-- all-MiniLM-L6-v2.zip
    |   |-- all-mpnet-base-v2.zip
    |-- llm/
        |-- tinyllama-1.1b-q4.zip
        |-- llama-3.2-3b-q4.zip
        |-- mistral-7b-q4.zip
```

---

## Download Flow

```
User opens "LLM Models" tab
    |
    v
Fetch model registry from R2: /models/_registry.json
    |
    v
Display model cards:
    - Installed status
    - Size
    - Hardware requirements
    - Compatible source dimensions
    |
    v
User clicks "Download" on model
    |
    v
Check disk space
    |
    v
Submit download job: POST /api/download-model
    |
    v
Download .zip from R2 with resume support
    |
    v
Extract to BACKUP_PATH/models/{type}/{model_id}/
    |
    v
Verify checksum
    |
    v
Update local _models.json registry
    |
    v
If embedding model: Update EMBEDDING_MODEL config
If LLM model: Update ollama config or llama.cpp path
    |
    v
Model ready to use
```

---

## Runtime Decisions

### LLM Runtime: llama.cpp

- Single binary + GGUF model file
- No separate service to manage (unlike Ollama)
- True portability - copy folder and run
- Cross-platform (Windows, Linux, macOS, ARM64)

### Embedding Runtime: sentence-transformers (native)

- Already works in current codebase
- Can load from custom path: `SentenceTransformer("/path/to/model")`
- No conversion needed
- Keep existing code, just add portable path detection

---

## Local Admin Submission Validation

**Key insight:** Local admins may have their own API keys and prefer online embedding quality.

### Submission Requirements

- Source pack MUST include vectors in at least ONE dimension (768 OR 1536)
- Having BOTH is optional but welcomed
- Global admin is responsible for ensuring both dimensions exist before publishing

### Validation Flow

```
Local Admin creates source pack
    |
    v
Check vectors present:
    |
    +-- Has 768-dim vectors? --> Valid for offline
    +-- Has 1536-dim vectors? --> Valid for online
    +-- Has both? --> Ideal, ready for publishing
    +-- Has neither? --> REJECT - vectors required
    |
    v
Submit to Global Admin
    |
    v
Global Admin reviews:
    |
    +-- Has both 768 + 1536? --> Ready to publish
    |
    +-- Has 768 only? --> Global admin generates 1536 via API
    |
    +-- Has 1536 only? --> Global admin generates 768 via local model
    |
    v
Publish to cloud:
    - Pinecone gets 1536-dim vectors (online search)
    - R2 backup gets 768-dim vectors (offline download)
```

### Why Allow Either Dimension

| Local Admin Setup | What They Submit | Why It's Valid |
|-------------------|------------------|----------------|
| Offline-only (no API key) | 768-dim only | Used local embedding model |
| Online-preferred (has API) | 1536-dim only | Used OpenAI for best quality |
| Hybrid setup | Both dimensions | Has both capabilities |

### Manifest Changes for Submission

```json
{
  "source_id": "my-local-source",
  "vectors": {
    "has_768": true,
    "has_1536": false,
    "embedding_model_768": "all-mpnet-base-v2",
    "embedding_model_1536": null
  },
  "submission_status": "pending_review",
  "needs_embedding": ["1536"]
}
```

### Global Admin Processing Code

```python
def process_submission(source_pack):
    has_768 = source_pack.vectors.has_768
    has_1536 = source_pack.vectors.has_1536

    if not has_768 and not has_1536:
        return reject("No vectors provided")

    # Generate missing dimension
    if not has_768:
        generate_768_vectors(source_pack)  # Local model

    if not has_1536:
        generate_1536_vectors(source_pack)  # OpenAI API

    # Now ready to publish
    upload_to_pinecone(source_pack, dimension=1536)
    upload_to_r2(source_pack, dimension=768)
```

### Benefits

1. Local admins can contribute regardless of their setup
2. No forced API costs for local admins
3. Global admin centralizes the "both dimensions" responsibility
4. Consistent quality - global admin uses same embedding models for all

---

## UI Design (LLM Models Tab)

```
+------------------------------------------------------------------+
| [All] [Embedding Models] [LLM Models] [Installed]                |
+------------------------------------------------------------------+
|                                                                  |
| EMBEDDING MODELS                                                 |
| ---------------------------------------------------------------- |
| +---------------------------+  +---------------------------+     |
| | MiniLM L6 v2 (384-dim)   |  | MPNet Base v2 (768-dim)   |     |
| | Size: 80 MB              |  | Size: 420 MB              |     |
| | RAM: 2GB min             |  | RAM: 4GB min              |     |
| | Quality: Basic           |  | Quality: Good             |     |
| |                          |  |                           |     |
| | [Download]               |  | [Installed] [Active]      |     |
| +---------------------------+  +---------------------------+     |
|                                                                  |
| LLM MODELS (for AI responses)                                    |
| ---------------------------------------------------------------- |
| +---------------------------+  +---------------------------+     |
| | TinyLlama 1.1B           |  | Llama 3.2 3B              |     |
| | Size: 700 MB             |  | Size: 2 GB                |     |
| | RAM: 2GB min             |  | RAM: 4GB min              |     |
| | Speed: 15-30 tok/s       |  | Speed: 8-15 tok/s         |     |
| | Quality: Basic           |  | Quality: Good             |     |
| |                          |  |                           |     |
| | [Download]               |  | [Installed] [Active]      |     |
| +---------------------------+  +---------------------------+     |
|                                                                  |
| +---------------------------+                                    |
| | Mistral 7B               |                                    |
| | Size: 4 GB               |                                    |
| | RAM: 8GB min             |                                    |
| | Speed: 5-10 tok/s        |                                    |
| | Quality: Excellent       |                                    |
| |                          |                                    |
| | [Download]               |                                    |
| +---------------------------+                                    |
|                                                                  |
| SYSTEM INFO                                                      |
| ---------------------------------------------------------------- |
| Detected RAM: 8 GB                                               |
| Recommended: Standard Bundle (768-dim + Llama 3.2 3B)            |
| Storage available: 45 GB                                         |
|                                                                  |
+------------------------------------------------------------------+
```

---

## Configuration Updates Needed

### local_settings.json additions

```json
{
  "models": {
    "embedding_model": "all-mpnet-base-v2",
    "embedding_dimensions": 768,
    "llm_model": "llama-3.2-3b-q4",
    "llm_runtime": "llama.cpp",
    "models_folder": ""
  }
}
```

If `models_folder` is empty, defaults to `BACKUP_PATH/models/`.

---

## Integration Points

### 1. Embedding Service (`offline_tools/embeddings.py`)

- Add model path detection from `BACKUP_PATH/models/embeddings/`
- Load ONNX models from portable location instead of HuggingFace cache
- Fall back to auto-download if not found locally

### 2. LLM Service

- Option A: Point Ollama to portable models folder
- Option B: Use llama.cpp directly with GGUF files
- Option C: Support both (Ollama for ease, llama.cpp for portability)

### 3. Source Downloads

- Warn if downloading 768-dim sources but only 384-dim embedding installed
- Suggest appropriate embedding model for source dimensions

### 4. Search Service

- Auto-detect installed embedding model dimensions
- Route to correct ChromaDB collection based on dimensions

---

## Implementation Phases

### Phase 1: Model Registry and Storage

- [ ] Create `BACKUP_PATH/models/` folder structure
- [ ] Define _models.json and _manifest.json schemas
- [ ] Create model registry class in `offline_tools/model_registry.py`
- [ ] Upload initial model packs to R2

### Phase 2: Download Infrastructure

- [ ] Add `/api/list-models` endpoint
- [ ] Add `/api/download-model` endpoint with resume support
- [ ] Add `/api/model-status` endpoint
- [ ] Implement checksum verification

### Phase 3: UI Implementation

- [ ] Replace placeholder in sources.html LLM Models tab
- [ ] Model cards with download progress
- [ ] Hardware detection and recommendations
- [ ] Active model selection

### Phase 4: Runtime Integration

- [ ] Update embeddings.py to load from portable path
- [ ] Add llama.cpp runtime option alongside Ollama
- [ ] Auto-configure based on installed models
- [ ] Dimension compatibility warnings

### Phase 5: Source-Model Coordination

- [ ] Track source pack dimensions in manifest
- [ ] Warn on dimension mismatch during download
- [ ] "Download matching embedding model" prompt
- [ ] Bundle suggestions (source + model together)

---

## Model Hosting Options

| Option | Pros | Cons |
|--------|------|------|
| **R2 direct** | Simple, consistent with sources | Egress costs for large models |
| **HuggingFace links** | Free hosting, always updated | External dependency |
| **GitHub releases** | Free for open source | Size limits |
| **Torrents** | Distributed, resilient | Complex setup |

**Recommendation:** Start with R2 for consistency. Models are downloaded once, so egress is manageable. Can add HuggingFace fallback later.

---

## Size Estimates for R2 Storage

| Model | Compressed | Uncompressed |
|-------|------------|--------------|
| all-MiniLM-L6-v2 | ~50MB | ~80MB |
| all-mpnet-base-v2 | ~250MB | ~420MB |
| TinyLlama 1.1B Q4 | ~650MB | ~700MB |
| Llama 3.2 3B Q4 | ~1.8GB | ~2GB |
| Mistral 7B Q4 | ~3.8GB | ~4GB |
| **Total** | **~6.5GB** | **~7.2GB** |

---

## Dual Embedding Architecture

### Problem Discovery

Current offline search does NOT use semantic search. Investigation revealed:

1. User query comes in
2. `AIService.search()` checks connection mode
3. If offline: calls `store.search_offline()` (keyword search)
4. If online: calls `store.search()` (semantic search with OpenAI embedding)

The `search_offline()` method does simple text matching - no embeddings at all.
The stored 1536-dim vectors are **completely unused** in offline mode.

**Root cause:** To do semantic search, the query must be embedded in the SAME dimension
as the stored vectors. Offline mode has no API access to generate 1536-dim query embeddings.

### Solution: Dual Embedding Standard

Standardize on two embedding dimensions:

| Context | Dimension | Model | Purpose |
|---------|-----------|-------|---------|
| Online (Pinecone) | 1536 | OpenAI text-embedding-3-small | Cloud search |
| Offline (local ChromaDB) | 768 | sentence-transformers / Ollama | Local search |

Global system stores BOTH. Users download only the 768-dim version for offline use.

### Three User Perspectives

```
PUBLIC USER                    LOCAL ADMIN                   GLOBAL ADMIN
===========                    ===========                   ============

[Chat interface]               [768 world only]              [Both worlds]
      |                              |                             |
      v                              v                             v
[Pinecone 1536]               [Download 768]                [Generate both]
      |                              |                             |
Never sees                    [Local ChromaDB 768]          [Validate both exist]
embeddings                           |                             |
                              [Create local 768]            [Sync 1536 -> Pinecone]
                                     |                             |
                              [Submit to global]            [Sync both -> R2]
                                     |
                              (Global re-embeds             [768 ready for download]
                               to 1536 if needed)
```

**Public user:** Searches Pinecone (1536-dim). Never sees embeddings.

**Local admin:** Lives entirely in "768 world":
- Downloads source packs with 768-dim vectors
- Creates local content with 768-dim (local models)
- Never needs to think about 1536
- True semantic search works offline

**Global admin:** Manages both worlds:
- Creates BOTH 768 and 1536 embeddings during indexing
- Validates both vector files exist
- Syncs 1536 to Pinecone (for online search)
- Syncs both to R2 (768 for user downloads, 1536 as backup)

### File Structure (Source Packs)

```
{source_id}/
    _manifest.json
    _metadata.json
    _index.json
    _vectors_768.json    # Offline embeddings (downloaded by users)
    _vectors_1536.json   # Online embeddings (Pinecone + backup)
```

### Search Flow

**Online (unchanged):**
```
Query -> OpenAI 1536-dim -> Pinecone -> Results
```

**Offline (new - true semantic search):**
```
Query -> Local 768-dim model -> Local ChromaDB 768 -> Results
```

### R2 Storage Structure

```
r2://disaster-clippy/
    sources/
        appropedia/
            _manifest.json
            _metadata.json
            _index.json
            _vectors_768.json     <-- Users download this
            _vectors_1536.json    <-- Backup only (not downloaded)
            pages.zip             <-- Optional HTML backup
```

### Download Flow

```
User clicks "Download Pack"
           |
           v
    [API fetches from R2:]
      - _manifest.json
      - _metadata.json
      - _index.json
      - _vectors_768.json    <-- Only 768, NOT 1536
           |
           v
    [Import to local ChromaDB]
           |
           v
    [768-dim collection ready]
           |
           v
    [Offline semantic search works!]
```

### Benefits

1. **True offline semantic search** - Not just keyword matching
2. **Smaller downloads** - 768-dim is half the size of 1536-dim
3. **No API needed offline** - Local model generates query embeddings
4. **No dimension conflicts** - Each context has one standard
5. **Clean separation** - Local admins never see 1536-dim complexity

### Storage Size Comparison

| Docs | 768-dim | 1536-dim | Savings |
|------|---------|----------|---------|
| 1,000 | ~3 MB | ~6 MB | 50% |
| 10,000 | ~30 MB | ~60 MB | 50% |
| 91,000 | ~275 MB | ~550 MB | 50% |

---

## User Tier System

The system supports four distinct user tiers, each with different capabilities, hardware requirements, and responsibilities.

### Tier Overview

| Tier | Hardware | Role | Primary Actions |
|------|----------|------|-----------------|
| **Consumer** | RPi5 / Field device | End user | Download, search, browse |
| **Local Admin** | Laptop 8-16GB | Content creator | Experiment, create sources, submit |
| **Global Admin** | Desktop + API | Curator | Review, standardize, publish, maintain |
| **Super Powered** | Cloud/GPU farm | Heavy processing | Mass indexing, re-embedding, migrations |

### Tier 1: Consumer Offline (RPi5 / Field Device)

**Use case:** Download packs, ask questions, browse content offline

**Key insight:** Consumers never embed documents - only embed queries (fast, single text).

**Hardware requirements:**
- RPi5 8GB RAM (recommended) or 4GB (minimum)
- 32-64GB storage
- No GPU needed

**What they do:**
- Download pre-embedded source packs (768-dim)
- Search using local 768-dim model for query embedding
- Optionally use small LLM for response generation
- Browse ZIM/HTML backups offline

**What they DON'T do:**
- Create new sources
- Generate embeddings for documents
- Submit content

**Optimizations for this tier:**
- Pre-load embedding model on startup (eliminate cold start)
- Cache recent query embeddings
- Memory-mapped ChromaDB to reduce RAM
- Optional: pre-computed common question cache

### Tier 2: Local Admin (Laptop/Desktop)

**Use case:** Experiment with source tools, process small-medium sources, mix online/offline

**Hardware requirements:**
- 8-16GB RAM
- 100GB+ storage
- Optional: GPU for faster local embedding

**What they do:**
- Download source packs from cloud
- Create new sources from HTML/ZIM/PDF
- Generate embeddings (any dimension based on their hardware)
- Submit sources to global admin for review
- Mix of online and offline operation

**Capabilities:**
- ChromaDB (local vector store)
- R2 backups read access
- R2 submissions write access
- Local embedding (384 or 768 dim)
- Optional API access for 1536-dim

**Limits:**
- No Pinecone write access
- No R2 backups write access
- Recommended max source size: 10-20k docs (local embedding)

### Tier 3: Global Admin (Current Production)

**Use case:** Review submissions, standardize dimensions, publish to production, maintain global index

**Hardware:** Personal desktop/laptop + OpenAI API access

**Responsibilities:**
- Review submissions from local admins
- Re-embed to standardized dimensions (768 + 1536)
- Publish to Pinecone and R2
- Maintain quality and consistency
- Process new large sources

**Capabilities:**
- Full ChromaDB access
- Full Pinecone read/write
- Full R2 read/write (both buckets)
- OpenAI API for 1536-dim embeddings
- Local model for 768-dim embeddings

### Tier Capabilities Matrix

| Capability | Consumer | Local Admin | Global Admin | Super Powered |
|------------|----------|-------------|--------------|---------------|
| Download packs | Yes | Yes | N/A | N/A |
| Search (semantic) | Yes (768) | Yes (768) | Yes (both) | Yes (both) |
| Create sources | No | Yes | Yes | Yes |
| Embed documents | No | Yes (any dim) | Yes (768+1536) | Yes (parallel) |
| Submit to global | No | Yes | N/A | N/A |
| Pinecone write | No | No | Yes | Yes |
| R2 backups write | No | No | Yes | Yes |
| Re-embed submissions | No | No | Yes | Yes |
| Parallel processing | No | No | No | Yes |

### Who Creates What

| Role | Creates 768? | Creates 1536? | Downloads 768? | Uses Pinecone? |
|------|--------------|---------------|----------------|----------------|
| Consumer | No | No | Yes | No |
| Local Admin | Optional (any dim) | Optional (if API) | Yes | No |
| Global Admin | Yes | Yes | N/A | Yes (write) |
| Super Powered | Yes (batch) | Yes (batch) | N/A | Yes (write) |

---

## RPi5 Consumer Tier Analysis

### The Question

Can an RPi5 run semantic search with 768-dim indexes AND provide an intelligent LLM feel?

### Answer: Yes

**Two separate components needed:**

| Component | Purpose | Model | RAM | RPi5 Feasible? |
|-----------|---------|-------|-----|----------------|
| Embedding model | Query to 768-dim vector | all-mpnet-base-v2 | ~500MB | Yes |
| LLM | Generate intelligent response | Llama 3.2 3B Q4 | ~2GB | Yes, but slow |

### RPi5 8GB Memory Budget

```
OS + ChromaDB:        ~1.5GB
Embedding model:      ~0.5GB (mpnet 768-dim)
LLM (3B quantized):   ~2.0GB
Buffer:               ~1.0GB
---------------------------------
Total needed:         ~5GB of 8GB available
```

### LLM Options for RPi5

| Model | Parameters | RAM (Q4) | Tokens/sec | Quality |
|-------|------------|----------|------------|---------|
| TinyLlama | 1.1B | 0.7GB | 15-30 | Basic |
| Llama 3.2 | 1B | 1.0GB | 20-40 | Decent |
| Qwen 2.5 | 1.5B | 1.0GB | 15-25 | Good |
| **Llama 3.2** | **3B** | **2.0GB** | **8-15** | **Recommended** |
| Phi-3 Mini | 3.8B | 2.5GB | 5-12 | Good |

### Realistic User Experience

```
User types: "how do I purify water in an emergency?"
    |
    v
[Embed query to 768-dim] - 200-500ms
    |
    v
[Search ChromaDB] - 50-100ms
    |
    v
[Retrieve top 5 docs] - 50ms
    |
    v
[Generate response ~150 tokens @ 10 tok/sec] - 15 seconds
    |
    v
Total: ~15-20 seconds (subsequent queries)
First query: +15-30 sec for model loading
```

### Search Mode vs Conversation Mode

User choice to optimize experience:

| Mode | What Happens | Response Time | Use Case |
|------|--------------|---------------|----------|
| **Search Mode** | Semantic search, show docs only | 1-2 sec | Quick lookup, browsing |
| **Conversation Mode** | Search + LLM response | 15-20 sec | Detailed answers, synthesis |

**UI implementation:**
- Toggle in chat interface
- Default to Search Mode on RPi5
- Auto-detect hardware and suggest appropriate mode

### Consumer Tier Optimizations

| Optimization | Impact | Implementation |
|--------------|--------|----------------|
| Pre-load models on startup | Eliminates cold start | Load embedding + LLM at app start |
| Query embedding cache | Repeat questions instant | LRU cache of recent queries |
| Memory-mapped ChromaDB | Reduces RAM ~50% | Configure on init |
| Pre-bundled models | No download wait | Ship with installer |
| Streaming responses | Feels faster | Show tokens as generated |

### What Consumer Pack Contains

```
source_pack/
    _manifest.json         # Metadata including embedding info
    _vectors_768.json      # Pre-embedded (user never re-embeds)
    _index.json            # Full content for display
    content.zim            # Optional: browsable backup
```

---

## Processing Time Reality

### By Tier and Source Size

| Source Size | Consumer | Local Admin (768) | Global Admin (API) | Super Powered |
|-------------|----------|-------------------|-------------------|---------------|
| 1,000 docs | N/A | ~10 min | ~2 min | ~30 sec |
| 10,000 docs | N/A | ~2 hours | ~15 min | ~3 min |
| 50,000 docs | N/A | ~10 hours | ~1 hour | ~10 min |
| 100,000 docs | N/A | ~20 hours | ~2 hours | ~20 min |
| 450,000 docs | N/A | **4+ days** | ~10 hours | ~2 hours |

### Practical Limits by Tier

| Tier | Recommended Max | Beyond This |
|------|-----------------|-------------|
| Local Admin (local embedding) | 20,000 docs | Use API or split source |
| Global Admin (API embedding) | 100,000 docs | Use Super Powered |
| Super Powered | 1,000,000+ docs | Distributed processing |

### Cost Estimates (OpenAI text-embedding-3-small)

| Source Size | Tokens (est) | Cost | Super Powered Time |
|-------------|--------------|------|-------------------|
| 10,000 docs | 10M | ~$1 | 3 min |
| 100,000 docs | 100M | ~$10 | 20 min |
| 450,000 docs | 450M | ~$45 | 2 hours |
| 1,000,000 docs | 1B | ~$100 | 4 hours |

### Hardware Detection for Local Admin

```python
def detect_local_admin_tier():
    ram_gb = get_system_ram()

    if ram_gb < 8:
        return {
            "tier": "constrained",
            "max_source_size": 2000,
            "recommended_embedding": "384",
            "recommended_model": "all-MiniLM-L6-v2",
            "warning": "Limited RAM - consider smaller sources"
        }
    elif ram_gb < 16:
        return {
            "tier": "standard",
            "max_source_size": 10000,
            "recommended_embedding": "768",
            "recommended_model": "all-mpnet-base-v2"
        }
    else:
        return {
            "tier": "capable",
            "max_source_size": 50000,
            "recommended_embedding": "768",
            "note": "For larger sources, consider API embedding"
        }
```

---

## Key Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Offline dimension | 768 | Works on RPi5, good quality, reasonable size |
| Online dimension | 1536 | OpenAI standard, best quality |
| Who creates both | Global Admin | Centralized quality control |
| What users download | 768 only | Smaller, works offline |
| Local admin submissions | Any dimension | Flexibility, re-embedded on approval |
| Backward compat default | 768 | Offline-first philosophy |
| Consumer LLM | Llama 3.2 3B Q4 | Best quality/speed for RPi5 |
| Embedding model (local) | all-mpnet-base-v2 | 768-dim, good quality |
| LLM Runtime | llama.cpp | Portable, no service to manage |
| Embedding Runtime | sentence-transformers | Already works, load from custom path |

---

*Last updated: December 2025*
