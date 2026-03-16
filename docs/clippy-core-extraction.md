# clippy_core Extraction

Reference document for the extractable chat/search module and the layered architecture it supports.

---

## Current Status

**`clippy_core/` is not currently present in the active repo.**

It previously existed as a standalone extraction layer in commit `6e67f72` (`clippy core refactor`, February 1, 2026), but was later removed during cleanup.

The architecture is still useful and should be treated as a live design direction:

- Disaster Clippy currently uses its existing code paths:
  - `app.py`
  - `admin/ai_service.py`
  - `offline_tools/vectordb/`
- `clippy_core/` should be understood as a future extraction target
- The goal is still to keep Disaster Clippy stable while making the chat/search engine portable

This document restores the design intent even though the original implementation is no longer in the tree.

---

## Overview

`clippy_core/` is intended to be a self-contained folder that can be copied into other projects with minimal or zero modification.

The broader goal is:

- Disaster Clippy remains the full product
- `clippy_core/` becomes the portable engine
- other projects can reuse the same chat/search stack without importing the full Disaster Clippy admin and source-prep system

This is especially relevant again now that Disaster Clippy is moving back toward a public/private repo split and a cleaner layered structure.

---

## Four Use Cases

| # | Use Case | Description | Layers Needed |
|---|----------|-------------|---------------|
| 1 | Global Production | Hosted public app, shared cloud vectors, global admin and catalog/update control plane | All layers |
| 2 | Offline/Local | RPi, air-gapped, translations, local LLMs, local source creation | All layers |
| 3 | Custom Online | Self-hosted full instance with custom cloud services and custom packs | All layers |
| 4 | Export to Other Projects | Just chat + search, pluggable storage/LLM/runtime shell | `clippy_core/` only |

---

## Layered Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                         app.py                                  │
│              (FastAPI, chat routes, hosted shell)               │
├─────────────────────────────────────────────────────────────────┤
│                        admin/                                   │
│      (Web UI, jobs, dashboard, settings, local admin tools)     │
├─────────────────────────────────────────────────────────────────┤
│                    offline_tools/                               │
│  (Indexers, scrapers, validation, source_manager, translation)  │
├─────────────────────────────────────────────────────────────────┤
│                      clippy_core/                               │
│     (Chat, search, LLM, embeddings, vectordb adapters)          │
└─────────────────────────────────────────────────────────────────┘
```

Dependency direction:

- `app.py` can import from `admin/`, `offline_tools/`, and `clippy_core/`
- `admin/` can import from `offline_tools/` and `clippy_core/`
- `offline_tools/` can import from `clippy_core/`
- `clippy_core/` should have zero dependencies on the layers above it

That dependency rule is the main reason this design still matters.

---

## Why This Matters Again

The project is heading toward a clearer workspace split:

- public repo: core runtime, local/offline app, local admin/source creation
- private repo: `.com` product site, account/profile layer, update/control plane, hosted ops

That split works much better if the actual chat/search engine is cleanly separable from:

- local admin UI
- source ingestion and creation tooling
- deployment-specific wrappers
- product-site and account-layer concerns

`clippy_core/` is the missing architectural seam for that.

---

## Proposed `clippy_core/` Structure

```text
clippy_core/
├── __init__.py
├── chat.py
├── search.py
├── llm.py
├── embeddings.py
├── config.py
├── schemas.py
└── vectordb/
    ├── __init__.py
    ├── base.py
    ├── chromadb.py
    ├── pinecone.py
    └── pgvector.py
```

### Intended responsibilities

| File | Responsibility |
|------|----------------|
| `chat.py` | Main `ChatService` orchestration |
| `search.py` | Search ranking, filtering, diversity, result shaping |
| `llm.py` | LLM abstraction across providers |
| `embeddings.py` | Query embedding generation |
| `config.py` | Portable configuration dataclasses |
| `schemas.py` | Shared message/result/request types |
| `vectordb/base.py` | Abstract vector store interface |
| `vectordb/chromadb.py` | Local/offline vector store adapter |
| `vectordb/pinecone.py` | Hosted cloud vector store adapter |
| `vectordb/pgvector.py` | Export-friendly SQL/vector adapter |

---

## Key Interface: ChatService

```python
class ChatService:
    def __init__(
        self,
        vector_store,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimension: int = 1536,
    ):
        ...

    async def chat(
        self,
        message: str,
        sources: list[str] | None = None,
        system_prompt: str | None = None,
        conversation_history: list | None = None,
        stream: bool = False,
    ):
        ...

    async def search(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int = 10,
    ):
        ...
```

The important point is not the exact signature. It is that the interface should be:

- storage-agnostic
- provider-agnostic
- reusable outside Disaster Clippy

---

## Key Interface: VectorStore

```python
from abc import ABC, abstractmethod

class VectorStore(ABC):
    @abstractmethod
    async def search(
        self,
        query: str,
        n_results: int = 10,
        source_filter: list[str] | None = None,
    ):
        ...

    @abstractmethod
    async def get_sources(self):
        ...
```

Optional capabilities can extend this:

- keyword fallback search
- metadata-only listing
- result diversity controls
- per-language or per-pack filtering

---

## Relationship to Current Repo

Today, the closest source files are:

| Current file | Likely future `clippy_core` home |
|--------------|----------------------------------|
| `admin/ai_service.py` | `clippy_core/chat.py`, `clippy_core/search.py`, parts of `clippy_core/llm.py` |
| `admin/connection_manager.py` | possibly a thin optional wrapper, or stay above core |
| `offline_tools/vectordb/store.py` | `clippy_core/vectordb/chromadb.py` |
| `offline_tools/vectordb/pinecone_store.py` | `clippy_core/vectordb/pinecone.py` |
| `offline_tools/vectordb/factory.py` | `clippy_core/vectordb/__init__.py` or factory helpers |
| `offline_tools/embeddings.py` | `clippy_core/embeddings.py` |
| API-layer request/response shaping in `app.py` | stays outside core |

Not everything should move. In particular, these should remain above core:

- admin pages and jobs
- source creation and validation
- scraping and indexing
- translation packs and video processing pipelines
- deployment-specific settings UI

---

## Relationship to Public and Private Repos

If Disaster Clippy returns to a sibling workspace layout like:

```text
disaster-clippy/
├── public/
└── private/
```

then the cleanest long-term split is:

- `public/`
  - full runtime
  - local/offline install
  - local admin/source creation tools
  - shared documentation
- `private/`
  - `.com` site
  - account/profile layer
  - update catalogs/manifests
  - hosted operational tooling
- `clippy_core/`
  - portable engine that either side can reuse

That does not mean `clippy_core/` has to live in the private repo. It means the architectural seam should exist so either side can depend on it cleanly.

---

## Migration Principle

The original refactor got this part right: `clippy_core/` should begin as a parallel implementation, not a forced big-bang rewrite.

Recommended approach:

1. Recreate `clippy_core/` as a new isolated package
2. Extract interfaces first, not every implementation detail
3. Wrap existing vector stores before rewriting them
4. Keep Disaster Clippy using existing code until parity is proven
5. Migrate internal call sites gradually only if the extraction proves worthwhile

That keeps the product stable while rebuilding the seam.

---

## Suggested Initial Scope

If the extraction is revived, the first phase should probably include only:

- shared schemas
- `ChatService`
- vector store abstraction
- wrappers for current ChromaDB and Pinecone stores
- embedding abstraction
- LLM abstraction

It should explicitly exclude at first:

- admin UI
- jobs
- source creation pipeline
- translation pipeline
- video pipeline
- PDF OCR pipeline

Those belong above core.

---

## Open Questions

- Should `clippy_core/` live inside the public repo, or later become its own package/repo?
- Should connection-state logic live inside core, or remain a higher-level runtime concern?
- Should source filtering, pack filtering, and entitlement filtering all share one interface?
- Should streaming be standardized at the core layer, or left to the API shell?
- Should `pgvector` be a first-class maintained adapter, or just an example integration path?

---

## Recommendation

The old code should not simply be restored wholesale.

What should be restored is:

- the four-use-case framing
- the four-layer architecture
- the rule that dependencies only flow downward
- the idea of a portable chat/search engine that can be reused outside the full app

That architecture still fits the direction the project is moving in now.

---

*Originally implemented in commit `6e67f72` on February 1, 2026.*  
*Recreated as a current design note on March 15, 2026.*  
*Status: architecture target, not active implementation.*
