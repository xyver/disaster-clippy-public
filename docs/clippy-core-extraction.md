# clippy_core Extraction

Reference document for the extractable chat/search module.

---

## Current Status

**clippy_core/ is CREATED and STANDALONE.**

- Disaster Clippy continues using its existing code (`admin/ai_service.py`, `offline_tools/vectordb/`)
- clippy_core/ is a parallel implementation for export to other projects
- No changes to existing disaster-clippy functionality

This is intentional - keeps disaster-clippy stable while providing a clean extraction point.

---

## Overview

`clippy_core/` is a self-contained folder that can be copied to other projects (like Sheltrium) with zero modifications.

**Goal**: When disaster-clippy improves its chat logic, update clippy_core and other projects get the upgrade by copying the folder.

---

## Four Use Cases

| # | Use Case | Description | Layers Needed |
|---|----------|-------------|---------------|
| 1 | Global Production | Main Railway instance, Pinecone, full admin | All layers |
| 2 | Offline/Local | RPi, air-gapped, translations, local LLMs, source creation | All layers |
| 3 | Custom Online | Self-hosted full instance with own cloud services | All layers |
| 4 | Export to Other Projects | Just chat + search, pluggable storage/LLM (Sheltrium) | clippy_core only |

---

## Layered Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         app.py                                  │
│              (FastAPI, chat routes, admin mount)                │
├─────────────────────────────────────────────────────────────────┤
│                        admin/                                   │
│         (Web UI, job manager, dashboard, settings)              │
├─────────────────────────────────────────────────────────────────┤
│                    offline_tools/                               │
│    (Indexers, scrapers, validation, source_manager, translation)│
├─────────────────────────────────────────────────────────────────┤
│                      clippy_core/                               │
│        (Chat, search, LLM, embeddings, vectordb adapters)       │
└─────────────────────────────────────────────────────────────────┘
```

**Dependency direction**: Each layer only imports from layers below it.
- `app.py` -> `admin/` -> `offline_tools/` -> `clippy_core/`
- `clippy_core/` has ZERO dependencies on layers above

---

## clippy_core/ Structure

```
clippy_core/
├── __init__.py              # Public API exports
├── chat.py                  # ChatService class - main entry point
├── search.py                # Search logic (diversity, filtering, result ranking)
├── llm.py                   # LLM abstraction (OpenAI, Anthropic, local)
├── embeddings.py            # Embedding generation (OpenAI, local models)
├── connection.py            # Online/offline detection (optional)
├── config.py                # ClippyConfig dataclass
├── schemas.py               # SearchResult, Message, ResponseResult types
└── vectordb/
    ├── __init__.py          # Exports get_vector_store, VectorStore
    ├── base.py              # Abstract VectorStore interface
    ├── chromadb.py          # ChromaDB implementation (local/offline)
    ├── pinecone.py          # Pinecone implementation (cloud)
    └── pgvector.py          # Supabase pgvector implementation (NEW)
```

---

## Key Interface: ChatService

```python
# clippy_core/chat.py

class ChatService:
    def __init__(
        self,
        vector_store: VectorStore,        # Injected - you choose the backend
        llm_provider: str = "openai",     # "openai", "anthropic", "local"
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        embedding_dimension: int = 1536,
    ):
        """
        Initialize chat service with pluggable vector store.

        Args:
            vector_store: Any VectorStore implementation (ChromaDB, Pinecone, pgvector)
            llm_provider: Which LLM to use for response generation
            llm_model: Specific model name
            embedding_model: Model for query embedding (must match indexed vectors)
            embedding_dimension: Dimension of embeddings (768, 1536, etc.)
        """

    async def chat(
        self,
        message: str,
        sources: list[str] = None,        # Filter to these source IDs
        system_prompt: str = None,        # Custom prompt (for context injection)
        conversation_history: list = None,
        stream: bool = False,
    ) -> str | AsyncGenerator[str, None]:
        """
        Main chat entry point.

        Args:
            message: User's question
            sources: List of source IDs to search (None = all sources)
            system_prompt: Override default system prompt (for custom context)
            conversation_history: Previous messages for context
            stream: If True, returns async generator of chunks

        Returns:
            Response text, or async generator if streaming
        """

    async def search(
        self,
        query: str,
        sources: list[str] = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Search without LLM response (for custom pipelines).
        """
```

---

## Key Interface: VectorStore (Abstract)

```python
# clippy_core/vectordb/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class VectorStore(ABC):
    """Abstract interface for vector storage backends."""

    @abstractmethod
    async def search(
        self,
        query: str,
        n_results: int = 10,
        source_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search.

        Args:
            query: Search query (will be embedded internally)
            n_results: Number of results to return
            source_filter: List of source IDs to include (None = all)

        Returns:
            List of documents with id, content, source, url, score, metadata
        """

    @abstractmethod
    async def get_sources(self) -> List[Dict[str, Any]]:
        """
        List available sources.

        Returns:
            List of {id, name, count} for each source
        """

    # Optional methods for stores that support them
    def search_keyword(self, query: str, **kwargs) -> List[Dict]:
        """Keyword fallback search (for offline mode)."""
        raise NotImplementedError("This store doesn't support keyword search")
```

---

## Current File Structure

clippy_core is a **parallel implementation**, not a refactor of existing code.

### What Was Created (NEW files)

| File | Purpose | Lines |
|------|---------|-------|
| `clippy_core/__init__.py` | Package exports | ~60 |
| `clippy_core/config.py` | ClippyConfig dataclass | ~160 |
| `clippy_core/schemas.py` | SearchResult, ChatMessage types | ~150 |
| `clippy_core/chat.py` | ChatService main interface | ~280 |
| `clippy_core/llm.py` | LLM abstraction (OpenAI/Anthropic) | ~300 |
| `clippy_core/vectordb/__init__.py` | Factory and exports | ~80 |
| `clippy_core/vectordb/base.py` | Abstract VectorStore interface | ~100 |
| `clippy_core/vectordb/factory.py` | Helper utilities | ~50 |
| `clippy_core/vectordb/pgvector.py` | Supabase pgvector (NEW) | ~350 |

**Total: ~1,530 lines of new code**

### What Stays Unchanged (disaster-clippy uses these)

| File | Purpose |
|------|---------|
| `admin/ai_service.py` | Existing chat/search orchestration |
| `admin/connection_manager.py` | Online/offline detection |
| `admin/local_config.py` | Settings from local_settings.json |
| `offline_tools/vectordb/*` | Existing ChromaDB/Pinecone stores |
| `offline_tools/embeddings.py` | Existing embedding service |
| All other files | Unchanged |

### How clippy_core Uses Existing Stores

clippy_core's `get_vector_store()` can wrap existing stores:
```python
# clippy_core/vectordb/__init__.py
try:
    from offline_tools.vectordb.store import VectorStore as ChromaDBStore
    # Use existing ChromaDB implementation
except ImportError:
    # Fall back to clippy_core's own implementation
    from .chromadb import ChromaDBStore
```

This means clippy_core works standalone OR can leverage existing disaster-clippy stores.

---

## Sheltrium Integration Example

```python
# sheltrium/backend/chat_api.py

from clippy_core import ChatService
from clippy_core.vectordb.pgvector import PgVectorStore

# Initialize with Sheltrium's own pgvector
store = PgVectorStore(
    connection_string=os.getenv("SUPABASE_DB_URL"),
    table_name="source_vectors",
    embedding_dimension=768
)

chat_service = ChatService(
    vector_store=store,
    llm_provider="openai",
    llm_model="gpt-4o-mini"
)

async def handle_user_chat(user_id: str, property_id: str, message: str):
    # Load Sheltrium-specific context
    profile = await get_user_profile(user_id)
    property = await get_property(property_id)
    risks = await get_risk_scores(property.loc_id)

    # Build custom system prompt with user context
    system_prompt = f"""You are a preparedness advisor for {property.name}.

USER CONTEXT:
- Location: {property.state}, ZIP {property.zip}
- Hurricane risk: {risks.hurricane_rating}
- Flood risk: {risks.flood_rating}
- Current equipment: {property.solar_setup or 'None installed'}
- Survey completed: {'Yes' if profile.survey_completed else 'No'}

INSTRUCTIONS:
- Provide specific advice for this user's location and risk profile
- Reference local building codes when relevant
- Consider their current equipment when recommending upgrades
- Cite sources when possible"""

    # Select sources based on location
    sources = get_sources_for_property(property, risks)
    # e.g., ["fema-ready", "fl-building-code", "fortified-2025", "fl-hurricane-guide"]

    # Use clippy_core for chat
    async for chunk in await chat_service.chat(
        message=message,
        sources=sources,
        system_prompt=system_prompt,
        stream=True
    ):
        yield chunk
```

---

## pgvector Implementation (NEW)

```python
# clippy_core/vectordb/pgvector.py

from .base import VectorStore
from typing import List, Dict, Any, Optional
import asyncpg

class PgVectorStore(VectorStore):
    """Supabase pgvector implementation."""

    def __init__(
        self,
        connection_string: str,
        table_name: str = "source_vectors",
        embedding_dimension: int = 768,
        embedding_function = None,  # Injected embedding function
    ):
        self.connection_string = connection_string
        self.table_name = table_name
        self.dimension = embedding_dimension
        self.embed = embedding_function
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.connection_string)
        return self._pool

    async def search(
        self,
        query: str,
        n_results: int = 10,
        source_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        # Embed query
        query_embedding = await self.embed(query)

        pool = await self._get_pool()
        async with pool.acquire() as conn:
            if source_filter:
                results = await conn.fetch("""
                    SELECT id, source_id, content, url,
                           1 - (embedding <=> $1) as score,
                           metadata
                    FROM source_vectors
                    WHERE source_id = ANY($2)
                    ORDER BY embedding <=> $1
                    LIMIT $3
                """, query_embedding, source_filter, n_results)
            else:
                results = await conn.fetch("""
                    SELECT id, source_id, content, url,
                           1 - (embedding <=> $1) as score,
                           metadata
                    FROM source_vectors
                    ORDER BY embedding <=> $1
                    LIMIT $2
                """, query_embedding, n_results)

        return [dict(r) for r in results]

    async def get_sources(self) -> List[Dict[str, Any]]:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            results = await conn.fetch("""
                SELECT source_id as id, source_id as name, COUNT(*) as count
                FROM source_vectors
                GROUP BY source_id
                ORDER BY source_id
            """)
        return [dict(r) for r in results]
```

---

## Implementation Status

### Phase 1: Create Structure - DONE
- [x] Create `clippy_core/` folder
- [x] Create `__init__.py` with public exports
- [x] Create `config.py` with ClippyConfig dataclass
- [x] Create `schemas.py` with data types

### Phase 2: VectorDB - DONE
- [x] Create `clippy_core/vectordb/` structure
- [x] Create `base.py` with abstract interface
- [x] Create `pgvector.py` (new Supabase implementation)
- [x] Create `factory.py` helper utilities
- [x] Create `__init__.py` with factory that can use existing stores

### Phase 3: Chat/LLM - DONE
- [x] Create `clippy_core/llm.py` (OpenAI/Anthropic abstraction)
- [x] Create `clippy_core/chat.py` (ChatService class)

### Phase 4: Update Existing Code - NOT DONE (intentional)
- [ ] Update `offline_tools/` to import from `clippy_core`
- [ ] Update `admin/` to import from `clippy_core`
- [ ] Update `app.py` to import from `clippy_core`

**This phase is intentionally deferred.** Disaster Clippy continues using existing code paths.
clippy_core is standalone for export only.

### Future: Unify Codebases (optional)
When ready to have disaster-clippy use clippy_core internally:
1. Update imports gradually
2. Remove duplicated code from admin/ai_service.py
3. Test all use cases still work

---

## Files Reference

### Current files to extract from:

**admin/ai_service.py** (~825 lines)
- `AIService` class - main orchestration
- `SearchResult`, `ResponseResult` dataclasses
- `_get_vector_store()` - store selection
- `search()` - unified search with fallback
- `generate_response()` / `generate_response_stream()` - LLM calls
- `_build_prompt()` - prompt construction

**admin/connection_manager.py** (~200 lines)
- `ConnectionManager` class
- Online/offline detection
- API success/failure tracking

**offline_tools/vectordb/store.py** (~400 lines)
- `VectorStore` class (ChromaDB)
- `search()`, `search_offline()`, `add()`, `delete()`

**offline_tools/vectordb/pinecone_store.py** (~300 lines)
- `PineconeStore` class
- Cloud vector operations

**offline_tools/vectordb/factory.py** (~230 lines)
- `get_vector_store()` factory
- Dimension detection
- Mode switching

**offline_tools/embeddings.py** (~150 lines)
- `get_embedding()` function
- OpenAI and local model support

---

## Testing Checklist

### Current State (clippy_core standalone)

- [x] `clippy_core/` imports work standalone
- [x] No circular imports
- [x] Disaster Clippy still works unchanged (`python app.py`)

### If Migrating Disaster Clippy to Use clippy_core

- [ ] Update `app.py` chat routes to use `ChatService`
- [ ] Update or wrap `admin/ai_service.py` to use clippy_core
- [ ] Verify offline mode still works (local LLM fallback)
- [ ] Verify all connection states work (online, hybrid, offline)
- [ ] Admin panel still works
- [ ] Source creation still works

---

## Files Created

```
clippy_core/
├── __init__.py           (2KB)  - Package exports
├── config.py             (6KB)  - ClippyConfig dataclass
├── schemas.py            (5KB)  - SearchResult, ChatMessage, etc.
├── chat.py              (10KB)  - ChatService (main interface)
├── llm.py               (11KB)  - LLM abstraction (OpenAI/Anthropic)
└── vectordb/
    ├── __init__.py       (3KB)  - Factory and exports
    ├── base.py           (4KB)  - Abstract VectorStore interface
    ├── factory.py        (1KB)  - Helper utilities
    └── pgvector.py      (13KB)  - Supabase pgvector implementation
```

---

*Created: February 2026*
*Updated: February 2026 - Initial implementation complete*
*For: disaster-clippy modularization*
