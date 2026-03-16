# System Architecture

This document describes the current architecture of Disaster Clippy at a higher level than the older local/global-only framing.

The project should now be understood as:

- a public application/runtime repo
- a private shell/control-plane repo
- a hosted app surface
- a local runtime surface
- an advanced local admin/source-creation surface

---

## Architectural Layers

Disaster Clippy works best when understood in four layers:

```text
app.py
admin/
offline_tools/
clippy_core/   (future extraction target)
```

See [`clippy-core-extraction.md`](clippy-core-extraction.md).

Dependency direction should flow downward:

- `app.py` can depend on `admin/`, `offline_tools/`, and future `clippy_core/`
- `admin/` can depend on `offline_tools/` and future `clippy_core/`
- `offline_tools/` can depend on future `clippy_core/`
- `clippy_core/` should not depend on the layers above it

---

## Runtime Modes

The codebase already supports three runtime roles through `VECTOR_DB_MODE`:

| Mode | Purpose |
|------|---------|
| `local` | full local runtime and local admin |
| `pinecone` | hosted public runtime |
| `global` | maintainer/global admin runtime |

These are not just technical toggles. They map to real product and operational surfaces.

---

## Public and Private Repos

The intended workspace structure is:

```text
disaster-clippy/
+-- disaster-clippy-public/
+-- disaster-clippy-private/
```

### Public repo responsibilities

- core application runtime
- local runtime
- local admin/source tools
- source processing and validation
- translation, video, and future OCR pipelines
- shared public docs

### Private repo responsibilities

- `.com` product and catalog site
- future account/profile layer
- future update/control-plane services
- private operational docs and hosted glue

The private repo should be the shell around the public engine, not a second copy of the whole engine.

---

## Product Surfaces

### Hosted App

The hosted app is the public runtime.

Characteristics:

- easy entry point
- no public source creation workflow
- shared cloud vector/search infrastructure
- account-aware pack/profile behavior later

### Local Runtime

The local runtime is for offline-first and self-hosted use.

Characteristics:

- local storage
- local search and optional local LLMs
- source pack installation and sync
- LAN and Raspberry Pi friendly direction

### Advanced Local Admin

The advanced admin/tooling layer is local-first.

Characteristics:

- scraping
- indexing
- validation
- translation
- video transcript workflows
- future OCR

This should remain a power-user workflow rather than becoming a public hosted source-builder.

---

## Source Pack Model

Source packs are the main unit of knowledge distribution and selection.

A source pack may contain:

- HTML backups
- ZIM archives
- PDFs
- metadata
- vector indexes
- transcript and translation artifacts

This model allows:

- hosted discovery
- local installation
- future account/profile sync across environments

---

## Data and Search Flow

At a high level:

1. Sources are prepared locally through backup/import/processing pipelines
2. Metadata and vectors are generated
3. Packs can be consumed locally or represented in hosted catalogs
4. User queries search the relevant vector store and return grounded responses

The exact infrastructure differs by mode:

| Mode | Main store |
|------|------------|
| `local` | local ChromaDB and local source data |
| `pinecone` | hosted cloud vector/search runtime |
| `global` | maintainer cloud write and publishing context |

---

## Offline-First Design

Offline capability remains central.

Important points:

- local search should remain first-class, not a degraded afterthought
- local installs may use local embeddings and local LLMs
- source packs should be portable between hosted discovery and local use
- translation and video pipelines should produce durable local artifacts
- future OCR should follow the same acquire-or-generate-text model used elsewhere

---

## Website and Domain Model

The architecture now assumes an eventual split between:

### `.io`

- the real hosted app
- public search/chat runtime

### `.com`

- product explanation site
- source pack library
- install/onboarding/docs surface
- later account/profile/update/catalog layer

See [`deployment.md`](deployment.md) for the fuller product/distribution framing.

---

## Current Priority Areas

The most important architectural workstreams now are:

- source pack distribution and selection
- update/catalog manifest design
- hosted vs local profile alignment
- video transcript acquisition and translation pipeline
- future OCR for scanned PDFs
- keeping the advanced admin path local-first

---

## Related Docs

- [`deployment.md`](deployment.md)
- [`clippy-core-extraction.md`](clippy-core-extraction.md)
- [`source-tools.md`](source-tools.md)
- [`language-packs.md`](language-packs.md)
- [`video_processing.md`](video_processing.md)

---

*Updated: March 15, 2026*
