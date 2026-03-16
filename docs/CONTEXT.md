# Disaster Clippy - AI Assistant Context

Read this first when orienting to the repo.

This document is meant to help quickly answer:

- what the project is now
- how the docs are organized
- which layer owns which kind of work

---

## What This Project Is

Disaster Clippy is an offline-capable preparedness knowledge system built around:

- conversational search
- source packs
- hosted and local runtime paths
- advanced local admin/source tooling

It is no longer best described as just a single local/global app.

The project now needs to be understood as four related surfaces:

1. hosted public app
2. product and catalog site
3. local runtime
4. advanced local admin toolkit

---

## Quick Orientation

| Term | Meaning |
|------|---------|
| **Source** | A collection of documents or media-derived text under one source ID |
| **Source Pack** | The distributable unit of knowledge selection and installation |
| **BACKUP_PATH** | Local folder where source-owned data and local artifacts live |
| **Local Runtime** | Self-hosted or offline-capable user install |
| **Advanced Local Admin** | Local source creation, validation, translation, and processing tools |
| **Global / Hosted Runtime** | Shared public hosted app and maintainer cloud context |
| **ZIM** | Compressed offline archive format |

---

## Best Docs To Read First

### Product and Deployment Direction

| Doc | Why it matters |
|-----|----------------|
| [deployment.md](deployment.md) | The clearest current statement of hosted vs local vs private-shell direction |
| [architecture.md](architecture.md) | Layering, runtime roles, and repo boundaries |
| [clippy-core-extraction.md](clippy-core-extraction.md) | Future portable-core seam |

### Working with Sources

| Doc | Why it matters |
|-----|----------------|
| [source-tools.md](source-tools.md) | Source creation and processing workflows |
| [validation.md](validation.md) | Readiness and release gates |
| [source-pack-release-policy.md](source-pack-release-policy.md) | Pack publishing expectations |

### Language, Video, and Processing Pipelines

| Doc | Why it matters |
|-----|----------------|
| [language-packs.md](language-packs.md) | Translation direction |
| [video_processing.md](video_processing.md) | Video transcript pipeline |
| [video_processing_plan.md](video_processing_plan.md) | Build plan and storage model |
| [document-type-weighting.md](document-type-weighting.md) | PDF behavior, document classification, relevance |

### Runtime and User-Facing Behavior

| Doc | Why it matters |
|-----|----------------|
| [ai-service.md](ai-service.md) | Search and response path |
| [api.md](api.md) | External API surface |
| [admin-guide.md](admin-guide.md) | Current local admin usage |
| [jobs.md](jobs.md) | Background work and checkpoints |

---

## Working Area Map

### Hosted App / Public Runtime

Read:

- `app.py`
- `admin/ai_service.py`
- `docs/api.md`
- `docs/deployment.md`

### Local Admin and Source Tools

Read:

- `admin/app.py`
- `admin/routes/source_tools.py`
- `admin/job_manager.py`
- `docs/source-tools.md`
- `docs/admin-guide.md`

### Source Processing

Read:

- `offline_tools/source_manager.py`
- `offline_tools/indexer.py`
- `offline_tools/validation.py`
- `docs/validation.md`

### Translation and Language Packs

Read:

- `offline_tools/translation.py`
- `offline_tools/language_registry.py`
- `docs/language-packs.md`

### Video Processing

Read:

- `offline_tools/video_analysis.py`
- `offline_tools/transcript_acquisition.py`
- `offline_tools/youtube_transcript.py`
- `docs/video_processing.md`
- `docs/video_processing_plan.md`

### Future Portable Core Direction

Read:

- `docs/clippy-core-extraction.md`

---

## Repo and Workspace Direction

The core repo here is the public engine.

The broader workspace direction is:

```text
disaster-clippy/
+-- disaster-clippy-public/
+-- disaster-clippy-private/
```

Where:

- public repo owns runtime, local tooling, and public docs
- private repo owns the `.com` site and future control-plane shell

---

## Current Priorities

Current architectural priorities are roughly:

- unify hosted/local/product-site language across docs
- strengthen source pack model and catalog thinking
- build the `.com` side as a product site first
- keep source creation local-first
- continue video transcript and translation pipeline work
- add OCR support for scanned PDFs later

---

## If You Are Updating the Website Copy

Use these as the main source of truth:

- [deployment.md](deployment.md)
- [architecture.md](architecture.md)
- [source-tools.md](source-tools.md)
- [language-packs.md](language-packs.md)
- [video_processing.md](video_processing.md)

Avoid relying only on older phrasing in `README.md` history or legacy local/global wording.

---

*Updated: March 15, 2026*
