# Deployment Guide

Canonical reference for how Disaster Clippy is intended to be deployed, distributed, and split across public and private surfaces.

This document replaces the older deployment note that was focused mainly on infrastructure scenarios. The project now needs a clearer product/runtime model.

---

## Overview

Disaster Clippy is no longer just "a FastAPI app you can run locally."

It now needs to support four related surfaces:

1. Hosted public app
2. Marketing and catalog site
3. Downloadable offline/local runtime
4. Downloadable advanced local admin/source-creation toolkit

These are all the same project family, but they are not the same deployment target.

---

## Runtime Modes

The codebase already has three runtime roles in [`admin/app.py`](/C:/Users/Bryan/Desktop/disaster-clippy-public/admin/app.py#L35):

| Mode | Purpose | Typical deployment |
|------|---------|--------------------|
| `local` | Full local admin with local storage and local source creation | desktop install, laptop, server, Raspberry Pi |
| `pinecone` | Hosted public runtime with admin UI blocked | public app / hosted `.io` |
| `global` | Maintainer/global-admin runtime with shared cloud write access | maintainer machine or private ops environment |

These should be treated as intentional product/runtime modes, not temporary flags.

---

## Public and Private Split

Disaster Clippy should be understood as a sibling workspace with separate repos:

```text
disaster-clippy/
├── public/
└── private/
```

### `public/`

Owns:

- core application runtime
- offline/local install
- local admin panel
- source creation and validation tools
- source pack consumption
- shared documentation
- portable architecture targets like `clippy_core`

This repo should remain the main product engine.

### `private/`

Owns:

- `.com` marketing/product site
- account and profile layer
- update/control-plane services
- hosted operational tooling
- optional billing/donations/entitlements if ever added
- private operational docs

This repo should not absorb the whole product engine. It is the shell around the public engine, not a replacement for it.

---

## Domain Split

The intended domain model is:

### `.io`

The real hosted app.

Responsibilities:

- public search/chat experience
- hosted runtime using shared cloud vectors
- lightweight account-aware customization
- pack-aware search behavior for signed-in users

This should feel like "use Disaster Clippy now."

### `.com`

The product, catalog, and download site.

Responsibilities:

- explain what Disaster Clippy is
- onboarding and install guides
- source pack library / catalog
- release notes
- account/profile management
- future update feeds and manifests

This should feel closer to Kiwix’s main website model:

- a product site first
- a catalog/discovery surface second
- operational/account features layered in later

---

## Product Surfaces

### 1. Hosted Public App

Audience:

- casual users
- researchers
- users who just want to search or ask questions

Behavior:

- runs in `pinecone` mode
- no local admin UI
- no public source creation workflow
- uses shared cloud-hosted vector/search infrastructure

### 2. Local Runtime

Audience:

- offline users
- preparedness groups
- Raspberry Pi and LAN deployments
- users syncing selected packs for offline use

Behavior:

- runs in `local` mode
- supports local models and offline search
- can sync or install chosen packs from the global registry
- should work without the hosted service once set up

### 3. Advanced Local Admin Toolkit

Audience:

- maintainers
- community contributors
- power users building custom packs

Behavior:

- still local-first
- includes source creation, validation, scraping, indexing, translation, video processing, and future OCR tools
- not intended to become a public hosted source-creation SaaS

This is important: source creation should remain a local advanced workflow unless there is a very strong future reason to expose part of it publicly.

### 4. Private Control Plane

Audience:

- maintainers
- future operators of hosted services

Behavior:

- manages update manifests, pack catalog metadata, account linkage, and hosted operational concerns
- belongs in the private repo

---

## Account and Pack Model

The intended user flow is:

1. User discovers Disaster Clippy on `.com`
2. User explores the pack library and product docs
3. User optionally creates an account
4. User selects packs, languages, and preferences from the global registry
5. User can:
   - use the hosted `.io` app with that profile
   - download the local runtime and sync that same profile offline
   - optionally download the advanced local admin toolkit if they want to build or curate sources

The account is not primarily for source creation. Its main job is profile and pack selection across hosted and offline use.

---

## Update System

The project needs a clearer update model going forward.

There are at least four distinct update channels:

| Update type | Examples |
|-------------|----------|
| App updates | new code release, bug fixes, UI changes |
| Source pack updates | new packs, revised packs, metadata updates |
| Model updates | embeddings, local LLM packs, language packs |
| Control/catalog updates | pack manifests, compatibility flags, deprecations |

### Recommended approach

Use signed or trusted manifests/catalogs rather than ad hoc download links.

Local installs should be able to check for:

- app updates
- pack updates
- model/language pack updates
- compatibility notices

This belongs primarily in the private control-plane side, but the public repo should be designed to consume those manifests cleanly.

---

## Layered Architecture

Deployment and repo boundaries work better if the internal architecture stays layered:

```text
app.py
admin/
offline_tools/
clippy_core/   (future extraction target)
```

See [`clippy-core-extraction.md`](./clippy-core-extraction.md).

That separation matters because:

- the hosted shell should not own all business logic
- the private repo should not need to duplicate the public app engine
- the local runtime should remain first-class
- the portable chat/search engine should eventually be reusable outside the full app

---

## Supported Deployment Scenarios

### Self-Hosted Local Server

Use case:

- personal server
- VPS
- LAN node
- family/community preparedness hub

Typical mode:

- `local`

What you get:

- full admin UI
- local backups and local vectors
- optional cloud sync
- optional local models

### Raspberry Pi or Air-Gapped Node

Use case:

- disaster preparedness node
- local network knowledge appliance
- field/offline deployment

Typical mode:

- `local`

What matters most:

- local storage
- local vector DB
- local LLM or fallback search-only behavior
- pack install/update process that can work from local files when needed

### Hosted Public App

Use case:

- public `.io` deployment

Typical mode:

- `pinecone`

What you get:

- hosted search/chat
- no admin UI
- cloud-hosted shared vectors
- account-aware pack selection later

### Global Maintainer Environment

Use case:

- official pack publishing
- cloud vector sync
- hosted ops and submissions review

Typical mode:

- `global`

What you get:

- write access to shared cloud systems
- submissions review and publishing authority
- private operational tooling

---

## What Stays Public vs Private

### Public repo should own

- app runtime
- local admin and source tools
- offline capability
- source pack install/consume logic
- translation/video/OCR processing pipelines
- public docs

### Private repo should own

- `.com` site
- account/profile services
- update/control plane
- hosted operational services
- any future commercial/account infrastructure

This separation keeps the open-source engine strong while still allowing a richer hosted product layer.

---

## Environment and Infrastructure Notes

The older deployment ideas are still valid in principle:

- local installs may use ChromaDB and local storage
- hosted public deployments may use Pinecone and cloud storage
- maintainers may need broader write access than public runtimes
- air-gapped deployments remain a first-class requirement

But those are implementation details underneath the larger product/deployment model above.

For current admin and cloud setup details, also see:

- [`architecture.md`](./architecture.md)
- [`admin-guide.md`](./admin-guide.md)

---

## Recommended Next Docs

This deployment model should eventually be complemented by:

- a private-repo control-plane doc
- an update-manifest/catalog doc
- a runtime/profile sync doc
- a `.com` product-site information architecture doc

---

*Updated: March 15, 2026*  
*Status: current deployment and distribution direction*
