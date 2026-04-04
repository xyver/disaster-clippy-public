# Disaster Clippy - Executive Summary

Disaster Clippy is a preparedness knowledge system built around conversational search, source packs, and offline-capable local deployments.

It is best understood as four related product surfaces:

1. hosted public app
2. product and catalog site
3. local runtime
4. advanced local admin/source creation toolkit

---

## Core Idea

Users ask practical questions in plain language and get grounded answers from curated preparedness and resilience material.

The project is designed so that this experience can work:

- in a hosted app
- in a self-hosted local install
- on constrained offline hardware
- with selected knowledge packs carried locally

---

## What Makes It Different

### Source Packs

Knowledge is packaged into source packs so users can:

- see what content they are using
- choose what to install or sync
- move the same pack choices between hosted and local use

### Offline-First Direction

The system is meant to remain useful even when internet access is unreliable or absent.

### Advanced Admin Stays Local

Source creation, scraping, indexing, translation, video processing, and future OCR workflows are treated as advanced local tooling, not a public hosted workflow.

---

## Main User Paths

### Hosted User

- opens the public app
- searches and chats immediately
- may later sign in and connect a pack/profile selection

### Local Runtime User

- installs the system locally
- uses selected packs offline
- can pair it with local embeddings and local LLMs

### Advanced Local Admin

- creates or curates sources
- runs validation and indexing
- manages translation/video/OCR workflows

### Maintainer / Global Operator

- manages official packs and cloud infrastructure
- operates the hosted/control-plane side

---

## Current Direction

The project is moving toward a clearer split between:

- public repo: runtime, local tooling, open engine
- private repo: `.com` site, control plane, account/update layer

This keeps the core engine open and reusable while allowing a richer hosted product shell around it.

---

## Best Documents For Orientation

- [`README.md`](../README.md)
- [`deployment.md`](deployment.md)
- [`distribution-alignment.md`](distribution-alignment.md)
- [`architecture.md`](architecture.md)
- [`source-tools.md`](source-tools.md)
- [`language-packs.md`](language-packs.md)
- [`video_processing.md`](video_processing.md)
- [`clippy-core-extraction.md`](clippy-core-extraction.md)

---

*Updated: March 15, 2026*
