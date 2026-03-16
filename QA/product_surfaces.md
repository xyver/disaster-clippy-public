# Product Surface QA

This note defines the main product surfaces QA should care about now that Disaster Clippy is no longer framed as only one local/global app.

## Surfaces

### Hosted Public App

What QA should verify:

- users can ask practical questions immediately
- responses remain grounded
- source filtering works
- admin-only concepts are not exposed as normal user requirements

### Local Runtime

What QA should verify:

- local setup instructions still make sense
- offline-first language stays accurate
- local embeddings and local LLM paths do not silently regress
- source packs remain a valid unit for local installation

### Advanced Local Admin

What QA should verify:

- source creation and validation remain local-first workflows
- translation workflows still match the docs
- video transcript acquisition and preparation still match the docs
- future OCR work can slot into the same acquire/generate-text mental model

### Product and Catalog Site

What QA should verify:

- site language matches public repo docs
- hosted vs local vs advanced-admin distinctions are clear
- source packs are explained consistently
- account/profile/update claims are not made prematurely

## Main Drift Risks

- old docs still speaking in the earlier local/global-only framing
- site copy promising product behavior that does not exist yet
- admin workflows being described as if they are part of normal hosted use
- pack model being described differently across README, docs, and site pages

## What To Automate First

1. Hosted API contract and retrieval checks
2. Simple route checks for public site/docs pages
3. Content linting for core wording:
   - hosted app
   - local runtime
   - source packs
   - local admin
4. Pack-model consistency checks once pack pages become real data-driven pages
