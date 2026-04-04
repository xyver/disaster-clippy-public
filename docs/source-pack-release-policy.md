# Source Pack Release Policy

This document makes Disaster Clippy source-pack release expectations explicit, using the existing validation model (`can_submit` and `can_publish`) as the primary quality gates.

Terminology in this doc:

- `source`: the content set being prepared by a contributor
- `source pack`: the distributable artifact and catalog unit derived from that source
- `public catalog`: the published list of source packs visible to users

---

## Purpose

- Standardize what "complete" means for a source.
- Keep local-admin contribution flow fast.
- Keep production publish flow strict and repeatable.
- Make source quality visible to contributors and reviewers.
- Clarify the full lifecycle from raw source preparation to published-pack consumption.

Primary references:
- `docs/validation.md`
- `docs/admin-guide.md`
- `docs/source-tools.md`

---

## Completeness Is Tiered

A source is not just complete/incomplete; it moves through release states:

1. `Incomplete`
2. `Workable Locally`
3. `Ready to Submit` (`can_submit = true`)
4. `Production Ready` (`can_publish = true`)
5. `Published`

---

## End-To-End Lifecycle

The full source-pack lifecycle has two distinct sides:

### Side 1: Preparing A Source Pack

```text
raw source data
  ->
source tools prepare backup, metadata, manifest, and at least one index
  ->
workable locally
  ->
ready to submit
  ->
production ready
  ->
published
```

### Side 2: Consuming A Published Source Pack

```text
published source-pack catalog
  ->
user installs pack to local machine
  ->
pack appears in local installed catalog
  ->
pack is activated in the runtime catalog
  ->
app uses it for search/chat/offline browsing
```

This document defines the quality gates for Side 1 and the publication/consumption
handoff point between both sides.

---

## What "Workable Locally" Means

`Workable Locally` is the practical middle state between raw preparation and formal
release gating.

It means the source is useful enough for local testing in the app, even if it is not
yet ready for submission or publication.

Typical characteristics:

- backup content exists locally
- `_manifest.json` exists with enough identity/config fields to work on the source
- `_metadata.json` exists
- at least one usable index/vector path exists
- the source can be searched, inspected, or spot-checked in the local runtime

Not guaranteed yet:

- full human verification
- both vector dimensions
- deep integrity validation
- catalog-ready quality for publication

This is the important "good enough to use and iterate on" stage for local admins.

---

## Manifest Requirements

Every source requires these fields in `_manifest.json` before submission:

- `name`: Human-readable display name used in the app and public site.
- `description`: One to two sentences describing what the source is and what it covers. This is the public-facing text shown on the source-pack catalog page at disasterclippy.com. Required before a source can appear in the public catalog.
- `tags`: Array of topic tags used for filtering and discovery.
- `license`: License identifier (CC-BY, CC0, Public Domain, etc.).
- `license_verified`: Boolean, must be true before submission.
- `base_url`: Canonical URL of the source.

The `description` field is not validated automatically but is required for the public catalog. A source without a description will not appear correctly on the public site even if all other validation gates pass.

---

## Gate 1: Ready to Submit (`can_submit`)

For local admin -> submissions queue.

Required:
- `_manifest.json` exists and valid (including `name`, `description`, `license`, `base_url`).
- `_metadata.json` exists and valid.
- Backup content exists and passes minimum size checks.
- At least one vector dimension exists (`_vectors.json` or `_vectors_768.json`).
- License is allowed (or Custom with notes) and human-verified.
- Offline links verified.
- Online links verified.

Not required:
- Both vector dimensions.
- Deep integrity scan.

Why: local contributors can submit useful work without paying full global processing cost.

Interpretation:

- `Workable Locally` means "I can test and improve this source in the app."
- `Ready to Submit` means "This source is standardized enough to hand off for global
  review."

---

## Gate 2: Production Ready (`can_publish`)

For global admin -> production cloud publish.

Required:
- All `can_submit` checks pass.
- Both vector dimensions exist:
  - 768-dim for offline/local semantic search.
  - 1536-dim for online/Pinecone semantic search.
- Deep validation passes.
- Integrity checks pass.

Why: production users need reliable behavior across online/offline modes.

Interpretation:

- `Production Ready` means the source pack is clean enough to publish repeatably into
  the shared download/catalog system.

---

## Validation Tiers

### Light Validation (fast, cached)

Used for:
- source cards
- badges/status boxes
- page loads
- quick submit gating

Checks:
- file existence
- parseability
- minimum field sanity
- license + human flags
- basic backup presence/size

### Deep Validation (fresh run)

Used for:
- publish gating
- explicit final review

Checks:
- vector dimension and value integrity (NaN/Inf/null/zero scans)
- metadata <-> vector ID cross-reference
- backup-type-specific content checks (HTML/ZIM/PDF)
- schema/version consistency checks

---

## Human QA Requirements

Human verification is mandatory, not optional metadata:

- `license_verified`
- `links_verified_offline`
- `links_verified_online`

Reviewer checklist:
- attribution and redistribution rights are clear
- sample local links open correctly
- sample online links resolve correctly
- content quality spot check (spam/duplication/garbage)
- tags and discovery metadata are reasonable

---

## Publish Workflow (Global Admin)

1. Intake submission.
2. Check available vector dimensions.
3. Generate missing dimension if needed.
4. Review tags/license/URLs/content quality.
5. Confirm `description` field in `_manifest.json` is written and accurate.
6. Run deep validation.
7. Publish to cloud targets (R2 + Pinecone).
8. Regenerate `published/catalog.json` via `generate_public_catalog()` in `cloud_upload.py`. This merges `_master.json` with per-source manifest data and uploads a single flat catalog file to R2 that the public site reads.
9. Mark source `Published`.

---

## Publication Handoff

Once a source reaches `Published`, it crosses from the preparation side into the
consumption side of the system.

That handoff means:

- the source pack is now allowed to appear in the public catalog
- local users can discover it through the catalog/storefront flow
- local installs should treat it as a stable published artifact, not an editable working
  folder
- future fixes or improvements should produce a revised published source-pack version,
  not mutate the meaning of the already-published gate

This is the boundary between:

- Source Tools workflow
- Pack catalog / install / activate workflow

---

## Consumption States After Publication

After publication, a source pack moves through a different set of states on a user's
machine:

1. `Available`
   Present in the published catalog.
2. `Installed`
   Present in the local `BACKUP_PATH` data root.
3. `Active`
   Selected for the local runtime catalog and used by the app.

These are not QA gates. They are consumption states.

That distinction matters:

- `can_submit` and `can_publish` standardize preparation quality
- available / installed / active standardize runtime consumption behavior

---

## Operational Rules

- Never publish without deep validation.
- Never publish without both vector dimensions.
- Cache light validation for UI responsiveness; bypass cache for final publish decisions.
- Treat published source-pack versions as immutable artifacts.

---

## Suggested Status Messaging

Use concise contributor-facing status text:

- `Incomplete`: missing required files/flags.
- `Workable Locally`: usable for local testing, but not yet standardized for handoff.
- `Ready to Submit`: acceptable for queue handoff.
- `Production Ready`: publishable now.
- `Published`: live in production distribution.

---

## Why This Model Works

- Enables community contribution with lower initial burden.
- Preserves high production quality standards.
- Supports dual-mode architecture (offline/local and online/cloud).
- Scales review effort through explicit gates and automation.

---

*Project-specific policy document for Disaster Clippy, aligned with current validation and admin workflows.*
