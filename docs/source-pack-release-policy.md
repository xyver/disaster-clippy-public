# Source Pack Release Policy

This document makes Disaster Clippy source-pack release expectations explicit, using the existing validation model (`can_submit` and `can_publish`) as the primary quality gates.

---

## Purpose

- Standardize what "complete" means for a source.
- Keep local-admin contribution flow fast.
- Keep production publish flow strict and repeatable.
- Make source quality visible to contributors and reviewers.

Primary references:
- `docs/validation.md`
- `docs/admin-guide.md`
- `docs/source-tools.md`

---

## Completeness Is Tiered

A source is not just complete/incomplete; it moves through release states:

1. `Incomplete`
2. `Ready to Submit` (`can_submit = true`)
3. `Production Ready` (`can_publish = true`)
4. `Published`

---

## Manifest Requirements

Every source requires these fields in `_manifest.json` before submission:

- `name`: Human-readable display name used in the app and public site.
- `description`: One to two sentences describing what the source is and what it covers. This is the public-facing text shown on the collections page at disasterclippy.com. Required before a source can appear in the public catalog.
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

## Operational Rules

- Never publish without deep validation.
- Never publish without both vector dimensions.
- Cache light validation for UI responsiveness; bypass cache for final publish decisions.
- Treat published pack versions as immutable artifacts.

---

## Suggested Status Messaging

Use concise contributor-facing status text:

- `Incomplete`: missing required files/flags.
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
