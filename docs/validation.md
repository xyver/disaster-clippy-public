# Validation System

This document covers the unified validation system for source quality control, including permission gates, validation tiers, and human verification.

---

## Table of Contents

1. [Overview](#overview)
2. [Permission Gates](#permission-gates)
3. [Validation Tiers](#validation-tiers)
4. [Human Verification Flags](#human-verification-flags)
5. [License Handling](#license-handling)
6. [Validation Caching](#validation-caching)
7. [Backup Validation](#backup-validation)
8. [Vector Validation](#vector-validation)
9. [UI Display](#ui-display)
10. [Validation-to-Tool Mapping](#validation-to-tool-mapping)
11. [ValidationResult Dataclass](#validationresult-dataclass)

---

## Overview

The validation system provides:

1. **Two validation tiers:** Light (fast, for UI) and Deep (thorough, for publishing)
2. **Two permission gates:** `can_submit` (local admin) and `can_publish` (global admin)
3. **Three human verification flags:** license, offline links, online links
4. **Validation caching:** Persistent status file with mtime-based invalidation

**File:** `offline_tools/validation.py`

---

## Permission Gates

### Gate 1: `can_submit` (Local Admin -> Submissions Queue)

**Purpose:** Allow local admins to submit sources for global review.

**Requirements:**
- has_manifest (manifest exists and valid JSON)
- has_metadata (metadata exists and valid JSON)
- has_backup (backup exists and >= 0.1 MB)
- has_at_least_one_index (1536 OR 768 vectors exist)
- license in allowlist OR "Custom" with notes
- license_verified == true (human confirmed)
- links_verified_offline == true (human confirmed)
- links_verified_online == true (human confirmed)

**NOT required:**
- Both vector dimensions (global admin fills in missing one)
- Deep validation (global admin does this)

```python
def can_submit(r: ValidationResult) -> bool:
    """Local admin gate - content + one vector dimension required."""
    has_at_least_one_index = r.has_vectors_1536 or r.has_vectors_768

    return (
        r.has_manifest and
        r.has_metadata and
        r.has_backup and
        r.backup_size_mb >= 0.1 and
        has_at_least_one_index and
        r.license_in_allowlist and
        r.license_verified and
        r.links_verified_offline and
        r.links_verified_online
    )
```

### Gate 2: `can_publish` (Global Admin -> Production)

**Purpose:** Publish to R2 backups + Pinecone for end users.

**Requirements:**
- All `can_submit` requirements pass
- has_vectors_768 (for offline users)
- has_vectors_1536 (for online users)
- Deep validation passes (all integrity checks)

```python
def can_publish(r: ValidationResult) -> bool:
    """Global admin gate - both dimensions + deep validation."""
    return (
        can_submit(r) and
        r.has_vectors_768 and
        r.has_vectors_1536 and
        r.deep_validated and
        r.integrity_passed
    )
```

---

## Validation Tiers

### Light Validation (< 100ms per source)

**Used for:** UI badges, page loads, quick status checks

| Category | Check | Method |
|----------|-------|--------|
| **Files Exist** | `_manifest.json` present | `os.path.exists()` |
| | `_metadata.json` present | `os.path.exists()` |
| | `_vectors.json` present | `os.path.exists()` |
| | `_vectors_768.json` present | `os.path.exists()` |
| | Backup exists (pages/ OR *.zim OR *.pdf) | `os.path.exists()` |
| **JSON Valid** | Manifest parseable | `json.load()` succeeds |
| | Metadata parseable | `json.load()` succeeds |
| **Field Check** | `source_id` in manifest | Key exists |
| | `document_count` > 0 | Value check |
| **Size Check** | Backup >= 0.1 MB | `os.path.getsize()` |
| **License** | License in allowlist OR "Custom" | String match |
| | If "Custom", has notes >= 20 chars | Length check |
| **Human Flags** | `license_verified` | Boolean check |
| | `links_verified_offline` | Boolean check |
| | `links_verified_online` | Boolean check |

### Deep Validation (5-30 seconds for large sources)

**Used for:** Publish button, "Validate" action, production gate

Includes ALL light validation checks, PLUS:

| Category | Check | Method |
|----------|-------|--------|
| **Vector Integrity** | All 1536 vectors correct dimension | Loop + length check |
| | All 768 vectors correct dimension | Loop + length check |
| | No NaN values | `math.isnan()` scan |
| | No Infinity values | `math.isinf()` scan |
| | No null/None values | Value check |
| | No all-zero vectors | Sum check |
| **ID Cross-Reference** | Metadata IDs == Vector 1536 IDs | Set comparison |
| | Metadata IDs == Vector 768 IDs | Set comparison |
| | No duplicate IDs in metadata | Set vs list length |
| **Backup Content** | HTML: Each metadata doc has file in pages/ | Path verification |
| | ZIM: Valid magic bytes (`ZIM\x04`) | Read first 4 bytes |
| | ZIM: Has entries (if libzim available) | `Archive.entry_count` |
| | PDF: Valid header (`%PDF`) | Read first 4 bytes |
| **Manifest** | `schema_version` == 3 | Exact match |
| | `source_id` matches folder name | String compare |
| | `backup_type` matches actual content | Type check |

### Computation Cost Comparison

| Operation | Time (10K docs) | Resource | Cost |
|-----------|-----------------|----------|------|
| Generate 1536 embeddings | 10-30 min | OpenAI API | ~$0.50-2.00 |
| Generate 768 embeddings | 5-15 min | Local CPU/GPU | Free |
| Deep scan vectors | 5-30 sec | Disk I/O + CPU | Negligible |
| Light validation | < 100 ms | Disk I/O | Negligible |

**Key insight:** Generation is 100-1000x slower than validation. Requiring local admins to generate one dimension saves 50% of global admin work.

---

## Human Verification Flags

Three boolean flags requiring human review:

| Flag | Purpose | Stored In |
|------|---------|-----------|
| `license_verified` | Human confirmed license is accurate | `_manifest.json` |
| `links_verified_offline` | Human confirmed internal links work | `_manifest.json` |
| `links_verified_online` | Human confirmed external URLs valid | `_manifest.json` |

All three must be `true` for `can_submit` gate.

### Manifest Schema

```json
{
  "source_id": "example-source",
  "name": "Example Source",
  "schema_version": 3,
  "backup_type": "html",
  "license": "CC-BY-SA",
  "license_verified": true,
  "license_notes": "",
  "links_verified_offline": true,
  "links_verified_online": true
}
```

---

## License Handling

### Allowed Licenses

```python
ALLOWED_LICENSES = [
    "CC0",
    "CC-BY",
    "CC-BY-SA",
    "CC-BY-NC",
    "CC-BY-NC-SA",
    "Public Domain",
    "MIT",
    "Apache-2.0",
    "GPL",
    "GFDL",
    "ODbL",
    "Custom"  # Requires license_notes
]
```

### Custom License Handling

If `license == "Custom"`:
- `license_notes` field is REQUIRED
- Must be >= 20 characters
- Should explain why distribution is allowed

Example:
```json
{
  "license": "Custom",
  "license_verified": true,
  "license_notes": "Author granted explicit permission for offline distribution via email 2024-01-15. Original site terms allow non-commercial redistribution."
}
```

---

## Validation Caching

### Cache File: `_validation_status.json`

Each source folder contains a validation cache file:

```json
{
  "last_validated": "2024-12-13T10:30:00Z",
  "file_mtimes": {
    "_manifest.json": 1702468200.0,
    "_metadata.json": 1702468200.0,
    "_vectors.json": 1702468200.0,
    "_vectors_768.json": 1702468200.0
  },
  "result": {
    "can_submit": true,
    "can_publish": false,
    "has_manifest": true,
    "has_metadata": true,
    "has_backup": true,
    "has_vectors_1536": true,
    "has_vectors_768": false,
    "license_verified": true,
    "links_verified_offline": true,
    "links_verified_online": true,
    "backup_size_mb": 45.2,
    "document_count": 1234
  }
}
```

### Cache Invalidation

Cache automatically invalidates when:
- Any source file is modified (mtime changes)
- Cache file is deleted
- Job completes for that source (explicitly clear cache)

### Validation Contexts

| Context | Validation Level | Uses Cache? |
|---------|-----------------|-------------|
| Sources page load | Light | Yes |
| Cloud page load | Light | Yes |
| Source card badges | Light | Yes |
| "Validate" button | Deep | No (fresh) |
| Submit to queue | Light | Yes |
| Publish to production | Deep | No (fresh) |
| Job completion | Clears cache | N/A |

---

## Backup Validation

### HTML Sources

```python
def validate_html_backup(source_path: Path, deep: bool = False) -> tuple[bool, list]:
    pages_dir = source_path / "pages"
    errors = []

    if not pages_dir.exists():
        errors.append("pages/ directory missing")
        return False, errors

    html_files = list(pages_dir.glob("**/*.html"))
    if len(html_files) == 0:
        errors.append("No HTML files in pages/")

    if deep:
        # Verify each metadata doc has corresponding file
        pass

    return len(errors) == 0, errors
```

### ZIM Sources

```python
def validate_zim_backup(zim_path: Path, deep: bool = False) -> tuple[bool, list]:
    errors = []

    # Light: Check magic bytes
    with open(zim_path, 'rb') as f:
        magic = f.read(4)
        if magic != b'ZIM\x04':
            errors.append("Invalid ZIM magic bytes")
            return False, errors

    if deep:
        # Deep: Check with libzim if available
        try:
            from libzim.reader import Archive
            zim = Archive(str(zim_path))
            if zim.entry_count == 0:
                errors.append("ZIM has no entries")
            if not zim.has_main_entry:
                errors.append("ZIM has no main entry page")
        except ImportError:
            pass  # libzim not available

    return len(errors) == 0, errors
```

### PDF Sources

```python
def validate_pdf_backup(pdf_path: Path, deep: bool = False) -> tuple[bool, list]:
    errors = []

    # Light: Check header
    with open(pdf_path, 'rb') as f:
        header = f.read(4)
        if header != b'%PDF':
            errors.append("Invalid PDF header")
            return False, errors

    if deep:
        # Deep: Try to read with PDF library
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            if len(reader.pages) == 0:
                errors.append("PDF has no pages")
        except ImportError:
            pass  # pypdf not available

    return len(errors) == 0, errors
```

---

## Vector Validation

```python
def validate_vectors(vector_file: Path, expected_dim: int) -> tuple[bool, list]:
    """Deep validation of vector file for corruption."""
    errors = []
    try:
        with open(vector_file, 'r') as f:
            data = json.load(f)

        vectors = data if isinstance(data, list) else data.get("vectors", [])

        for i, vec in enumerate(vectors):
            doc_id = vec.get("id", f"index_{i}")
            values = vec.get("values", [])

            if not values:
                errors.append(f"{doc_id}: empty vector")
                continue

            if len(values) != expected_dim:
                errors.append(f"{doc_id}: dimension {len(values)}, expected {expected_dim}")

            # Check for corruption
            for j, val in enumerate(values):
                if val is None:
                    errors.append(f"{doc_id}: null at position {j}")
                    break
                elif math.isnan(val):
                    errors.append(f"{doc_id}: NaN at position {j}")
                    break
                elif math.isinf(val):
                    errors.append(f"{doc_id}: Infinity at position {j}")
                    break

            # Check for all-zeros (failed embedding)
            if values and all(v == 0 for v in values):
                errors.append(f"{doc_id}: all-zero vector")

    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")

    return len(errors) == 0, errors
```

---

## UI Display

### Status Boxes (6 items)

| Box | Shows | Green | Yellow | Red |
|-----|-------|-------|--------|-----|
| Config | `has_manifest` | Has manifest | - | Missing |
| Backup | `has_backup` | Has content | - | Missing |
| Metadata | `has_metadata` | Has docs | - | Missing |
| 1536 | `has_vectors_1536` | Has vectors | Missing | - |
| 768 | `has_vectors_768` | Has vectors | Missing | - |
| License | License status | Verified | Exists | Unknown |

### Status Badges

| Condition | Badge Text | Color | Meaning |
|-----------|------------|-------|---------|
| Missing requirements | "Incomplete" | Orange | Needs work |
| `can_submit` passed | "Ready to Submit" | Blue | Local admin can submit |
| `can_publish` passed | "Production Ready" | Green | Global admin can publish |
| Already published | "Published" | Green (dim) | In production |

---

## Validation-to-Tool Mapping

### How to Pass Each Check

| Check | Tool/Action | Page | Description |
|-------|-------------|------|-------------|
| **has_manifest** | Auto-created | Jobs | Created when source folder exists |
| **has_metadata** | "Create Metadata" job | Jobs | Extracts titles, URLs from backup |
| **has_backup** | Manual or Scraper | External | Copy pages/, ZIM, or PDF to source folder |
| **backup >= 0.1MB** | - | - | Backup must have content |
| **has_vectors_1536** | "Create Index (Online)" job | Jobs | Generates OpenAI embeddings |
| **has_vectors_768** | "Create Index (Offline)" job | Jobs | Generates local model embeddings |
| **license_in_allowlist** | Edit Source | Sources | Select license from dropdown |
| **license_verified** | Edit Source checkbox | Sources | Human confirms license is correct |
| **links_verified_offline** | Edit Source checkbox | Sources | Human confirms internal links work |
| **links_verified_online** | Edit Source checkbox | Sources | Human confirms external URLs valid |

### Fix Action URLs

| Missing Check | Fix Action | Button Text | Redirect URL |
|---------------|------------|-------------|--------------|
| has_metadata | Create Metadata | "Generate Metadata" | /jobs?source={id}&job=metadata |
| has_vectors_1536 | Create Online Index | "Create Online Index" | /jobs?source={id}&job=index_online |
| has_vectors_768 | Create Offline Index | "Create Offline Index" | /jobs?source={id}&job=index_offline |
| license_in_allowlist | Edit Source | "Set License" | /sources/tools?source={id}#license |
| license_verified | Edit Source | "Verify License" | /sources/tools?source={id}#license |
| links_verified_offline | Edit Source | "Verify Offline Links" | /sources/tools?source={id}#links |
| links_verified_online | Edit Source | "Verify Online Links" | /sources/tools?source={id}#links |

---

## ValidationResult Dataclass

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class ValidationResult:
    """Result of validation checks."""
    # Gate results
    can_submit: bool = False
    can_publish: bool = False

    # File checks
    has_manifest: bool = False
    has_metadata: bool = False
    has_backup: bool = False
    has_vectors_1536: bool = False
    has_vectors_768: bool = False

    # License
    license: str = "Unknown"
    license_in_allowlist: bool = False
    license_verified: bool = False
    license_notes: str = ""

    # Human verification
    links_verified_offline: bool = False
    links_verified_online: bool = False

    # Counts
    document_count: int = 0
    vector_count_1536: int = 0
    vector_count_768: int = 0
    backup_size_mb: float = 0.0

    # Errors and warnings
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Deep validation
    deep_validated: bool = False
    integrity_passed: bool = False
```

---

## Workflow Diagram

```
LOCAL ADMIN PATH                         GLOBAL ADMIN PATH
================                         =================

[Create Source]                          [Receive Submission]
      |                                        |
      v                                        v
[Generate Backup]                        [Check which dimension exists]
[Generate Metadata]                            |
      |                                        v
      v                                  [Generate MISSING dimension]
[Generate ONE index]                     (saves 50% of work)
(1536 via OpenAI OR                            |
 768 via local model)                          v
      |                                  [Deep Validation]
      v                                  - Vector integrity scan
[Set human verification flags]           - ID cross-reference
- license_verified                       - Backup content verify
- links_verified_offline                       |
- links_verified_online                        v
      |                                  can_publish = TRUE
      v                                        |
[Light Validation]                             v
      |                                  [Publish to R2 + Pinecone]
      v
can_submit = TRUE
      |
      v
[Submit to Queue] ---------------------->
```

---

## Related Documentation

- [Source Tools](source-tools.md) - Source creation pipeline
- [Jobs System](jobs.md) - Background job processing
- [Admin Guide](admin-guide.md) - Global admin review process

---

*Last Updated: December 2025*
