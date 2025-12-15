"""
Unified Validation Module for Disaster Clippy

This module provides:
- ValidationResult dataclass with all check fields
- validate_light() - fast checks for UI badges (<100ms)
- validate_deep() - thorough checks for publishing (5-30s)
- can_submit() - local admin gate (needs 1 vector dimension)
- can_publish() - global admin gate (needs both dimensions + deep validation)
- Caching with _validation_status.json per source
- FIX_ACTIONS mapping for UI guidance

Two validation tiers:
- Light: File existence, metadata scans, header checks (for UI)
- Deep: Full content validation, vector integrity, ID cross-refs (for publishing)

Two permission gates:
- can_submit: Local admin can submit for review
- can_publish: Global admin can publish to production
"""

import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# =============================================================================
# CONSTANTS
# =============================================================================

# Allowed licenses for production sources
ALLOWED_LICENSES = [
    "CC0",           # Public Domain Dedication
    "CC-BY",         # Creative Commons Attribution
    "CC-BY-SA",      # Creative Commons Attribution ShareAlike
    "CC-BY-NC",      # Creative Commons Attribution NonCommercial
    "CC-BY-NC-SA",   # Creative Commons Attribution NonCommercial ShareAlike
    "Public Domain", # Explicit public domain
    "MIT",           # MIT License
    "Apache-2.0",    # Apache License 2.0
    "GPL",           # GNU General Public License (any version)
    "GFDL",          # GNU Free Documentation License
    "ODbL",          # Open Database License
    "Custom",        # Requires license_notes >= 20 chars
]

# Minimum backup size in MB (files smaller than this are likely empty/corrupt)
MIN_BACKUP_SIZE_MB = 0.1

# Minimum license notes length when using Custom license
MIN_LICENSE_NOTES_LENGTH = 20

# Cache file name (stored in each source folder)
VALIDATION_CACHE_FILE = "_validation_status.json"

# Cache TTL in seconds (5 minutes for light validation)
CACHE_TTL_SECONDS = 300

# =============================================================================
# VALIDATION RESULT DATACLASS
# =============================================================================

@dataclass
class ValidationResult:
    """
    Complete validation status for a source.

    Contains all checks needed for can_submit and can_publish gates.
    """
    # Identity
    source_id: str = ""
    validated_at: str = ""
    validation_tier: str = "light"  # "light" or "deep"

    # File existence checks
    has_manifest: bool = False
    has_metadata: bool = False
    has_backup: bool = False
    has_config: bool = False  # Alias for has_manifest (backwards compat)

    # Vector checks
    has_vectors_1536: bool = False
    has_vectors_768: bool = False
    has_vectors_384: bool = False
    has_vectors_1024: bool = False

    # Size checks
    backup_size_mb: float = 0.0
    backup_size_bytes: int = 0
    document_count: int = 0
    vector_count_1536: int = 0
    vector_count_768: int = 0

    # Backup type
    backup_type: str = ""  # "html", "zim", "pdf", or ""

    # License checks
    license: str = "Unknown"
    license_in_allowlist: bool = False
    license_verified: bool = False
    license_notes: str = ""

    # Link verification (human flags)
    links_verified_offline: bool = False
    links_verified_online: bool = False

    # Language checks
    language: str = ""  # "en", "es", etc.
    language_is_english: bool = False
    language_verified: bool = False
    language_confidence: float = 0.0

    # Deep validation (global admin only)
    deep_validated: bool = False
    integrity_passed: bool = False
    vector_integrity_1536: bool = False
    vector_integrity_768: bool = False
    id_crossref_passed: bool = False

    # Computed gates
    can_submit: bool = False
    can_publish: bool = False

    # Issues and warnings
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing: List[Dict[str, Any]] = field(default_factory=list)

    # Cache info
    cached: bool = False
    cache_valid_until: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ValidationResult":
        """Create from dictionary."""
        # Handle missing fields gracefully
        result = cls()
        for key, value in data.items():
            if hasattr(result, key):
                setattr(result, key, value)
        return result


# =============================================================================
# FIX ACTIONS MAPPING
# =============================================================================

FIX_ACTIONS = {
    "has_backup": {
        "job": "scrape",
        "label": "Scrape Content",
        "url": "/useradmin/jobs?add=scrape&source={source_id}"
    },
    "has_manifest": {
        "job": "metadata",
        "label": "Generate Manifest",
        "url": "/useradmin/jobs?add=metadata&source={source_id}"
    },
    "has_metadata": {
        "job": "metadata",
        "label": "Generate Metadata",
        "url": "/useradmin/jobs?add=metadata&source={source_id}"
    },
    "has_vectors_1536": {
        "job": "index_online",
        "label": "Create Online Index",
        "url": "/useradmin/jobs?add=index_online&source={source_id}"
    },
    "has_vectors_768": {
        "job": "index_offline",
        "label": "Create Offline Index",
        "url": "/useradmin/jobs?add=index_offline&source={source_id}"
    },
    "license_in_allowlist": {
        "job": None,
        "label": "Set License",
        "url": "/useradmin/sources/tools?source={source_id}#license"
    },
    "license_verified": {
        "job": None,
        "label": "Verify License",
        "url": "/useradmin/sources/tools?source={source_id}#license"
    },
    "links_verified_offline": {
        "job": "verify_links_offline",
        "label": "Verify Offline Links",
        "url": "/useradmin/sources/tools?source={source_id}#links"
    },
    "links_verified_online": {
        "job": None,
        "label": "Verify Online Links",
        "url": "/useradmin/sources/tools?source={source_id}#links"
    },
    "language_is_english": {
        "job": "detect_language",
        "label": "Detect Language",
        "url": "/useradmin/jobs?add=detect_language&source={source_id}"
    },
    "language_verified": {
        "job": None,
        "label": "Verify Language",
        "url": "/useradmin/sources/tools?source={source_id}#language"
    },
}


def get_fix_action(check_name: str, source_id: str) -> Optional[Dict[str, Any]]:
    """Get the fix action for a failed validation check."""
    action = FIX_ACTIONS.get(check_name)
    if action:
        return {
            "check": check_name,
            "job": action.get("job"),
            "label": action.get("label"),
            "url": action.get("url", "").format(source_id=source_id)
        }
    return None


# =============================================================================
# LIGHT VALIDATION (Fast - for UI)
# =============================================================================

def validate_light(source_path: str, source_id: str, use_cache: bool = True) -> ValidationResult:
    """
    Fast validation for UI display (<100ms).

    Checks:
    - File existence (manifest, metadata, vectors, backup)
    - Manifest field values (license, language, verification flags)
    - Basic size checks (backup > 0.1 MB)

    Does NOT check:
    - Vector content integrity
    - ID cross-references
    - Actual language detection from content

    Args:
        source_path: Path to source folder
        source_id: Source identifier
        use_cache: Whether to use cached results

    Returns:
        ValidationResult with all checks populated
    """
    path = Path(source_path)
    result = ValidationResult(
        source_id=source_id,
        validated_at=datetime.utcnow().isoformat() + "Z",
        validation_tier="light"
    )

    # Try cache first
    if use_cache:
        cached = _load_cache(path)
        if cached and cached.validation_tier == "light":
            cached.cached = True
            # Recompute gates (in case logic changed)
            cached.can_submit = _compute_can_submit(cached)
            cached.can_publish = _compute_can_publish(cached)
            return cached

    # Import schema helpers
    try:
        from offline_tools.schemas import (
            get_manifest_file, get_metadata_file,
            get_vectors_1536_file, get_vectors_768_file,
            get_vectors_384_file, get_vectors_1024_file
        )
    except ImportError:
        # Fallback if import fails
        get_manifest_file = lambda: "_manifest.json"
        get_metadata_file = lambda: "_metadata.json"
        get_vectors_1536_file = lambda: "_vectors.json"
        get_vectors_768_file = lambda: "_vectors_768.json"
        get_vectors_384_file = lambda: "_vectors_384.json"
        get_vectors_1024_file = lambda: "_vectors_1024.json"

    # Check file existence
    result.has_manifest = (path / get_manifest_file()).exists()
    result.has_config = result.has_manifest  # Alias
    result.has_metadata = (path / get_metadata_file()).exists()
    result.has_vectors_1536 = (path / get_vectors_1536_file()).exists()
    result.has_vectors_768 = (path / get_vectors_768_file()).exists()
    result.has_vectors_384 = (path / get_vectors_384_file()).exists()
    result.has_vectors_1024 = (path / get_vectors_1024_file()).exists()

    # Check backup
    result.has_backup, result.backup_type, result.backup_size_bytes = _check_backup(path, source_id)
    result.backup_size_mb = result.backup_size_bytes / (1024 * 1024)

    # Load manifest for metadata
    if result.has_manifest:
        manifest_data = _load_json(path / get_manifest_file())
        if manifest_data:
            result.license = manifest_data.get("license", "Unknown")
            result.license_verified = manifest_data.get("license_verified", False)
            result.license_notes = manifest_data.get("license_notes", "")
            result.links_verified_offline = manifest_data.get("links_verified_offline", False)
            result.links_verified_online = manifest_data.get("links_verified_online", False)
            result.language = manifest_data.get("language", "")
            result.language_verified = manifest_data.get("language_verified", False)
            result.document_count = manifest_data.get("total_docs", 0)

    # Check license against allowlist
    result.license_in_allowlist = _check_license_allowlist(result.license, result.license_notes)

    # Check language
    result.language_is_english = result.language.lower() in ["en", "english", "eng"]

    # Get vector counts (light check - just file parsing, not integrity)
    if result.has_vectors_1536:
        result.vector_count_1536 = _count_vectors_light(path / get_vectors_1536_file())
    if result.has_vectors_768:
        result.vector_count_768 = _count_vectors_light(path / get_vectors_768_file())

    # Compute gates and missing items
    result.can_submit = _compute_can_submit(result)
    result.can_publish = _compute_can_publish(result)
    result.missing = _compute_missing(result)

    # Cache the result locally (master.json is rebuilt at server startup)
    _save_cache(path, result)

    return result


# =============================================================================
# DEEP VALIDATION (Thorough - for publishing)
# =============================================================================

def validate_deep(source_path: str, source_id: str) -> ValidationResult:
    """
    Thorough validation for publishing (5-30 seconds).

    Runs all light checks PLUS:
    - Vector integrity (NaN, Infinity, null, wrong dimensions)
    - ID cross-reference (metadata IDs match vector IDs)
    - Backup content validation (HTML structure, ZIM magic bytes)
    - Language detection from actual content (if not already set)

    Args:
        source_path: Path to source folder
        source_id: Source identifier

    Returns:
        ValidationResult with deep validation flags
    """
    # Start with light validation
    result = validate_light(source_path, source_id, use_cache=False)
    result.validation_tier = "deep"
    result.validated_at = datetime.utcnow().isoformat() + "Z"

    path = Path(source_path)

    # Import schema helpers
    try:
        from offline_tools.schemas import get_vectors_1536_file, get_vectors_768_file
    except ImportError:
        get_vectors_1536_file = lambda: "_vectors.json"
        get_vectors_768_file = lambda: "_vectors_768.json"

    # Deep vector integrity checks
    if result.has_vectors_1536:
        result.vector_integrity_1536, issues = _validate_vectors_deep(
            path / get_vectors_1536_file(), expected_dim=1536
        )
        result.issues.extend(issues)
    else:
        result.vector_integrity_1536 = True  # Not required to pass if not present

    if result.has_vectors_768:
        result.vector_integrity_768, issues = _validate_vectors_deep(
            path / get_vectors_768_file(), expected_dim=768
        )
        result.issues.extend(issues)
    else:
        result.vector_integrity_768 = True  # Not required to pass if not present

    # ID cross-reference check
    if result.has_metadata and (result.has_vectors_1536 or result.has_vectors_768):
        result.id_crossref_passed, issues = _validate_id_crossref(path, source_id)
        result.issues.extend(issues)
    else:
        result.id_crossref_passed = True  # Can't check without both

    # Backup content validation
    if result.has_backup:
        backup_valid, issues = _validate_backup_content(path, source_id, result.backup_type)
        if not backup_valid:
            result.issues.extend(issues)

    # Language detection from content (if not set)
    if not result.language or not result.language_is_english:
        detected_lang, confidence = _detect_language_from_content(path, result.backup_type)
        if detected_lang:
            result.language = detected_lang
            result.language_confidence = confidence
            result.language_is_english = detected_lang.lower() in ["en", "english", "eng"]

    # Compute deep validation status
    result.integrity_passed = (
        result.vector_integrity_1536 and
        result.vector_integrity_768 and
        result.id_crossref_passed
    )
    result.deep_validated = result.integrity_passed and len(result.issues) == 0

    # Recompute gates
    result.can_submit = _compute_can_submit(result)
    result.can_publish = _compute_can_publish(result)
    result.missing = _compute_missing(result)

    # Cache deep result locally (master.json is rebuilt at server startup)
    _save_cache(path, result)

    return result


# =============================================================================
# PERMISSION GATES
# =============================================================================

def _compute_can_submit(r: ValidationResult) -> bool:
    """
    Local admin gate - can submit source for review.

    Requires:
    - has_manifest
    - has_metadata
    - has_backup with size >= 0.1 MB
    - At least ONE vector dimension (768 OR 1536)
    - license_in_allowlist
    - license_verified
    - links_verified_offline
    - links_verified_online
    - language_is_english
    """
    has_at_least_one_index = r.has_vectors_1536 or r.has_vectors_768

    return (
        r.has_manifest and
        r.has_metadata and
        r.has_backup and
        r.backup_size_mb >= MIN_BACKUP_SIZE_MB and
        has_at_least_one_index and
        r.license_in_allowlist and
        r.license_verified and
        r.links_verified_offline and
        r.links_verified_online and
        r.language_is_english
    )


def _compute_can_publish(r: ValidationResult) -> bool:
    """
    Global admin gate - can publish to production.

    Requires:
    - All can_submit requirements
    - BOTH vector dimensions (768 AND 1536)
    - deep_validated = True
    - integrity_passed = True
    - language_verified = True
    """
    return (
        _compute_can_submit(r) and
        r.has_vectors_768 and
        r.has_vectors_1536 and
        r.deep_validated and
        r.integrity_passed and
        r.language_verified
    )


def can_submit(result: ValidationResult) -> bool:
    """Check if source can be submitted by local admin."""
    return _compute_can_submit(result)


def can_publish(result: ValidationResult) -> bool:
    """Check if source can be published by global admin."""
    return _compute_can_publish(result)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _check_backup(path: Path, source_id: str) -> Tuple[bool, str, int]:
    """
    Check for backup content and determine type.

    Uses fast sampling - does not traverse all files.

    Returns:
        (has_backup, backup_type, size_bytes)
    """
    total_size = 0
    backup_type = ""

    # Check for ZIM file first (single file = fast)
    zim_file = path / f"{source_id}.zim"
    if not zim_file.exists():
        # Check for any ZIM in source folder
        zim_files = list(path.glob("*.zim"))[:1]
        if zim_files:
            zim_file = zim_files[0]

    if zim_file.exists():
        total_size = zim_file.stat().st_size
        backup_type = "zim"
        return True, backup_type, total_size

    # Check pages/ folder (HTML backup) - fast sampling only
    pages_folder = path / "pages"
    if pages_folder.exists() and pages_folder.is_dir():
        # Sample first few files for size estimate (don't traverse all)
        sample_size = 0
        file_count = 0
        for item in pages_folder.iterdir():
            if file_count >= 10:
                break
            if item.is_file():
                sample_size += item.stat().st_size
                file_count += 1
            elif item.is_dir():
                # Sample one file from subdirectory
                for sub_item in list(item.iterdir())[:1]:
                    if sub_item.is_file():
                        sample_size += sub_item.stat().st_size
                        file_count += 1
        if file_count > 0:
            backup_type = "html"
            return True, backup_type, sample_size

    # Check for PDF files
    pdfs = list(path.glob("*.pdf"))[:10]  # Limit to first 10
    if pdfs:
        for pdf in pdfs:
            total_size += pdf.stat().st_size
        backup_type = "pdf"
        return True, backup_type, total_size

    return False, "", 0


def _check_license_allowlist(license_str: str, license_notes: str) -> bool:
    """Check if license is in allowlist (flexible matching)."""
    if not license_str:
        return False

    import re

    # Normalize license string - lowercase, strip whitespace
    normalized = license_str.strip().lower()

    # Remove version numbers like "3.0", "4.0", "2.0"
    normalized_no_version = re.sub(r'\s*\d+(\.\d+)?\s*$', '', normalized)

    # Normalize CC license format: "cc by-sa" -> "cc-by-sa", "cc by sa" -> "cc-by-sa"
    normalized_cc = normalized_no_version.replace(' ', '-').replace('--', '-')

    # Check against allowlist
    for allowed in ALLOWED_LICENSES:
        allowed_lower = allowed.lower()

        # Exact match
        if normalized == allowed_lower:
            if allowed == "Custom":
                return len(license_notes.strip()) >= MIN_LICENSE_NOTES_LENGTH
            return True

        # Match without version
        if normalized_no_version == allowed_lower:
            if allowed == "Custom":
                return len(license_notes.strip()) >= MIN_LICENSE_NOTES_LENGTH
            return True

        # Match normalized CC format (handles "CC BY-SA 4.0" matching "CC-BY-SA")
        if normalized_cc == allowed_lower:
            if allowed == "Custom":
                return len(license_notes.strip()) >= MIN_LICENSE_NOTES_LENGTH
            return True

        # Partial match for CC licenses (e.g., "cc-by-sa" in "cc-by-sa-4.0")
        if allowed_lower.startswith('cc') and allowed_lower in normalized_cc:
            if allowed == "Custom":
                return len(license_notes.strip()) >= MIN_LICENSE_NOTES_LENGTH
            return True

    return False


def _load_json(filepath: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file safely."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _count_vectors_light(filepath: Path) -> int:
    """Count vectors without loading full content (light check)."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "vectors" in data:
            return len(data["vectors"])
        elif isinstance(data, list):
            return len(data)
    except Exception:
        pass
    return 0


def _validate_vectors_deep(filepath: Path, expected_dim: int) -> Tuple[bool, List[str]]:
    """
    Deep validation of vector file.

    Checks:
    - All vectors have correct dimension
    - No NaN values
    - No Infinity values
    - No null/None values
    - No all-zero vectors

    Returns:
        (passed, list of issues)
    """
    issues = []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        vectors = []
        if isinstance(data, dict) and "vectors" in data:
            vectors = data["vectors"]
        elif isinstance(data, list):
            vectors = data

        if not vectors:
            issues.append(f"No vectors found in {filepath.name}")
            return False, issues

        corrupt_count = 0
        wrong_dim_count = 0
        zero_count = 0

        for i, entry in enumerate(vectors):
            vec = None
            if isinstance(entry, dict):
                vec = entry.get("vector") or entry.get("embedding") or entry.get("values")
            elif isinstance(entry, list):
                vec = entry

            if vec is None:
                corrupt_count += 1
                continue

            # Check dimension
            if len(vec) != expected_dim:
                wrong_dim_count += 1
                continue

            # Check for bad values
            has_bad = False
            all_zero = True
            for v in vec:
                if v is None:
                    has_bad = True
                    break
                if isinstance(v, float):
                    if math.isnan(v) or math.isinf(v):
                        has_bad = True
                        break
                    if v != 0:
                        all_zero = False
                elif v != 0:
                    all_zero = False

            if has_bad:
                corrupt_count += 1
            elif all_zero:
                zero_count += 1

        if corrupt_count > 0:
            issues.append(f"{corrupt_count} vectors have corrupt values (NaN/Inf/null)")
        if wrong_dim_count > 0:
            issues.append(f"{wrong_dim_count} vectors have wrong dimension (expected {expected_dim})")
        if zero_count > 0:
            issues.append(f"{zero_count} vectors are all zeros (likely failed embeddings)")

        return len(issues) == 0, issues

    except Exception as e:
        issues.append(f"Failed to validate {filepath.name}: {str(e)}")
        return False, issues


def _validate_id_crossref(path: Path, source_id: str) -> Tuple[bool, List[str]]:
    """
    Validate that metadata IDs match vector IDs.

    Returns:
        (passed, list of issues)
    """
    issues = []

    try:
        from offline_tools.schemas import get_metadata_file, get_vectors_1536_file, get_vectors_768_file
    except ImportError:
        get_metadata_file = lambda: "_metadata.json"
        get_vectors_1536_file = lambda: "_vectors.json"
        get_vectors_768_file = lambda: "_vectors_768.json"

    # Load metadata IDs
    metadata_path = path / get_metadata_file()
    metadata_ids = set()
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "documents" in data:
            for doc in data["documents"]:
                if isinstance(doc, dict) and "doc_id" in doc:
                    metadata_ids.add(doc["doc_id"])
        elif isinstance(data, list):
            for doc in data:
                if isinstance(doc, dict) and "doc_id" in doc:
                    metadata_ids.add(doc["doc_id"])
    except Exception as e:
        issues.append(f"Failed to load metadata: {str(e)}")
        return False, issues

    # Load vector IDs
    def get_vector_ids(filepath: Path) -> set:
        ids = set()
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            vectors = []
            if isinstance(data, dict) and "vectors" in data:
                vectors = data["vectors"]
            elif isinstance(data, list):
                vectors = data
            for v in vectors:
                if isinstance(v, dict):
                    vid = v.get("id") or v.get("doc_id")
                    if vid:
                        ids.add(vid)
        except Exception:
            pass
        return ids

    # Check 1536
    vec_1536_path = path / get_vectors_1536_file()
    if vec_1536_path.exists():
        vec_ids = get_vector_ids(vec_1536_path)
        missing = metadata_ids - vec_ids
        if missing:
            issues.append(f"{len(missing)} metadata docs missing from 1536 vectors")

    # Check 768
    vec_768_path = path / get_vectors_768_file()
    if vec_768_path.exists():
        vec_ids = get_vector_ids(vec_768_path)
        missing = metadata_ids - vec_ids
        if missing:
            issues.append(f"{len(missing)} metadata docs missing from 768 vectors")

    return len(issues) == 0, issues


def _validate_backup_content(path: Path, source_id: str, backup_type: str) -> Tuple[bool, List[str]]:
    """
    Validate backup content structure.

    Returns:
        (passed, list of issues)
    """
    issues = []

    if backup_type == "html":
        pages_folder = path / "pages"
        if not pages_folder.exists():
            issues.append("pages/ folder not found")
            return False, issues

        # Check for at least some HTML files
        html_files = list(pages_folder.rglob("*.html")) + list(pages_folder.rglob("*.htm"))
        if not html_files:
            issues.append("No HTML files found in pages/")

    elif backup_type == "zim":
        # Check ZIM magic bytes
        parent = path.parent
        zim_file = parent / f"{source_id}.zim"
        if not zim_file.exists():
            zim_files = list(path.glob("*.zim"))
            if zim_files:
                zim_file = zim_files[0]
            else:
                issues.append("ZIM file not found")
                return False, issues

        try:
            with open(zim_file, "rb") as f:
                magic = f.read(4)
            # ZIM magic bytes: 0x5A 0x49 0x4D 0x04 (ZIM + version 4)
            if magic[:3] != b"ZIM":
                issues.append("Invalid ZIM file (wrong magic bytes)")
        except Exception as e:
            issues.append(f"Failed to read ZIM file: {str(e)}")

    elif backup_type == "pdf":
        pdfs = list(path.glob("*.pdf"))
        if not pdfs:
            issues.append("No PDF files found")
            return False, issues

        # Check PDF header
        for pdf in pdfs[:1]:  # Check first PDF only
            try:
                with open(pdf, "rb") as f:
                    header = f.read(5)
                if not header.startswith(b"%PDF"):
                    issues.append(f"Invalid PDF file: {pdf.name}")
            except Exception as e:
                issues.append(f"Failed to read PDF: {str(e)}")

    return len(issues) == 0, issues


def _detect_language_from_content(path: Path, backup_type: str) -> Tuple[str, float]:
    """
    Detect language from backup content.

    Returns:
        (language_code, confidence)
    """
    try:
        from langdetect import detect_langs
    except ImportError:
        # langdetect not available
        return "", 0.0

    # Sample some text content
    sample_text = ""

    if backup_type == "html":
        pages_folder = path / "pages"
        if pages_folder.exists():
            # Get a few HTML files
            html_files = list(pages_folder.rglob("*.html"))[:5]
            for html_file in html_files:
                try:
                    with open(html_file, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    # Extract text between tags (simple approach)
                    import re
                    text = re.sub(r'<[^>]+>', ' ', content)
                    text = re.sub(r'\s+', ' ', text)
                    sample_text += text[:1000] + " "
                except Exception:
                    continue

    if not sample_text or len(sample_text) < 50:
        return "", 0.0

    try:
        langs = detect_langs(sample_text)
        if langs:
            top_lang = langs[0]
            return top_lang.lang, top_lang.prob
    except Exception:
        pass

    return "", 0.0


def _compute_missing(result: ValidationResult) -> List[Dict[str, Any]]:
    """Compute list of missing items with fix actions."""
    missing = []

    checks = [
        ("has_manifest", result.has_manifest, "Missing manifest file"),
        ("has_metadata", result.has_metadata, "Missing metadata file"),
        ("has_backup", result.has_backup, "Missing backup content"),
        ("has_vectors_1536", result.has_vectors_1536, "Missing 1536-dim vectors (online index)"),
        ("has_vectors_768", result.has_vectors_768, "Missing 768-dim vectors (offline index)"),
        ("license_in_allowlist", result.license_in_allowlist, f"License '{result.license}' not in allowlist"),
        ("license_verified", result.license_verified, "License not verified by human"),
        ("links_verified_offline", result.links_verified_offline, "Offline links not verified"),
        ("links_verified_online", result.links_verified_online, "Online links not verified"),
        ("language_is_english", result.language_is_english, f"Language is '{result.language}' (must be English)"),
        ("language_verified", result.language_verified, "Language not verified by human"),
    ]

    # Add size check
    if result.has_backup and result.backup_size_mb < MIN_BACKUP_SIZE_MB:
        missing.append({
            "check": "backup_size",
            "message": f"Backup too small ({result.backup_size_mb:.2f} MB, need >= {MIN_BACKUP_SIZE_MB} MB)",
            "fix_action": None,
            "fix_label": None,
            "fix_url": None
        })

    for check_name, passed, message in checks:
        if not passed:
            action = get_fix_action(check_name, result.source_id)
            missing.append({
                "check": check_name,
                "message": message,
                "fix_action": action.get("job") if action else None,
                "fix_label": action.get("label") if action else None,
                "fix_url": action.get("url") if action else None
            })

    return missing


# =============================================================================
# CACHE MANAGEMENT
# =============================================================================

def _load_cache(path: Path) -> Optional[ValidationResult]:
    """Load cached validation result if still valid."""
    cache_file = path / VALIDATION_CACHE_FILE
    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Check if cache is still valid
        cache_time = datetime.fromisoformat(data.get("validated_at", "").rstrip("Z"))
        age_seconds = (datetime.utcnow() - cache_time).total_seconds()

        if age_seconds > CACHE_TTL_SECONDS:
            return None

        # Check if any source files changed since validation
        if not _cache_still_valid(path, cache_time):
            return None

        return ValidationResult.from_dict(data)
    except Exception:
        return None


def _save_cache(path: Path, result: ValidationResult):
    """Save validation result to cache."""
    cache_file = path / VALIDATION_CACHE_FILE
    try:
        result.cache_valid_until = (
            datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        )
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
    except Exception:
        pass  # Don't fail if cache can't be written


def _cache_still_valid(path: Path, cache_time: datetime) -> bool:
    """Check if any relevant files changed since cache was created."""
    try:
        from offline_tools.schemas import get_manifest_file, get_metadata_file
    except ImportError:
        get_manifest_file = lambda: "_manifest.json"
        get_metadata_file = lambda: "_metadata.json"

    files_to_check = [
        path / get_manifest_file(),
        path / get_metadata_file(),
        path / "_vectors.json",
        path / "_vectors_768.json",
    ]

    for filepath in files_to_check:
        if filepath.exists():
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            if mtime > cache_time:
                return False

    return True


def invalidate_cache(source_path: str):
    """Invalidate the validation cache for a source."""
    cache_file = Path(source_path) / VALIDATION_CACHE_FILE
    if cache_file.exists():
        try:
            cache_file.unlink()
        except Exception:
            pass


# =============================================================================
# MASTER.JSON - Centralized validation summary for all sources
# =============================================================================

MASTER_FILE = "_master.json"
MASTER_VERSION = 2  # Increment when schema changes


def update_master_validation(backup_folder: str, source_id: str, result: ValidationResult):
    """
    Update the master.json file with validation summary for a source.

    Called after validation completes to keep master.json in sync.
    This enables fast dashboard loading without per-source file reads.
    """
    master_path = Path(backup_folder) / MASTER_FILE

    # Load existing master or create new
    master_data = {"sources": {}, "version": MASTER_VERSION, "last_updated": ""}
    if master_path.exists():
        try:
            with open(master_path, "r", encoding="utf-8") as f:
                master_data = json.load(f)
            # Ensure sources dict exists
            if "sources" not in master_data:
                master_data["sources"] = {}
        except Exception:
            master_data = {"sources": {}, "version": MASTER_VERSION, "last_updated": ""}

    # Create summary for this source
    summary = {
        "name": result.source_id,  # Will be overwritten if manifest has name
        "validated_at": result.validated_at,
        "validation_tier": result.validation_tier,
        "can_submit": result.can_submit,
        "can_publish": result.can_publish,
        "has_manifest": result.has_manifest,
        "has_metadata": result.has_metadata,
        "has_backup": result.has_backup,
        "has_vectors_1536": result.has_vectors_1536,
        "has_vectors_768": result.has_vectors_768,
        "backup_type": result.backup_type,
        "backup_size_mb": round(result.backup_size_mb, 2),
        "document_count": result.document_count,
        "license": result.license,
        "license_in_allowlist": result.license_in_allowlist,
        "license_verified": result.license_verified,
        "license_notes": result.license_notes,
        "links_verified_offline": result.links_verified_offline,
        "links_verified_online": result.links_verified_online,
        "language": result.language,
        "language_is_english": result.language_is_english,
        "language_verified": result.language_verified,
        "vector_count_1536": result.vector_count_1536,
        "vector_count_768": result.vector_count_768,
        "deep_validated": result.deep_validated,
        "integrity_passed": result.integrity_passed,
        "missing_count": len(result.missing),
        "issues_count": len(result.issues),
    }

    # Try to get name from manifest
    try:
        from offline_tools.schemas import get_manifest_file
        manifest_path = Path(backup_folder) / source_id / get_manifest_file()
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            summary["name"] = manifest.get("name", source_id)
            summary["description"] = manifest.get("description", "")
            summary["base_url"] = manifest.get("base_url", "")
            summary["tags"] = manifest.get("tags", [])
    except Exception:
        pass

    # Update source in master
    master_data["sources"][source_id] = summary
    master_data["last_updated"] = datetime.utcnow().isoformat() + "Z"
    master_data["version"] = MASTER_VERSION

    # Write master file
    try:
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(master_data, f, indent=2)
    except Exception as e:
        print(f"Failed to update master.json: {e}")


def get_master_validation(backup_folder: str) -> Dict[str, Any]:
    """
    Read the master.json file for fast dashboard loading.

    Returns dict with 'sources' containing validation summaries for all sources.
    """
    master_path = Path(backup_folder) / MASTER_FILE

    if not master_path.exists():
        return {"sources": {}, "version": MASTER_VERSION, "last_updated": ""}

    try:
        with open(master_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"sources": {}, "version": MASTER_VERSION, "last_updated": ""}


def remove_from_master(backup_folder: str, source_id: str):
    """Remove a source from master.json (when source is deleted)."""
    master_path = Path(backup_folder) / MASTER_FILE

    if not master_path.exists():
        return

    try:
        with open(master_path, "r", encoding="utf-8") as f:
            master_data = json.load(f)

        if source_id in master_data.get("sources", {}):
            del master_data["sources"][source_id]
            master_data["last_updated"] = datetime.utcnow().isoformat() + "Z"

            with open(master_path, "w", encoding="utf-8") as f:
                json.dump(master_data, f, indent=2)
    except Exception:
        pass


# System folders that should not be treated as sources
SYSTEM_FOLDERS = {
    "chroma_db_1536",
    "chroma_db_768",
    "models",
    "translations",
    "visualisation",
    "visualisations",
    "cache",
    "temp",
    "tmp",
}


def rebuild_master_from_sources(backup_folder: str) -> Dict[str, Any]:
    """
    Rebuild _master.json by reading all per-source validation files.

    Call this at server startup to aggregate validation data.
    Each source's _validation_status.json is the source of truth.

    Returns the rebuilt master data.
    """
    backup_path = Path(backup_folder)
    master_data = {"sources": {}, "version": MASTER_VERSION, "last_updated": ""}

    if not backup_path.exists():
        return master_data

    # Scan all source folders
    for source_folder in backup_path.iterdir():
        # Skip non-directories, underscore-prefixed, and system folders
        if not source_folder.is_dir():
            continue
        if source_folder.name.startswith("_"):
            continue
        if source_folder.name.lower() in SYSTEM_FOLDERS:
            continue

        source_id = source_folder.name
        cache_file = source_folder / VALIDATION_CACHE_FILE

        # Read per-source validation cache
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)

                # Create summary from cache
                summary = {
                    "name": cache_data.get("source_id", source_id),
                    "validated_at": cache_data.get("validated_at", ""),
                    "validation_tier": cache_data.get("validation_tier", "light"),
                    "can_submit": cache_data.get("can_submit", False),
                    "can_publish": cache_data.get("can_publish", False),
                    "has_manifest": cache_data.get("has_manifest", False),
                    "has_metadata": cache_data.get("has_metadata", False),
                    "has_backup": cache_data.get("has_backup", False),
                    "has_vectors_1536": cache_data.get("has_vectors_1536", False),
                    "has_vectors_768": cache_data.get("has_vectors_768", False),
                    "backup_type": cache_data.get("backup_type", ""),
                    "backup_size_mb": cache_data.get("backup_size_mb", 0),
                    "document_count": cache_data.get("document_count", 0),
                    "license": cache_data.get("license", "Unknown"),
                    "license_in_allowlist": cache_data.get("license_in_allowlist", False),
                    "license_verified": cache_data.get("license_verified", False),
                    "license_notes": cache_data.get("license_notes", ""),
                    "links_verified_offline": cache_data.get("links_verified_offline", False),
                    "links_verified_online": cache_data.get("links_verified_online", False),
                    "language": cache_data.get("language", ""),
                    "language_is_english": cache_data.get("language_is_english", False),
                    "language_verified": cache_data.get("language_verified", False),
                    "vector_count_1536": cache_data.get("vector_count_1536", 0),
                    "vector_count_768": cache_data.get("vector_count_768", 0),
                    "deep_validated": cache_data.get("deep_validated", False),
                    "integrity_passed": cache_data.get("integrity_passed", False),
                }

                # Try to get name/description from manifest
                try:
                    from offline_tools.schemas import get_manifest_file
                    manifest_path = source_folder / get_manifest_file()
                    if manifest_path.exists():
                        with open(manifest_path, "r", encoding="utf-8") as f:
                            manifest = json.load(f)
                        summary["name"] = manifest.get("name", source_id)
                        summary["description"] = manifest.get("description", "")
                        summary["base_url"] = manifest.get("base_url", "")
                        summary["tags"] = manifest.get("tags", [])
                except Exception:
                    pass

                master_data["sources"][source_id] = summary

            except Exception as e:
                print(f"Error reading validation cache for {source_id}: {e}")
                # Source exists but no valid cache - mark as needing validation
                master_data["sources"][source_id] = {
                    "name": source_id,
                    "needs_validation": True,
                    "validated_at": "",
                }
        else:
            # No validation cache - source needs validation
            # Still add basic info from manifest if available
            try:
                from offline_tools.schemas import get_manifest_file
                manifest_path = source_folder / get_manifest_file()
                if manifest_path.exists():
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    master_data["sources"][source_id] = {
                        "name": manifest.get("name", source_id),
                        "description": manifest.get("description", ""),
                        "base_url": manifest.get("base_url", ""),
                        "tags": manifest.get("tags", []),
                        "license": manifest.get("license", "Unknown"),
                        "needs_validation": True,
                        "validated_at": "",
                    }
                else:
                    master_data["sources"][source_id] = {
                        "name": source_id,
                        "needs_validation": True,
                        "validated_at": "",
                    }
            except Exception:
                master_data["sources"][source_id] = {
                    "name": source_id,
                    "needs_validation": True,
                    "validated_at": "",
                }

    # Update timestamp and write
    master_data["last_updated"] = datetime.utcnow().isoformat() + "Z"
    master_data["version"] = MASTER_VERSION

    try:
        master_path = backup_path / MASTER_FILE
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(master_data, f, indent=2)
    except Exception as e:
        print(f"Failed to write master.json: {e}")

    return master_data


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_source(source_path: str, source_id: str, deep: bool = False) -> ValidationResult:
    """
    Main validation entry point.

    Args:
        source_path: Path to source folder
        source_id: Source identifier
        deep: If True, run deep validation; otherwise light validation

    Returns:
        ValidationResult
    """
    if deep:
        return validate_deep(source_path, source_id)
    return validate_light(source_path, source_id)


def get_validation_summary(result: ValidationResult) -> Dict[str, Any]:
    """
    Get a summary suitable for API responses.

    Returns a simplified dict with key status info.
    """
    return {
        "source_id": result.source_id,
        "can_submit": result.can_submit,
        "can_publish": result.can_publish,
        "validation_tier": result.validation_tier,
        "has_manifest": result.has_manifest,
        "has_metadata": result.has_metadata,
        "has_backup": result.has_backup,
        "has_vectors_1536": result.has_vectors_1536,
        "has_vectors_768": result.has_vectors_768,
        "license": result.license,
        "license_verified": result.license_verified,
        "language": result.language,
        "language_is_english": result.language_is_english,
        "document_count": result.document_count,
        "backup_size_mb": round(result.backup_size_mb, 2),
        "missing_count": len(result.missing),
        "missing": result.missing,
        "issues": result.issues,
        "warnings": result.warnings,
    }
