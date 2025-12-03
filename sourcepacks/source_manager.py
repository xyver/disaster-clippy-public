"""
Unified Source Manager for creating and validating source packages.

Provides a single interface for:
- Creating backups (HTML, ZIM, PDF, Substack)
- Creating indexes
- Validating sources (license, tags, completeness)
- Preparing sources for distribution

Used by both Global Admin and Local Admin dashboards.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class BackupResult:
    """Result of a backup operation"""
    success: bool
    source_id: str
    backup_type: str  # html, zim, pdf, substack
    page_count: int = 0
    backup_path: str = ""
    error: str = ""
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass
class IndexResult:
    """Result of an indexing operation"""
    success: bool
    source_id: str
    indexed_count: int = 0
    skipped_count: int = 0
    total_chars: int = 0
    error: str = ""
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class ValidationResult:
    """Result of source validation"""
    is_valid: bool
    source_id: str
    has_backup: bool = False
    has_index: bool = False
    has_metadata: bool = False
    has_embeddings: bool = False  # Has exportable embeddings
    has_license: bool = False
    license_verified: bool = False
    has_tags: bool = False
    production_ready: bool = False  # Ready for R2/Pinecone upload
    issues: List[str] = None
    warnings: List[str] = None
    detected_license: str = ""
    suggested_tags: List[str] = None

    # New schema v2 fields
    schema_version: int = 1  # 1=legacy, 2=new layered format
    has_source_metadata: bool = False  # {source_id}_source.json
    has_documents_file: bool = False  # {source_id}_documents.json
    has_embeddings_file: bool = False  # {source_id}_embeddings.json
    has_categories: bool = False  # {source_id}_categories/
    legacy_files: List[str] = None  # Old format files that need migration
    redundant_files: List[str] = None  # Files that can be safely deleted (v2 exists)
    needs_migration: bool = False  # True if legacy files present
    has_cleanup_needed: bool = False  # True if redundant files can be deleted

    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.warnings is None:
            self.warnings = []
        if self.suggested_tags is None:
            self.suggested_tags = []
        if self.legacy_files is None:
            self.legacy_files = []
        if self.redundant_files is None:
            self.redundant_files = []


class SourceManager:
    """
    Unified interface for creating and managing source packages.

    Handles multiple source types (HTML, ZIM, PDF, Substack) with a consistent API.
    Works with BACKUP_PATH as the data directory.
    """

    # Known license patterns for auto-detection
    LICENSE_PATTERNS = {
        "CC-BY-SA": [
            r"creative\s*commons.*share\s*alike",
            r"cc[- ]by[- ]sa",
            r"attribution.*sharealike",
        ],
        "CC-BY": [
            r"creative\s*commons.*attribution(?!.*share)",
            r"cc[- ]by(?![- ]sa|[- ]nc)",
        ],
        "CC-BY-NC": [
            r"creative\s*commons.*non\s*commercial",
            r"cc[- ]by[- ]nc(?![- ]sa)",
        ],
        "CC-BY-NC-SA": [
            r"cc[- ]by[- ]nc[- ]sa",
            r"attribution.*noncommercial.*sharealike",
        ],
        "CC0": [
            r"cc0",
            r"public\s*domain\s*dedication",
            r"creative\s*commons\s*zero",
        ],
        "Public Domain": [
            r"public\s*domain",
            r"no\s*copyright",
            r"free\s*of\s*known\s*copyright",
        ],
        "MIT": [
            r"mit\s*license",
            r"permission\s*is\s*hereby\s*granted.*mit",
        ],
        "GPL": [
            r"gnu\s*general\s*public\s*license",
            r"gpl[- ]?\d",
        ],
        "Open Access": [
            r"open\s*access",
            r"freely\s*available",
        ],
    }

    # Common topic keywords for tag suggestions
    TOPIC_KEYWORDS = {
        "water": ["water", "filtration", "purification", "well", "pump", "irrigation"],
        "solar": ["solar", "photovoltaic", "pv", "sun", "renewable"],
        "energy": ["energy", "power", "electricity", "generator", "battery"],
        "food": ["food", "agriculture", "farming", "garden", "crop", "cooking"],
        "shelter": ["shelter", "housing", "building", "construction", "roof"],
        "health": ["health", "medical", "medicine", "first aid", "sanitation"],
        "emergency": ["emergency", "disaster", "survival", "preparedness", "crisis"],
        "diy": ["diy", "homemade", "build", "make", "construct"],
    }

    def __init__(self, backup_path: str = None):
        """
        Args:
            backup_path: Root path for all backup data. Defaults to local_config, then BACKUP_PATH env var.
        """
        if backup_path:
            self.backup_path = Path(backup_path)
        else:
            # Prioritize local_config (user's GUI setting) over env var
            self.backup_path = None
            try:
                from useradmin.local_config import get_local_config
                config = get_local_config()
                folder = config.get_backup_folder()
                if folder:
                    self.backup_path = Path(folder)
            except ImportError:
                pass

            # Fall back to BACKUP_PATH env var
            if not self.backup_path:
                self.backup_path = Path(os.getenv("BACKUP_PATH", ""))

        if not self.backup_path:
            self.backup_path = Path("data")

        self.backup_path.mkdir(parents=True, exist_ok=True)

    def get_source_path(self, source_id: str) -> Path:
        """Get the path for a source's data folder"""
        return self.backup_path / source_id

    def _force_close_zim_handles(self, zim_path: Path) -> None:
        """
        Attempt to close any open ZIM file handles.
        This helps when trying to rename/move ZIM files that might be open.
        """
        import gc

        # Try to close any ZIMFile instances that might have this path open
        try:
            from zimply_core.zim_core import ZIMFile
            # Force garbage collection to clean up any unreferenced ZIMFile objects
            gc.collect()

            # On Windows, we can't easily force-close file handles from other processes
            # But we can clean up handles in our own process
        except ImportError:
            pass

        # Force garbage collection again
        gc.collect()

    def normalize_filenames(self, source_id: str) -> Dict[str, Any]:
        """
        Normalize backup file names to standard format.

        Renames files like 'bitcoin_en_all_maxi_2021-03.zim' to 'bitcoin.zim'
        for consistency across the system.

        Returns:
            Dict with renamed files and any errors
        """
        source_path = self.get_source_path(source_id)
        renamed = []
        errors = []

        if not source_path.exists():
            return {"renamed": [], "errors": ["Source folder does not exist"]}

        # Normalize ZIM files
        zim_files = list(source_path.glob("*.zim"))
        standard_zim = source_path / f"{source_id}.zim"

        if zim_files and not standard_zim.exists():
            # There's a ZIM file but it's not named correctly
            if len(zim_files) == 1:
                old_path = zim_files[0]
                try:
                    # Try to force-close any open ZIM handles first
                    self._force_close_zim_handles(old_path)
                    old_path.rename(standard_zim)
                    renamed.append({
                        "old": old_path.name,
                        "new": standard_zim.name,
                        "type": "zim"
                    })
                except Exception as e:
                    errors.append(f"Failed to rename {old_path.name}: {e}. Try restarting the server.")
            else:
                # Multiple ZIM files - don't know which to use
                errors.append(f"Multiple ZIM files found: {[f.name for f in zim_files]}. Please keep only one.")

        # Also check root backup folder for ZIM
        root_zim_files = [f for f in self.backup_path.glob("*.zim")
                         if f.stem.startswith(source_id) and f.name != f"{source_id}.zim"]
        root_standard_zim = self.backup_path / f"{source_id}.zim"

        if root_zim_files and not root_standard_zim.exists():
            if len(root_zim_files) == 1:
                old_path = root_zim_files[0]
                try:
                    self._force_close_zim_handles(old_path)
                    old_path.rename(root_standard_zim)
                    renamed.append({
                        "old": old_path.name,
                        "new": root_standard_zim.name,
                        "type": "zim",
                        "location": "root"
                    })
                except Exception as e:
                    errors.append(f"Failed to rename {old_path.name}: {e}. Try restarting the server.")

        return {
            "renamed": renamed,
            "errors": errors,
            "source_id": source_id
        }

    def get_expected_files(self, source_id: str, source_type: str = None) -> List[str]:
        """
        Get list of expected files for a v2 schema source package.

        Args:
            source_id: Source identifier
            source_type: 'zim', 'html', or 'pdf' (auto-detected if not provided)

        Returns:
            List of expected filenames (relative to source folder)
        """
        if not source_type:
            source_type = self._detect_source_type(source_id)

        expected = [
            f"{source_id}_source.json",      # Source-level metadata
            f"{source_id}_documents.json",   # Document metadata
            f"{source_id}_embeddings.json",  # Vector embeddings
            f"{source_id}_manifest.json",    # Distribution manifest
        ]

        # Add backup file based on type
        if source_type == "zim":
            expected.append(f"{source_id}.zim")

        return expected

    def cleanup_redundant_files(self, source_id: str) -> Dict[str, Any]:
        """
        Delete legacy v1 files ONLY if v2 equivalents exist.

        This safely removes redundant files like:
        - {source_id}_index.json (if _embeddings.json exists)

        NOTE: _metadata.json and _backup_manifest.json are NOT deleted because
        they serve different purposes than _documents.json and _manifest.json:
        - _metadata.json: metadata summary from generate_metadata
        - _backup_manifest.json: backup scan info from scan_backup
        - _documents.json: indexed document content
        - _manifest.json: distribution manifest for packs

        Will NOT delete anything if the source is still in v1 format.

        Returns:
            Dict with deleted files, kept files, and any errors
        """
        source_path = self.get_source_path(source_id)
        deleted = []
        kept = []
        errors = []

        if not source_path.exists():
            return {"deleted": [], "kept": [], "errors": ["Source folder does not exist"], "source_id": source_id}

        # Check if v2 schema files exist - ONLY cleanup if we have v2 format
        v2_embeddings = source_path / f"{source_id}_embeddings.json"
        v2_source = source_path / f"{source_id}_source.json"

        if not v2_embeddings.exists():
            # No v2 embeddings file - this source hasn't been indexed yet
            # DO NOT delete anything
            return {
                "deleted": [],
                "kept": [f.name for f in source_path.iterdir()],
                "expected": [],
                "errors": ["Source not indexed yet - run Create Index first"],
                "source_id": source_id,
                "freed_mb": 0
            }

        # V2 exists, now we can safely identify redundant legacy files
        # Only _index.json is truly redundant (replaced by _embeddings.json)
        legacy_patterns = [
            f"{source_id}_index.json",      # Replaced by _embeddings.json
        ]

        # File extensions to always keep (backup content)
        keep_extensions = {'.zim', '.pdf', '.html', '.htm'}

        # Directories to always keep (backup content)
        keep_dirs = {'pages', 'pdfs', 'images', 'assets'}

        # V2 schema files and other important files to keep
        v2_files = {
            f"{source_id}_source.json",
            f"{source_id}_documents.json",
            f"{source_id}_embeddings.json",
            f"{source_id}_manifest.json",
            f"{source_id}_metadata.json",         # Metadata summary (from generate_metadata)
            f"{source_id}_backup_manifest.json",  # Backup scan info (from scan_backup)
            "_collection.json",                   # PDF collection metadata
        }

        # Iterate through files in source folder
        for item in source_path.iterdir():
            filename = item.name

            # Check if file should be kept
            should_keep = True  # Default to keeping files

            # Only delete if it's a known legacy file
            if filename in legacy_patterns:
                should_keep = False

            # Always keep v2 files
            if filename in v2_files:
                should_keep = True

            # Always keep backup content by extension
            if item.is_file():
                ext = item.suffix.lower()
                if ext in keep_extensions:
                    should_keep = True

            # Always keep backup directories
            if item.is_dir() and filename.lower() in keep_dirs:
                should_keep = True

            if should_keep:
                kept.append(filename)
            else:
                # Delete this legacy file
                try:
                    if item.is_file():
                        size_mb = item.stat().st_size / (1024 * 1024)
                        item.unlink()
                        deleted.append({
                            "file": filename,
                            "size_mb": round(size_mb, 2)
                        })
                except Exception as e:
                    errors.append(f"Failed to delete {filename}: {e}")

        total_freed = sum(d.get("size_mb", 0) for d in deleted)

        return {
            "deleted": deleted,
            "kept": kept,
            "expected": list(v2_files),
            "errors": errors,
            "source_id": source_id,
            "freed_mb": round(total_freed, 2)
        }

    # =========================================================================
    # BACKUP SCANNING
    # =========================================================================

    def scan_backup(self, source_id: str, progress_callback: Callable = None) -> Dict[str, Any]:
        """
        Scan an existing backup folder and create/update the backup manifest.

        This is useful when:
        - HTML files were downloaded from R2 without a proper manifest
        - Files were manually added to the pages folder
        - The manifest is missing or corrupted

        Supports multiple source types:
        - HTML: Scans pages/ folder for .html files
        - PDF: Uses _collection.json metadata
        - ZIM: Extracts info from .zim file

        Args:
            source_id: Source identifier
            progress_callback: Function(current, total, message) for progress

        Returns:
            Dict with scan results including page count and manifest path
        """
        source_path = self.get_source_path(source_id)

        # Check for PDF source (_collection.json)
        collection_file = source_path / "_collection.json"
        if collection_file.exists():
            return self._scan_pdf_backup(source_id, source_path, collection_file, progress_callback)

        # Check for ZIM file
        zim_files = list(source_path.glob("*.zim"))
        if zim_files:
            return self._scan_zim_backup(source_id, source_path, zim_files[0], progress_callback)

        # Default: HTML source with pages/ folder
        return self._scan_html_backup(source_id, source_path, progress_callback)

    def _scan_pdf_backup(self, source_id: str, source_path: Path, collection_file: Path,
                         progress_callback: Callable = None) -> Dict[str, Any]:
        """Scan a PDF collection backup."""
        try:
            with open(collection_file, 'r', encoding='utf-8') as f:
                collection_data = json.load(f)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read _collection.json: {e}",
                "page_count": 0
            }

        documents = collection_data.get("documents", {})
        collection_info = collection_data.get("collection", {})

        if not documents:
            return {
                "success": False,
                "error": "No documents found in _collection.json",
                "page_count": 0
            }

        # Count PDF files and total size
        pdf_files = list(source_path.glob("*.pdf"))
        total_size = sum(f.stat().st_size for f in pdf_files)

        # Create backup manifest from collection data
        manifest = {
            "version": 2,
            "source_id": source_id,
            "source_type": "pdf",
            "created_at": datetime.now().isoformat(),
            "scanned_at": datetime.now().isoformat(),
            "collection_name": collection_info.get("name", source_id),
            "document_count": len(documents),
            "pdf_file_count": len(pdf_files),
            "total_size_bytes": total_size,
            "documents": documents
        }

        # Save manifest
        manifest_path = source_path / f"{source_id}_backup_manifest.json"
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to save manifest: {e}",
                "page_count": len(documents)
            }

        return {
            "success": True,
            "source_id": source_id,
            "source_type": "pdf",
            "page_count": len(documents),
            "pdf_file_count": len(pdf_files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "manifest_path": str(manifest_path),
            "message": f"Scanned PDF collection: {len(documents)} documents, {len(pdf_files)} PDF files"
        }

    def _scan_zim_backup(self, source_id: str, source_path: Path, zim_file: Path,
                         progress_callback: Callable = None) -> Dict[str, Any]:
        """Scan a ZIM file backup."""
        # ZIM files are self-contained, just record basic info
        file_size = zim_file.stat().st_size

        manifest = {
            "version": 2,
            "source_id": source_id,
            "source_type": "zim",
            "created_at": datetime.now().isoformat(),
            "scanned_at": datetime.now().isoformat(),
            "zim_file": zim_file.name,
            "total_size_bytes": file_size
        }

        # Save manifest
        manifest_path = source_path / f"{source_id}_backup_manifest.json"
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to save manifest: {e}",
                "page_count": 0
            }

        return {
            "success": True,
            "source_id": source_id,
            "source_type": "zim",
            "page_count": 0,  # Would need to extract from ZIM
            "total_size_mb": round(file_size / (1024 * 1024), 2),
            "manifest_path": str(manifest_path),
            "message": f"Scanned ZIM file: {zim_file.name} ({round(file_size / (1024 * 1024), 1)} MB)"
        }

    def _scan_html_backup(self, source_id: str, source_path: Path,
                          progress_callback: Callable = None) -> Dict[str, Any]:
        """Scan an HTML pages backup."""
        from bs4 import BeautifulSoup

        pages_dir = source_path / "pages"

        if not pages_dir.exists():
            return {
                "success": False,
                "error": f"No pages folder found at {pages_dir}. For PDF sources, use _collection.json.",
                "page_count": 0
            }

        # Find all HTML files
        html_files = list(pages_dir.glob("*.html")) + list(pages_dir.glob("*.htm"))

        if not html_files:
            return {
                "success": False,
                "error": "No HTML files found in pages folder",
                "page_count": 0
            }

        if progress_callback:
            progress_callback(0, len(html_files), f"Scanning {len(html_files)} HTML files...")

        # Build pages dict
        pages = {}
        errors = []

        for i, html_file in enumerate(html_files):
            try:
                filename = html_file.name

                # Try to extract title from HTML
                title = filename.replace(".html", "").replace(".htm", "").replace("_", " ")
                try:
                    with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(10000)  # Read first 10KB for title
                    soup = BeautifulSoup(content, 'html.parser')
                    if soup.title and soup.title.string:
                        title = soup.title.string.strip()
                except Exception:
                    pass  # Use filename-based title

                # Reconstruct URL from filename
                url_path = filename.replace(".html", "").replace(".htm", "").replace("_", "/")
                url = f"/{url_path}"

                # Get file size
                file_size = html_file.stat().st_size

                pages[url] = {
                    "filename": filename,
                    "title": title,
                    "size": file_size
                }

                if progress_callback and (i + 1) % 10 == 0:
                    progress_callback(i + 1, len(html_files), f"Scanned: {title[:50]}...")

            except Exception as e:
                errors.append(f"Error scanning {html_file.name}: {e}")

        if progress_callback:
            progress_callback(len(html_files), len(html_files), "Creating manifest...")

        # Calculate total size
        total_size = sum(p.get("size", 0) for p in pages.values())

        # Create backup manifest
        manifest = {
            "version": 2,
            "source_id": source_id,
            "created_at": datetime.now().isoformat(),
            "scanned_at": datetime.now().isoformat(),
            "page_count": len(pages),
            "total_size_bytes": total_size,
            "pages": pages
        }

        # Save manifest
        manifest_path = source_path / f"{source_id}_backup_manifest.json"
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to save manifest: {e}",
                "page_count": len(pages)
            }

        return {
            "success": True,
            "source_id": source_id,
            "page_count": len(pages),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "manifest_path": str(manifest_path),
            "errors": errors if errors else None,
            "message": f"Scanned {len(pages)} pages, created backup manifest"
        }

    # =========================================================================
    # BACKUP OPERATIONS
    # =========================================================================

    def create_backup(self, source_id: str, source_type: str,
                      base_url: str = None, scraper_type: str = "static",
                      limit: int = 1000,
                      progress_callback: Callable = None) -> BackupResult:
        """
        Create a backup for a source.

        Args:
            source_id: Unique identifier for this source
            source_type: One of: html, zim, pdf, substack
            base_url: Base URL for web scraping (required for html/substack)
            scraper_type: For HTML: mediawiki, static, fandom
            limit: Maximum pages/items to backup
            progress_callback: Function(current, total, message)

        Returns:
            BackupResult with success status and details
        """
        source_type = source_type.lower()

        if source_type == "html":
            return self._backup_html(source_id, base_url, scraper_type, limit, progress_callback)
        elif source_type == "substack":
            return self._backup_substack(source_id, base_url, limit, progress_callback)
        elif source_type == "zim":
            # ZIM files are pre-downloaded, just validate
            return self._validate_zim(source_id)
        elif source_type == "pdf":
            # PDF collections don't need backup - files are the backup
            return self._validate_pdf_folder(source_id)
        else:
            return BackupResult(
                success=False,
                source_id=source_id,
                backup_type=source_type,
                error=f"Unknown source type: {source_type}"
            )

    def _backup_html(self, source_id: str, base_url: str, scraper_type: str,
                     limit: int, progress_callback: Callable) -> BackupResult:
        """Backup an HTML website"""
        if not base_url:
            return BackupResult(
                success=False,
                source_id=source_id,
                backup_type="html",
                error="base_url is required for HTML backup"
            )

        try:
            from offline_tools.html_backup import run_backup

            backup_path = str(self.get_source_path(source_id))

            result = run_backup(
                backup_path=backup_path,
                source_id=source_id,
                base_url=base_url,
                scraper_type=scraper_type,
                limit=limit
            )

            return BackupResult(
                success=result.get("success", False),
                source_id=source_id,
                backup_type="html",
                page_count=result.get("page_count", 0),
                backup_path=backup_path,
                error=result.get("error", ""),
                warnings=result.get("warnings", [])
            )

        except ImportError as e:
            return BackupResult(
                success=False,
                source_id=source_id,
                backup_type="html",
                error=f"HTML backup tool not available: {e}"
            )
        except Exception as e:
            return BackupResult(
                success=False,
                source_id=source_id,
                backup_type="html",
                error=str(e)
            )

    def _backup_substack(self, source_id: str, base_url: str,
                         limit: int, progress_callback: Callable) -> BackupResult:
        """Backup a Substack newsletter"""
        if not base_url:
            return BackupResult(
                success=False,
                source_id=source_id,
                backup_type="substack",
                error="base_url is required for Substack backup"
            )

        try:
            from offline_tools.substack_backup import SubstackBackup

            backup = SubstackBackup(
                substack_url=base_url,
                output_dir=str(self.get_source_path(source_id))
            )

            result = backup.run(limit=limit)

            return BackupResult(
                success=result.get("success", False),
                source_id=source_id,
                backup_type="substack",
                page_count=result.get("post_count", 0),
                backup_path=str(self.get_source_path(source_id)),
                error=result.get("error", "")
            )

        except ImportError as e:
            return BackupResult(
                success=False,
                source_id=source_id,
                backup_type="substack",
                error=f"Substack backup tool not available: {e}"
            )
        except Exception as e:
            return BackupResult(
                success=False,
                source_id=source_id,
                backup_type="substack",
                error=str(e)
            )

    def _validate_zim(self, source_id: str) -> BackupResult:
        """Validate a ZIM file exists"""
        # Check root folder first
        zim_path = self.backup_path / f"{source_id}.zim"
        if zim_path.exists():
            return BackupResult(
                success=True,
                source_id=source_id,
                backup_type="zim",
                backup_path=str(zim_path)
            )

        # Check inside source folder for any .zim file
        source_path = self.get_source_path(source_id)
        if source_path.exists():
            zim_files = list(source_path.glob("*.zim"))
            if zim_files:
                return BackupResult(
                    success=True,
                    source_id=source_id,
                    backup_type="zim",
                    backup_path=str(zim_files[0])
                )

        return BackupResult(
            success=False,
            source_id=source_id,
            backup_type="zim",
            error=f"ZIM file not found in {self.backup_path} or {source_path}"
        )

    def _validate_pdf_folder(self, source_id: str) -> BackupResult:
        """Validate a PDF folder exists and has PDFs"""
        pdf_path = self.get_source_path(source_id)
        if not pdf_path.exists():
            return BackupResult(
                success=False,
                source_id=source_id,
                backup_type="pdf",
                error=f"PDF folder not found: {pdf_path}"
            )

        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            return BackupResult(
                success=False,
                source_id=source_id,
                backup_type="pdf",
                error=f"No PDF files found in: {pdf_path}"
            )

        return BackupResult(
            success=True,
            source_id=source_id,
            backup_type="pdf",
            page_count=len(pdf_files),
            backup_path=str(pdf_path)
        )

    # =========================================================================
    # INDEX OPERATIONS
    # =========================================================================

    def create_index(self, source_id: str, source_type: str = None,
                     limit: int = 1000, skip_existing: bool = True,
                     progress_callback: Callable = None) -> IndexResult:
        """
        Create an index for a source.

        Args:
            source_id: Source identifier
            source_type: One of: html, zim, pdf (auto-detected if not provided)
            limit: Maximum items to index
            skip_existing: Skip already-indexed items
            progress_callback: Function(current, total, message)

        Returns:
            IndexResult with success status and counts
        """
        # Note: normalize_filenames is called in validate_source, which should
        # be called before indexing. We don't duplicate it here.

        # Auto-detect source type if not provided
        if not source_type:
            source_type = self._detect_source_type(source_id)

        if not source_type:
            return IndexResult(
                success=False,
                source_id=source_id,
                error="Could not detect source type. Specify source_type explicitly."
            )

        source_type = source_type.lower()

        if source_type == "html":
            return self._index_html(source_id, limit, skip_existing, progress_callback)
        elif source_type == "zim":
            return self._index_zim(source_id, limit, progress_callback)
        elif source_type == "pdf":
            return self._index_pdf(source_id, limit, skip_existing, progress_callback)
        else:
            return IndexResult(
                success=False,
                source_id=source_id,
                error=f"Unknown source type: {source_type}"
            )

    def _detect_source_type(self, source_id: str) -> Optional[str]:
        """Auto-detect source type based on files present"""
        source_path = self.get_source_path(source_id)

        # Check for ZIM file - first in root backup folder
        zim_path = self.backup_path / f"{source_id}.zim"
        if zim_path.exists():
            return "zim"

        # Check for ZIM file inside source folder (any .zim file)
        if source_path.exists():
            zim_files = list(source_path.glob("*.zim"))
            if zim_files:
                return "zim"

            # Check for PDF collection (_collection.json is the definitive marker)
            if (source_path / "_collection.json").exists():
                return "pdf"

            # Check for PDFs (fallback - might not have _collection.json yet)
            if list(source_path.glob("*.pdf")):
                return "pdf"

            # Check for HTML backup manifest
            manifest = source_path / f"{source_id}_backup_manifest.json"
            if manifest.exists():
                return "html"

            # Legacy manifest
            if (source_path / "manifest.json").exists():
                return "html"

            # Check for pages folder (HTML backup)
            if (source_path / "pages").exists():
                return "html"

            # Check for HTML files directly
            if list(source_path.glob("*.html")):
                return "html"

        return None

    def _index_html(self, source_id: str, limit: int, skip_existing: bool,
                    progress_callback: Callable) -> IndexResult:
        """Index an HTML backup"""
        try:
            from offline_tools.indexer import index_html_backup

            source_path = self.get_source_path(source_id)

            result = index_html_backup(
                backup_path=str(source_path),
                source_id=source_id,
                limit=limit,
                progress_callback=progress_callback,
                skip_existing=skip_existing,
                backup_folder=str(source_path)
            )

            return IndexResult(
                success=result.get("success", False),
                source_id=source_id,
                indexed_count=result.get("indexed_count", 0),
                skipped_count=result.get("skipped", 0),
                total_chars=result.get("total_chars", 0),
                error=result.get("error", ""),
                errors=result.get("errors", [])
            )

        except Exception as e:
            return IndexResult(
                success=False,
                source_id=source_id,
                error=str(e)
            )

    def _index_zim(self, source_id: str, limit: int,
                   progress_callback: Callable) -> IndexResult:
        """Index a ZIM file"""
        try:
            from offline_tools.indexer import index_zim_file

            # Find ZIM file - check root first, then source folder
            zim_path = self.backup_path / f"{source_id}.zim"
            if not zim_path.exists():
                source_path = self.get_source_path(source_id)
                zim_files = list(source_path.glob("*.zim")) if source_path.exists() else []
                if zim_files:
                    zim_path = zim_files[0]
                else:
                    return IndexResult(
                        success=False,
                        source_id=source_id,
                        error=f"No ZIM file found for {source_id}"
                    )

            result = index_zim_file(
                zim_path=str(zim_path),
                source_id=source_id,
                limit=limit,
                progress_callback=progress_callback,
                backup_folder=str(self.get_source_path(source_id))
            )

            return IndexResult(
                success=result.get("success", False),
                source_id=source_id,
                indexed_count=result.get("indexed_count", 0),
                total_chars=result.get("total_chars", 0),
                error=result.get("error", ""),
                errors=result.get("errors", [])
            )

        except Exception as e:
            return IndexResult(
                success=False,
                source_id=source_id,
                error=str(e)
            )

    def _index_pdf(self, source_id: str, limit: int, skip_existing: bool,
                   progress_callback: Callable) -> IndexResult:
        """Index a PDF folder"""
        try:
            from offline_tools.indexer import index_pdf_folder

            source_path = self.get_source_path(source_id)

            result = index_pdf_folder(
                pdf_path=str(source_path),
                source_id=source_id,
                limit=limit,
                progress_callback=progress_callback,
                skip_existing=skip_existing,
                backup_folder=str(source_path)
            )

            return IndexResult(
                success=result.get("success", False),
                source_id=source_id,
                indexed_count=result.get("indexed_count", 0),
                skipped_count=result.get("skipped", 0),
                error=result.get("error", ""),
                errors=result.get("errors", [])
            )

        except Exception as e:
            return IndexResult(
                success=False,
                source_id=source_id,
                error=str(e)
            )

    # =========================================================================
    # VALIDATION & LICENSE DETECTION
    # =========================================================================

    def validate_source(self, source_id: str,
                        source_config: Dict[str, Any] = None) -> ValidationResult:
        """
        Validate a source is ready for distribution.

        Checks:
        - Has backup files
        - Has index/metadata
        - Has embeddings file (for offline use)
        - Has license (tries auto-detection if missing)
        - Has proper tags

        Args:
            source_id: Source identifier
            source_config: Optional source config from _source.json

        Returns:
            ValidationResult with issues and suggestions
        """
        result = ValidationResult(
            is_valid=True,
            source_id=source_id
        )

        source_path = self.get_source_path(source_id)
        source_config = source_config or {}

        # Normalize file names first (e.g., rename ZIM files to standard format)
        # This runs early so validation sees the correct file names
        normalize_result = self.normalize_filenames(source_id)
        if normalize_result["renamed"]:
            for r in normalize_result["renamed"]:
                print(f"Normalized: {r['old']} -> {r['new']}")

        # Check backup exists
        source_type = self._detect_source_type(source_id)
        if source_type:
            result.has_backup = True
        else:
            result.has_backup = False
            result.issues.append("No backup found (no HTML, ZIM, or PDF files)")
            result.is_valid = False

        # V2 schema file paths
        documents_file = source_path / f"{source_id}_documents.json"
        embeddings_file = source_path / f"{source_id}_embeddings.json"
        source_file = source_path / f"{source_id}_source.json"

        # Legacy v1 file paths (for detection)
        legacy_metadata = source_path / f"{source_id}_metadata.json"
        legacy_index = source_path / f"{source_id}_index.json"

        # Check metadata/documents - prefer v2, fall back to v1
        if documents_file.exists():
            result.has_metadata = True
            result.has_index = True
            # Verify document count
            try:
                with open(documents_file, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
                doc_count = len(docs_data.get("documents", []))
                if doc_count == 0:
                    result.has_metadata = False
                    result.issues.append("Documents file exists but is empty - run 'Generate Metadata'")
                    result.is_valid = False
            except Exception:
                result.has_metadata = False
                result.issues.append("Documents file exists but could not be read - run 'Generate Metadata'")
                result.is_valid = False
        elif legacy_metadata.exists():
            # Has v1 format only
            result.has_metadata = True
            result.has_index = True
            result.warnings.append("Using legacy v1 metadata - run 'Generate Metadata' to upgrade to v2")
        else:
            result.has_metadata = False
            result.has_index = False
            result.issues.append("No metadata - run 'Generate Metadata' to create")
            result.is_valid = False

        # Check embeddings - prefer v2, fall back to v1
        if embeddings_file.exists():
            result.has_embeddings = True
            # Verify it has actual content
            try:
                with open(embeddings_file, 'r', encoding='utf-8') as f:
                    embed_data = json.load(f)
                # Check both "vectors" (v2 format) and "embeddings" (legacy) keys
                embed_count = len(embed_data.get("vectors", embed_data.get("embeddings", [])))
                # Also check document_count field as fallback
                if embed_count == 0:
                    embed_count = embed_data.get("document_count", 0)
                if embed_count == 0:
                    result.has_embeddings = False
                    result.issues.append("Embeddings file exists but is empty - run 'Create Index'")
                    result.is_valid = False
            except Exception:
                result.has_embeddings = False
                result.issues.append("Embeddings file exists but could not be read - run 'Create Index'")
                result.is_valid = False
        elif legacy_index.exists():
            # Has v1 format only
            result.has_embeddings = True
            result.warnings.append("Using legacy v1 embeddings - run 'Create Index' to upgrade to v2")
        else:
            result.has_embeddings = False
            result.issues.append("No embeddings - run 'Create Index' to create")
            result.is_valid = False

        # Check license
        license_val = source_config.get("license", "")
        if license_val and license_val.lower() not in ["unknown", ""]:
            result.has_license = True
            result.license_verified = source_config.get("license_verified", False)
            if not result.license_verified:
                result.warnings.append("License not verified")
        else:
            result.has_license = False
            result.issues.append("No license specified")
            result.is_valid = False

            # Try auto-detection
            detected = self.detect_license(source_id)
            if detected:
                result.detected_license = detected
                result.warnings.append(f"Auto-detected license: {detected} (needs verification)")

        # Check tags - check source_config first, then _collection.json for PDF sources
        tags = source_config.get("tags", []) or source_config.get("topics", [])

        # For PDF sources, also check _collection.json topics
        if not tags and source_type == "pdf":
            collection_file = source_path / "_collection.json"
            if collection_file.exists():
                try:
                    with open(collection_file, 'r', encoding='utf-8') as f:
                        collection_data = json.load(f)
                    tags = collection_data.get("collection", {}).get("topics", [])
                except Exception:
                    pass

        if tags:
            result.has_tags = True
        else:
            result.has_tags = False
            result.warnings.append("No tags specified")

            # Suggest tags
            suggested = self.suggest_tags(source_id)
            if suggested:
                result.suggested_tags = suggested
                result.warnings.append(f"Suggested tags: {', '.join(suggested)}")

        # Check v2 format files (already defined above, just set result attributes)
        categories_dir = source_path / f"{source_id}_categories"

        result.has_source_metadata = source_file.exists()
        result.has_documents_file = documents_file.exists()
        result.has_embeddings_file = embeddings_file.exists()
        result.has_categories = categories_dir.exists() and categories_dir.is_dir()

        # Check for legacy files that can be cleaned up ONLY if v2 equivalents exist
        # This is conservative - we only suggest cleanup for known legacy files
        # NOTE: _metadata.json and _backup_manifest.json are NOT redundant - they serve
        # different purposes than _documents.json and _manifest.json
        legacy_files_to_check = [
            f"{source_id}_index.json",      # Replaced by _embeddings.json
        ]

        if source_path.exists():
            for legacy_file in legacy_files_to_check:
                legacy_path = source_path / legacy_file
                if legacy_path.exists():
                    # Only mark as redundant if v2 equivalent exists
                    if "_index.json" in legacy_file and embeddings_file.exists():
                        result.redundant_files.append(legacy_file)

        # Determine schema version
        if result.has_source_metadata and result.has_documents_file and result.has_embeddings_file:
            result.schema_version = 2
        else:
            result.schema_version = 1

        result.needs_migration = len(result.legacy_files) > 0
        result.has_cleanup_needed = len(result.redundant_files) > 0

        # Add warnings
        if result.needs_migration:
            result.warnings.append(f"Legacy files need migration: {', '.join(result.legacy_files)}")
        if result.has_cleanup_needed:
            result.warnings.append(f"Unexpected files found (not in v2 schema): {', '.join(result.redundant_files)}")

        # Determine if production ready (all critical checks pass)
        # For now, accept both schema v1 and v2, but prefer v2
        result.production_ready = (
            result.has_backup and
            result.has_metadata and
            result.has_embeddings and
            result.has_license
        )

        return result

    def validate_for_production(self, source_id: str,
                                source_config: Dict[str, Any] = None,
                                require_v2: bool = False) -> ValidationResult:
        """
        Strict validation for production upload (R2 backups/ or Pinecone).

        This is the gate that prevents incomplete packages from being
        uploaded to the global repository. A source must pass ALL checks:

        REQUIRED:
        - Has backup files (HTML/ZIM/PDF)
        - Has metadata file
        - Has embeddings file with content
        - Has license specified (not "Unknown")

        SCHEMA v2 (when require_v2=True):
        - {source_id}_source.json (source-level metadata)
        - {source_id}_documents.json (document metadata)
        - {source_id}_embeddings.json (vectors only, no content duplication)

        RECOMMENDED (warnings only):
        - License is verified
        - Has tags for categorization

        Args:
            source_id: Source identifier
            source_config: Source configuration from _source.json
            require_v2: If True, require schema v2 format (default False for transition)

        Returns:
            ValidationResult with production_ready flag
        """
        result = self.validate_source(source_id, source_config)

        # Add stricter checks for production
        if not result.has_embeddings:
            # Promote from warning to issue for production
            result.issues.append("PRODUCTION BLOCKER: No embeddings file - users cannot search offline")
            result.is_valid = False
            result.production_ready = False

        # Check if schema v2 is required
        if require_v2 and result.schema_version < 2:
            result.issues.append("PRODUCTION BLOCKER: Schema v2 format required - run migration first")
            result.is_valid = False
            result.production_ready = False

            # Add specific missing files
            if not result.has_source_metadata:
                result.issues.append(f"Missing: {source_id}_source.json")
            if not result.has_documents_file:
                result.issues.append(f"Missing: {source_id}_documents.json")
            if not result.has_embeddings_file:
                result.issues.append(f"Missing: {source_id}_embeddings.json")

        # Check manifest exists
        source_path = self.get_source_path(source_id)
        manifest_file = source_path / f"{source_id}_manifest.json"
        if not manifest_file.exists():
            result.warnings.append("No distribution manifest - will be created on upload")

        return result

    def detect_license(self, source_id: str) -> Optional[str]:
        """
        Try to auto-detect license from source content.

        Scans backup content and metadata for license indicators.

        Returns:
            Detected license string or None
        """
        source_path = self.get_source_path(source_id)

        # Check metadata file first
        metadata_file = source_path / f"{source_id}_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Check if metadata has license info
                if "license" in metadata:
                    return metadata["license"]

                # Scan document content for license patterns
                docs = metadata.get("documents", {})
                sample_content = ""
                for doc_id, doc_info in list(docs.items())[:10]:
                    # Try to read actual content from backup
                    content = self._get_sample_content(source_path, doc_info)
                    if content:
                        sample_content += " " + content[:2000]

                if sample_content:
                    detected = self._match_license_patterns(sample_content)
                    if detected:
                        return detected

            except Exception:
                pass

        # Check for common license files
        for license_file in ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"]:
            lf = source_path / license_file
            if lf.exists():
                try:
                    content = lf.read_text(encoding='utf-8', errors='ignore')
                    detected = self._match_license_patterns(content)
                    if detected:
                        return detected
                except Exception:
                    pass

        # Check backup manifest for license info
        manifest = source_path / f"{source_id}_backup_manifest.json"
        if manifest.exists():
            try:
                with open(manifest, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if "license" in data:
                    return data["license"]
            except Exception:
                pass

        return None

    def _get_sample_content(self, source_path: Path, doc_info: dict) -> str:
        """Get sample content from a document for license scanning"""
        # Try to find the actual file
        url = doc_info.get("url", "")
        if url.startswith("file://"):
            url = url[7:]

        # Check pages folder for HTML
        pages_dir = source_path / "pages"
        if pages_dir.exists():
            # Try to find file by content hash or filename
            content_hash = doc_info.get("content_hash", "")
            for html_file in pages_dir.glob("*.html"):
                try:
                    content = html_file.read_text(encoding='utf-8', errors='ignore')
                    return content[:5000]  # First 5k chars
                except Exception:
                    continue

        return ""

    def _match_license_patterns(self, text: str) -> Optional[str]:
        """Match text against known license patterns"""
        text_lower = text.lower()

        for license_name, patterns in self.LICENSE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return license_name

        return None

    def suggest_tags(self, source_id: str) -> List[str]:
        """
        Suggest tags based on source content.

        Scans titles and content for topic keywords.
        For PDF sources, also checks _collection.json topics.

        Returns:
            List of suggested tag strings
        """
        source_path = self.get_source_path(source_id)
        suggested = set()

        # For PDF sources, check _collection.json topics first
        collection_file = source_path / "_collection.json"
        if collection_file.exists():
            try:
                with open(collection_file, 'r', encoding='utf-8') as f:
                    collection_data = json.load(f)
                topics = collection_data.get("collection", {}).get("topics", [])
                for topic in topics:
                    suggested.add(topic)
            except Exception:
                pass

        # Check metadata for titles
        metadata_file = source_path / f"{source_id}_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                # Scan titles
                all_text = ""
                for doc_id, doc_info in metadata.get("documents", {}).items():
                    title = doc_info.get("title", "")
                    all_text += " " + title.lower()

                # Match topic keywords
                for tag, keywords in self.TOPIC_KEYWORDS.items():
                    for keyword in keywords:
                        if keyword in all_text:
                            suggested.add(tag)
                            break

            except Exception:
                pass

        # Also check source_id itself
        source_lower = source_id.lower()
        for tag, keywords in self.TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in source_lower:
                    suggested.add(tag)
                    break

        return list(suggested)[:5]  # Limit to 5 suggestions

    # =========================================================================
    # PACK CREATION
    # =========================================================================

    def create_pack(self, source_id: str, source_config: Dict[str, Any],
                    approval_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a distribution pack for a source.

        Args:
            source_id: Source identifier
            source_config: Source configuration from _source.json
            approval_info: Optional approval information

        Returns:
            Dict with pack info and file paths
        """
        from .pack_tools import create_pack_manifest, load_metadata

        source_path = self.get_source_path(source_id)

        # Validate first
        validation = self.validate_source(source_id, source_config)
        if not validation.is_valid:
            return {
                "success": False,
                "error": "Source validation failed",
                "issues": validation.issues
            }

        # Load metadata
        metadata = load_metadata(source_id)
        if not metadata:
            return {
                "success": False,
                "error": "Could not load metadata"
            }

        # Detect backup info
        source_type = self._detect_source_type(source_id)
        backup_info = {"type": source_type}

        if source_type == "zim":
            # Find ZIM file - check root first, then source folder
            zim_path = self.backup_path / f"{source_id}.zim"
            if not zim_path.exists():
                zim_files = list(source_path.glob("*.zim")) if source_path.exists() else []
                if zim_files:
                    zim_path = zim_files[0]
            backup_info["size_mb"] = zim_path.stat().st_size / (1024 * 1024) if zim_path.exists() else 0
        elif source_type == "html":
            # Calculate HTML backup size
            total_size = sum(f.stat().st_size for f in source_path.rglob("*") if f.is_file())
            backup_info["size_mb"] = total_size / (1024 * 1024)
        elif source_type == "pdf":
            total_size = sum(f.stat().st_size for f in source_path.glob("*.pdf"))
            backup_info["size_mb"] = total_size / (1024 * 1024)

        # Create manifest
        manifest = create_pack_manifest(
            source_id=source_id,
            source_config=source_config,
            metadata=metadata,
            backup_info=backup_info,
            approval_info=approval_info
        )

        # Save manifest
        manifest_path = source_path / f"{source_id}_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

        return {
            "success": True,
            "source_id": source_id,
            "manifest_path": str(manifest_path),
            "manifest": manifest,
            "validation": asdict(validation)
        }
