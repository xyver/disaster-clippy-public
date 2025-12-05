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

from .schemas import (
    get_manifest_file, get_metadata_file, get_index_file,
    get_vectors_file, get_backup_manifest_file
)


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

    # Schema v3 fields
    schema_version: int = 1  # 1=legacy, 2=old layered, 3=new v3 format
    has_manifest: bool = False  # _manifest.json
    has_metadata_file: bool = False  # _metadata.json
    has_index_file: bool = False  # _index.json
    has_vectors_file: bool = False  # _vectors.json
    has_backup_manifest: bool = False  # backup_manifest.json
    legacy_files: List[str] = None  # Old format files that need migration
    redundant_files: List[str] = None  # Files that can be safely deleted
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

    # Topic keywords for tag suggestions during indexing
    # These tags help categorize content for search across multiple sources
    # Format: "tag_name": ["keyword1", "keyword2", ...]
    TOPIC_KEYWORDS = {
        # Water and sanitation
        "water": ["water", "filtration", "purification", "well", "pump", "irrigation", "rainwater", "cistern", "aquifer", "groundwater"],
        "sanitation": ["sanitation", "sewage", "latrine", "toilet", "waste", "hygiene", "handwashing"],

        # Energy systems
        "solar": ["solar", "photovoltaic", "pv panel", "sun power", "solar cooker", "solar oven", "solar still"],
        "energy": ["energy", "power", "electricity", "generator", "battery", "inverter", "off-grid", "microgrid"],
        "wind": ["wind turbine", "windmill", "wind power", "wind energy"],
        "biogas": ["biogas", "methane", "digester", "anaerobic"],
        "fuel": ["fuel", "gasoline", "diesel", "propane", "wood gas", "charcoal", "firewood", "alcohol fuel"],

        # Food and agriculture
        "food": ["food", "cooking", "preservation", "canning", "drying", "smoking", "fermenting", "recipe"],
        "agriculture": ["agriculture", "farming", "garden", "crop", "seed", "soil", "compost", "fertilizer", "permaculture"],
        "livestock": ["livestock", "chicken", "goat", "rabbit", "cattle", "poultry", "animal husbandry", "beekeeping"],
        "aquaculture": ["fishing", "aquaculture", "fish farming", "aquaponics"],
        "foraging": ["foraging", "wild edible", "mushroom", "wild plant", "hunting"],

        # Shelter and construction
        "shelter": ["shelter", "housing", "tent", "tarp", "emergency shelter"],
        "construction": ["construction", "building", "masonry", "concrete", "timber", "earthbag", "cob", "adobe", "rammed earth"],
        "tools": ["tools", "workshop", "forge", "metalwork", "woodwork", "blacksmith"],

        # Health and medical
        "medical": ["medical", "medicine", "first aid", "trauma", "wound", "infection", "disease", "illness", "health"],
        "herbal": ["herbal", "medicinal plant", "natural remedy", "herb", "botanical"],
        "mental-health": ["mental health", "stress", "psychological", "coping", "resilience"],
        "nutrition": ["nutrition", "vitamin", "mineral", "malnutrition", "diet"],

        # Emergency and disaster types
        "emergency": ["emergency", "disaster", "survival", "preparedness", "crisis", "evacuation", "bug out"],
        "fire": ["wildfire", "fire safety", "firefighting", "fire starting", "forest fire"],
        "earthquake": ["earthquake", "seismic", "tremor", "quake"],
        "flood": ["flood", "flooding", "flash flood", "levee", "dam"],
        "hurricane": ["hurricane", "typhoon", "cyclone", "storm surge", "tropical storm"],
        "nuclear": ["nuclear", "radiation", "fallout", "radioactive", "nbc", "emp"],
        "pandemic": ["pandemic", "epidemic", "quarantine", "infectious disease", "outbreak"],

        # Skills and knowledge
        "navigation": ["navigation", "compass", "map", "gps", "orienteering", "wayfinding"],
        "communication": ["communication", "radio", "ham radio", "amateur radio", "signal", "morse code"],
        "security": ["security", "self-defense", "protection", "perimeter", "opsec"],
        "knots": ["knots", "rope", "cordage", "lashing", "splicing"],

        # Technology categories
        "appropriate-tech": ["appropriate technology", "low tech", "intermediate technology", "sustainable technology"],
        "electronics": ["electronics", "circuit", "arduino", "raspberry pi", "microcontroller"],
        "vehicles": ["vehicle", "car", "truck", "bicycle", "boat", "motorcycle", "engine"],

        # Content types
        "reference": ["reference", "manual", "handbook", "guide", "encyclopedia", "wikipedia"],
        "how-to": ["how to", "tutorial", "instructions", "step by step", "diy", "homemade", "build your own"],
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
                from admin.local_config import get_local_config
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

    def rename_source(self, old_source_id: str, new_source_id: str) -> Dict[str, Any]:
        """
        Rename a source ID and all associated files.

        This renames:
        - The source folder itself
        - All JSON config/metadata files with source_id prefix
        - The source_id field inside JSON files
        - ZIM files (if present)
        - Updates _master.json entry
        - Updates ChromaDB collection metadata

        Args:
            old_source_id: Current source ID
            new_source_id: New source ID (must be valid slug format)

        Returns:
            Dict with renamed files, updated references, and any errors
        """
        import shutil
        import re

        # Validate new source ID format (slug: lowercase, alphanumeric, underscores, hyphens)
        if not re.match(r'^[a-z0-9][a-z0-9_-]*$', new_source_id):
            return {
                "success": False,
                "error": "Invalid source ID format. Use lowercase letters, numbers, underscores, and hyphens. Must start with letter or number.",
                "old_source_id": old_source_id,
                "new_source_id": new_source_id
            }

        if new_source_id == old_source_id:
            return {
                "success": False,
                "error": "New source ID is the same as the old one",
                "old_source_id": old_source_id,
                "new_source_id": new_source_id
            }

        old_path = self.get_source_path(old_source_id)
        new_path = self.get_source_path(new_source_id)

        if not old_path.exists():
            return {
                "success": False,
                "error": f"Source folder not found: {old_source_id}",
                "old_source_id": old_source_id,
                "new_source_id": new_source_id
            }

        if new_path.exists():
            return {
                "success": False,
                "error": f"Target source ID already exists: {new_source_id}",
                "old_source_id": old_source_id,
                "new_source_id": new_source_id
            }

        renamed_files = []
        updated_json = []
        errors = []

        # Step 1: Rename files inside the folder BEFORE renaming the folder
        try:
            # Find all files with old_source_id prefix
            for file_path in old_path.iterdir():
                if file_path.is_file() and file_path.name.startswith(f"{old_source_id}_"):
                    # Rename file: old_source_id_xxx.json -> new_source_id_xxx.json
                    suffix = file_path.name[len(old_source_id):]  # e.g., "_source.json"
                    new_file_name = f"{new_source_id}{suffix}"
                    new_file_path = file_path.parent / new_file_name

                    try:
                        file_path.rename(new_file_path)
                        renamed_files.append({
                            "old": file_path.name,
                            "new": new_file_name,
                            "type": "config"
                        })
                    except Exception as e:
                        errors.append(f"Failed to rename {file_path.name}: {e}")

                # Also rename ZIM files
                elif file_path.is_file() and file_path.suffix == ".zim":
                    if file_path.stem == old_source_id or file_path.stem.startswith(f"{old_source_id}_"):
                        new_zim_name = f"{new_source_id}.zim"
                        new_zim_path = file_path.parent / new_zim_name

                        try:
                            self._force_close_zim_handles(file_path)
                            file_path.rename(new_zim_path)
                            renamed_files.append({
                                "old": file_path.name,
                                "new": new_zim_name,
                                "type": "zim"
                            })
                        except Exception as e:
                            errors.append(f"Failed to rename ZIM file {file_path.name}: {e}")
        except Exception as e:
            errors.append(f"Error scanning source folder: {e}")

        # Step 2: Update source_id field inside JSON files
        json_files_to_update = [
            get_manifest_file(),
            get_metadata_file(),
            get_index_file(),
            get_vectors_file(),
            get_backup_manifest_file(),
        ]

        for json_file in json_files_to_update:
            json_path = old_path / json_file
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # Update source_id field if present
                    modified = False
                    if data.get("source_id") == old_source_id:
                        data["source_id"] = new_source_id
                        modified = True

                    # Update name if it matches old source_id
                    if data.get("name") == old_source_id:
                        data["name"] = new_source_id.replace("_", " ").replace("-", " ").title()
                        modified = True

                    if modified:
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2)
                        updated_json.append(json_file)

                except Exception as e:
                    errors.append(f"Failed to update {json_file}: {e}")

        # Step 3: Rename the folder itself
        try:
            old_path.rename(new_path)
            renamed_files.append({
                "old": old_source_id,
                "new": new_source_id,
                "type": "folder"
            })
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to rename source folder: {e}. Files inside may have been renamed.",
                "old_source_id": old_source_id,
                "new_source_id": new_source_id,
                "renamed_files": renamed_files,
                "errors": errors
            }

        # Step 4: Update _master.json
        master_path = self.backup_path / "_master.json"
        master_updated = False
        if master_path.exists():
            try:
                with open(master_path, 'r', encoding='utf-8') as f:
                    master_data = json.load(f)

                sources = master_data.get("sources", {})
                if old_source_id in sources:
                    # Move entry to new key
                    sources[new_source_id] = sources.pop(old_source_id)
                    sources[new_source_id]["source_id"] = new_source_id
                    master_data["sources"] = sources
                    master_data["last_updated"] = datetime.now().isoformat()

                    with open(master_path, 'w', encoding='utf-8') as f:
                        json.dump(master_data, f, indent=2)
                    master_updated = True

            except Exception as e:
                errors.append(f"Failed to update _master.json: {e}")

        # Step 5: Update ChromaDB (delete old, re-add with new source_id)
        chromadb_updated = False
        chromadb_count = 0
        try:
            from offline_tools.vectordb import get_vector_store
            store = get_vector_store()

            # Get all documents with old source_id
            results = store.collection.get(
                where={"source_id": old_source_id},
                include=["documents", "metadatas", "embeddings"]
            )

            if results and results.get("ids"):
                chromadb_count = len(results["ids"])

                # Update metadata with new source_id
                new_metadatas = []
                for meta in results.get("metadatas", []):
                    meta["source_id"] = new_source_id
                    new_metadatas.append(meta)

                # Delete old entries
                store.collection.delete(ids=results["ids"])

                # Re-add with updated metadata
                store.collection.upsert(
                    ids=results["ids"],
                    documents=results.get("documents", []),
                    metadatas=new_metadatas,
                    embeddings=results.get("embeddings", [])
                )
                chromadb_updated = True

        except Exception as e:
            errors.append(f"Failed to update ChromaDB: {e}")

        return {
            "success": len(errors) == 0,
            "old_source_id": old_source_id,
            "new_source_id": new_source_id,
            "renamed_files": renamed_files,
            "updated_json": updated_json,
            "master_updated": master_updated,
            "chromadb_updated": chromadb_updated,
            "chromadb_documents": chromadb_count,
            "errors": errors,
            "message": f"Renamed {old_source_id} to {new_source_id}" if len(errors) == 0 else f"Rename completed with {len(errors)} error(s)"
        }

    def get_expected_files(self, source_id: str, source_type: str = None) -> List[str]:
        """
        Get list of expected files for a source package.

        Args:
            source_id: Source identifier
            source_type: 'zim', 'html', or 'pdf' (auto-detected if not provided)

        Returns:
            List of expected filenames (relative to source folder)
        """
        from .schemas import (
            get_manifest_file, get_metadata_file, get_index_file,
            get_vectors_file, get_backup_manifest_file
        )

        if not source_type:
            source_type = self._detect_source_type(source_id)

        expected = [
            get_manifest_file(),      # _manifest.json
            get_metadata_file(),      # _metadata.json
            get_index_file(),         # _index.json
            get_vectors_file(),       # _vectors.json
        ]

        # Add backup file based on type
        if source_type == "zim":
            expected.append(f"{source_id}.zim")
        elif source_type == "html":
            expected.append(get_backup_manifest_file())  # backup_manifest.json

        return expected

    def cleanup_redundant_files(self, source_id: str) -> Dict[str, Any]:
        """
        Delete legacy/redundant files from a source folder.

        This safely removes redundant files like:
        - {source_id}_source.json (if _manifest.json exists)
        - {source_id}_documents.json (if _metadata.json exists)
        - {source_id}_embeddings.json (if _vectors.json exists)
        - {source_id}_index.json (legacy)
        - {source_id}_manifest.json (merged into _manifest.json)
        - {source_id}_backup_manifest.json (if backup_manifest.json exists)

        Will NOT delete anything if the source is still in legacy format.

        Returns:
            Dict with deleted files, kept files, and any errors
        """
        from .schemas import (
            get_manifest_file, get_metadata_file, get_index_file,
            get_vectors_file, get_backup_manifest_file
        )

        source_path = self.get_source_path(source_id)
        deleted = []
        kept = []
        errors = []

        if not source_path.exists():
            return {"deleted": [], "kept": [], "errors": ["Source folder does not exist"], "source_id": source_id}

        # Check if schema files exist - ONLY cleanup if source is fully indexed
        manifest_path = source_path / get_manifest_file()
        vectors_path = source_path / get_vectors_file()

        if not vectors_path.exists():
            # No vectors file - this source hasn't been indexed yet
            return {
                "deleted": [],
                "kept": [f.name for f in source_path.iterdir()],
                "expected": [],
                "errors": ["Source not indexed yet - run Create Index first"],
                "source_id": source_id,
                "freed_mb": 0
            }

        # Identify legacy files that can be deleted
        legacy_patterns = [
            f"{source_id}_source.json",           # Replaced by _manifest.json
            f"{source_id}_documents.json",        # Replaced by _metadata.json
            f"{source_id}_embeddings.json",       # Replaced by _vectors.json
            f"{source_id}_index.json",            # Legacy
            f"{source_id}_manifest.json",         # Merged into _manifest.json
            f"{source_id}_metadata.json",         # Legacy metadata
            f"{source_id}_backup_manifest.json",  # Replaced by backup_manifest.json
        ]

        # File extensions to always keep (backup content)
        keep_extensions = {'.zim', '.pdf', '.html', '.htm'}

        # Directories to always keep (backup content)
        keep_dirs = {'pages', 'pdfs', 'images', 'assets'}

        # Schema files and other important files to keep
        schema_files = {
            get_manifest_file(),        # _manifest.json
            get_metadata_file(),        # _metadata.json
            get_index_file(),           # _index.json
            get_vectors_file(),         # _vectors.json
            get_backup_manifest_file(), # backup_manifest.json
            "_collection.json",         # PDF collection metadata
        }

        # Iterate through files in source folder
        for item in source_path.iterdir():
            filename = item.name

            # Check if file should be kept
            should_keep = True  # Default to keeping files

            # Only delete if it's a known legacy file
            if filename in legacy_patterns:
                should_keep = False

            # Always keep v3 files
            if filename in schema_files:
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
            "expected": list(schema_files),
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
        from .schemas import get_backup_manifest_file
        manifest_path = source_path / get_backup_manifest_file()
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
        from .schemas import get_backup_manifest_file
        manifest_path = source_path / get_backup_manifest_file()
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
        from .schemas import get_backup_manifest_file
        manifest_path = source_path / get_backup_manifest_file()
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
            from offline_tools.backup.html import run_backup

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
            from offline_tools.backup.substack import SubstackBackup

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
            source_config: Optional source config from _manifest.json

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

        # Schema file paths
        manifest_file = source_path / get_manifest_file()
        metadata_file = source_path / get_metadata_file()
        index_file = source_path / get_index_file()
        vectors_file = source_path / get_vectors_file()
        backup_manifest_file = source_path / get_backup_manifest_file()

        # Check metadata
        if metadata_file.exists():
            result.has_metadata = True
            result.has_index = True
            # Verify document count
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
                doc_count = len(docs_data.get("documents", {}))
                if doc_count == 0:
                    result.has_metadata = False
                    result.issues.append("Metadata file exists but is empty - run 'Create Index'")
                    result.is_valid = False
            except Exception:
                result.has_metadata = False
                result.issues.append("Metadata file exists but could not be read - run 'Create Index'")
                result.is_valid = False
        else:
            result.has_metadata = False
            result.has_index = False
            result.issues.append("No metadata - run 'Create Index' to create")
            result.is_valid = False

        # Check vectors - prefer v3, fall back to legacy
        if vectors_file.exists():
            result.has_embeddings = True
            # Verify it has actual content
            try:
                with open(vectors_file, 'r', encoding='utf-8') as f:
                    embed_data = json.load(f)
                embed_count = len(embed_data.get("vectors", {}))
                if embed_count == 0:
                    embed_count = embed_data.get("document_count", 0)
                if embed_count == 0:
                    result.has_embeddings = False
                    result.issues.append("Vectors file exists but is empty - run 'Create Index'")
                    result.is_valid = False
            except Exception:
                result.has_embeddings = False
                result.issues.append("Vectors file exists but could not be read - run 'Create Index'")
                result.is_valid = False
        elif legacy_embeddings.exists():
            # Has legacy format only
            result.has_embeddings = True
            result.warnings.append("Using legacy embeddings format - run 'Create Index' to upgrade")
        elif legacy_index.exists():
            result.has_embeddings = True
            result.warnings.append("Using very old index format - run 'Create Index' to upgrade")
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

        # Check v3 format files
        result.has_manifest = manifest_file.exists()
        result.has_metadata_file = metadata_file.exists()
        result.has_index_file = index_file.exists()
        result.has_vectors_file = vectors_file.exists()
        result.has_backup_manifest = backup_manifest_file.exists()

        # Check for legacy files that can be cleaned up
        legacy_files_to_check = [
            f"{source_id}_source.json",
            f"{source_id}_documents.json",
            f"{source_id}_embeddings.json",
            f"{source_id}_index.json",
            f"{source_id}_manifest.json",
            f"{source_id}_backup_manifest.json",
        ]

        if source_path.exists():
            for legacy_file in legacy_files_to_check:
                legacy_path = source_path / legacy_file
                if legacy_path.exists():
                    result.legacy_files.append(legacy_file)
                    # Mark as redundant if v3 equivalent exists
                    if vectors_file.exists() or metadata_file.exists():
                        result.redundant_files.append(legacy_file)

        # Determine schema version
        if result.has_manifest and result.has_metadata_file and result.has_vectors_file:
            result.schema_version = 3
        elif legacy_source.exists() and legacy_documents.exists() and legacy_embeddings.exists():
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

        SCHEMA v3 (when require_v2=True):
        - _manifest.json (source-level metadata)
        - _metadata.json (document metadata)
        - _vectors.json (vectors only, no content duplication)

        RECOMMENDED (warnings only):
        - License is verified
        - Has tags for categorization

        Args:
            source_id: Source identifier
            source_config: Source configuration from _manifest.json
            require_v2: If True, require schema v3 format

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

        # Check if schema v3 is required
        if require_v2 and result.schema_version < 3:
            result.issues.append("PRODUCTION BLOCKER: Schema v3 format required - run Create Index first")
            result.is_valid = False
            result.production_ready = False

            # Add specific missing files
            from .schemas import get_manifest_file, get_metadata_file, get_vectors_file
            if not result.has_manifest:
                result.issues.append(f"Missing: {get_manifest_file()}")
            if not result.has_metadata_file:
                result.issues.append(f"Missing: {get_metadata_file()}")
            if not result.has_vectors_file:
                result.issues.append(f"Missing: {get_vectors_file()}")

        # Check manifest exists
        source_path = self.get_source_path(source_id)
        from .schemas import get_manifest_file
        manifest_file_path = source_path / get_manifest_file()
        if not manifest_file_path.exists():
            result.warnings.append("No manifest - will be created on upload")

        return result

    def detect_license(self, source_id: str) -> Optional[str]:
        """
        Try to auto-detect license from source content.

        Scans backup content and metadata for license indicators.

        Returns:
            Detected license string or None
        """
        from .schemas import get_metadata_file, get_manifest_file, get_backup_manifest_file

        source_path = self.get_source_path(source_id)

        # Check metadata file first (v3 or legacy)
        metadata_file = source_path / get_metadata_file()
        if not metadata_file.exists():
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

        # Check backup manifest for license info (v3 or legacy)
        manifest = source_path / get_backup_manifest_file()
        if not manifest.exists():
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
        from .schemas import get_metadata_file

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

        # Check metadata for titles (v3 or legacy)
        metadata_file = source_path / get_metadata_file()
        if not metadata_file.exists():
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

        return list(suggested)[:10]  # Limit to 10 suggestions

    # =========================================================================
    # PACK CREATION
    # =========================================================================

    def create_pack(self, source_id: str, source_config: Dict[str, Any],
                    approval_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a distribution pack for a source.

        Args:
            source_id: Source identifier
            source_config: Source configuration from _manifest.json
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
        from .schemas import get_manifest_file
        manifest_path = source_path / get_manifest_file()
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)

        return {
            "success": True,
            "source_id": source_id,
            "manifest_path": str(manifest_path),
            "manifest": manifest,
            "validation": asdict(validation)
        }
