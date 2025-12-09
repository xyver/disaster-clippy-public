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
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, asdict

from .schemas import (
    get_manifest_file, get_metadata_file, get_index_file,
    get_vectors_file, get_backup_manifest_file, validate_source_files,
    html_filename_to_url
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
    language_filtered: int = 0  # Count of articles filtered by language
    resumed: bool = False  # True if resumed from checkpoint
    error: str = ""
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class ValidationIssue:
    """
    A validation issue with an actionable fix hint.

    Provides clear guidance to users on how to fix each issue.
    """
    message: str
    severity: str = "error"  # "error" blocks validity, "warning" does not
    action: str = ""  # e.g., "step2_config", "step3_metadata", "step5_index"
    action_label: str = ""  # Human-readable action, e.g., "Run Generate Metadata"
    step: int = 0  # Wizard step number (1-5) where this can be fixed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message": self.message,
            "severity": self.severity,
            "action": self.action,
            "action_label": self.action_label,
            "step": self.step
        }


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

    # Schema v3 fields (current format - underscore-prefixed JSON files)
    schema_version: int = 3
    has_manifest: bool = False  # _manifest.json
    has_metadata_file: bool = False  # _metadata.json
    has_index_file: bool = False  # _index.json
    has_vectors_file: bool = False  # _vectors.json
    has_backup_manifest: bool = False  # _backup_manifest.json

    # Actionable issues with fix hints
    actionable_issues: List[ValidationIssue] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.warnings is None:
            self.warnings = []
        if self.suggested_tags is None:
            self.suggested_tags = []
        if self.actionable_issues is None:
            self.actionable_issues = []

    def add_issue(self, message: str, action: str = "", action_label: str = "", step: int = 0):
        """Add an issue with optional action hint."""
        self.issues.append(message)
        if action or action_label:
            self.actionable_issues.append(ValidationIssue(
                message=message,
                severity="error",
                action=action,
                action_label=action_label,
                step=step
            ))

    def add_warning(self, message: str, action: str = "", action_label: str = "", step: int = 0):
        """Add a warning with optional action hint."""
        self.warnings.append(message)
        if action or action_label:
            self.actionable_issues.append(ValidationIssue(
                message=message,
                severity="warning",
                action=action,
                action_label=action_label,
                step=step
            ))


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
        "drought": ["drought", "water shortage", "desertification", "arid"],

        # Climate and environment
        "climate": ["climate", "climate change", "global warming", "greenhouse", "carbon", "emission", "ipcc", "sea level", "temperature rise"],
        "environment": ["environment", "environmental", "ecosystem", "biodiversity", "conservation", "pollution", "deforestation"],
        "weather": ["weather", "meteorology", "forecast", "storm", "severe weather", "heat wave", "cold wave"],

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

    # Common English stopwords to filter from term discovery
    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can", "need",
        "this", "that", "these", "those", "it", "its", "they", "them", "their",
        "he", "she", "him", "her", "his", "we", "us", "our", "you", "your",
        "i", "me", "my", "who", "what", "which", "when", "where", "why", "how",
        "all", "each", "every", "both", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "just", "also", "now", "here", "there", "then", "once", "if",
        "about", "after", "before", "above", "below", "between", "into", "through",
        "during", "under", "again", "further", "any", "new", "old", "first",
        "last", "long", "great", "little", "own", "other", "much", "many",
        "over", "out", "up", "down", "off", "well", "back", "even", "still",
        "way", "use", "used", "using", "one", "two", "three", "part", "parts",
        "made", "make", "see", "also", "known", "called", "like", "get", "got",
        "however", "example", "including", "because", "while", "being", "since",
        "based", "within", "without", "although", "either", "another", "per",
        "list", "lists", "article", "articles", "page", "pages", "section",
        "wikipedia", "wikimedia", "category", "categories", "file", "files",
        "index", "main", "content", "contents", "external", "links", "link",
        "references", "reference", "source", "sources", "note", "notes",
        "image", "images", "figure", "figures", "table", "tables", "data",
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

                # Reconstruct URL from filename using centralized function
                url = html_filename_to_url(filename)

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
    # METADATA GENERATION
    # =========================================================================

    def generate_metadata(self, source_id: str, progress_callback: Callable = None,
                          language_filter: str = None, resume: bool = False,
                          cancel_checker: Callable = None) -> Dict[str, Any]:
        """
        Generate metadata for a source from its backup files.

        For HTML sources: Scans HTML files and extracts titles, snippets
        For ZIM sources: Extracts article metadata from ZIM file (with checkpoint support)
        For PDF sources: Uses _collection.json document info

        Args:
            source_id: Source identifier
            progress_callback: Function(current, total, message) for progress
            language_filter: ISO language code to filter (e.g., 'en', 'es').
                           Only applies to ZIM files. None = include all languages.
            resume: If True, attempt to resume from checkpoint (ZIM only)
            cancel_checker: Function that returns True if job was cancelled

        Returns:
            Dict with success status and document count
        """
        source_type = self._detect_source_type(source_id)

        if not source_type:
            return {
                "success": False,
                "error": "Could not detect source type",
                "document_count": 0
            }

        source_path = self.get_source_path(source_id)

        if source_type == "zim":
            return self._generate_zim_metadata(source_id, source_path, progress_callback, language_filter, resume, cancel_checker)
        elif source_type == "html":
            return self._generate_html_metadata(source_id, source_path, progress_callback)
        elif source_type == "pdf":
            return self._generate_pdf_metadata(source_id, source_path, progress_callback)
        else:
            return {
                "success": False,
                "error": f"Unknown source type: {source_type}",
                "document_count": 0
            }

    def _generate_zim_metadata(self, source_id: str, source_path: Path,
                               progress_callback: Callable = None,
                               language_filter: str = None,
                               resume: bool = False,
                               cancel_checker: Callable = None) -> Dict[str, Any]:
        """
        Generate metadata from ZIM file with checkpoint support.

        Checkpoints are saved every 60 seconds OR every 2000 articles.
        The partial work is stored in _jobs/{source_id}_metadata.partial.json

        Args:
            source_id: Source identifier
            source_path: Path to source folder
            progress_callback: Progress reporting function
            language_filter: ISO language code to filter (e.g., 'en')
            resume: If True, attempt to resume from checkpoint
            cancel_checker: Function that returns True if job was cancelled

        Returns:
            Dict with success status and document count
        """
        import time as time_module
        import re
        from collections import Counter
        from .schemas import get_metadata_file
        from .indexer import should_include_article, extract_text_lenient

        # Import checkpoint functions
        try:
            from admin.job_manager import (
                Checkpoint, save_checkpoint, load_checkpoint, delete_checkpoint,
                get_partial_file_path
            )
            checkpointing_available = True
        except ImportError:
            checkpointing_available = False
            print("[metadata] Checkpoint system not available")

        zim_files = list(source_path.glob("*.zim"))
        if not zim_files:
            return {"success": False, "error": "No ZIM file found", "document_count": 0}

        try:
            from zimply_core.zim_core import ZIMFile
        except ImportError:
            return {"success": False, "error": "zimply-core not installed", "document_count": 0}

        zim_path = zim_files[0]
        try:
            zim = ZIMFile(str(zim_path), 'utf-8')
        except Exception as e:
            return {"success": False, "error": f"Failed to open ZIM: {e}", "document_count": 0}

        article_count = zim.header_fields.get('articleCount', 0)

        # Initialize or load from checkpoint
        documents = {}
        word_freq = Counter()  # Track word frequency for term discovery
        start_index = 0
        language_filtered = 0
        checkpoint = None
        resumed = False

        # Build set of existing keywords to exclude from term discovery
        existing_keywords = set()
        for keywords in self.TOPIC_KEYWORDS.values():
            for kw in keywords:
                existing_keywords.update(kw.lower().split())

        if checkpointing_available and resume:
            checkpoint = load_checkpoint(source_id, "metadata")
            if checkpoint:
                # Load partial work
                partial_path = get_partial_file_path(source_id, "metadata")
                if partial_path and partial_path.exists():
                    try:
                        with open(partial_path, 'r', encoding='utf-8') as f:
                            partial_data = json.load(f)
                        documents = partial_data.get("documents", {})
                        language_filtered = partial_data.get("language_filtered", 0)
                        # Restore word frequency from checkpoint if available
                        saved_freq = partial_data.get("word_freq", {})
                        word_freq = Counter(saved_freq)
                        start_index = checkpoint.last_article_index + 1
                        resumed = True
                        print(f"[metadata] Resuming from checkpoint: {len(documents)} docs, starting at article {start_index}")
                    except Exception as e:
                        print(f"[metadata] Error loading partial file: {e}")
                        # Start fresh if partial file is corrupted
                        documents = {}
                        word_freq = Counter()
                        start_index = 0

        if not checkpoint and checkpointing_available:
            # Create new checkpoint
            checkpoint = Checkpoint(
                job_type="metadata",
                source_id=source_id,
                work_range_end=article_count
            )

        processed = len(documents)
        last_checkpoint_time = time_module.time()
        articles_since_checkpoint = 0
        CHECKPOINT_INTERVAL_SECONDS = 60
        CHECKPOINT_INTERVAL_ARTICLES = 2000

        lang_msg = f" (language: {language_filter})" if language_filter else ""
        resume_msg = f" (resumed from {start_index})" if resumed else ""
        if progress_callback:
            progress_callback(start_index, article_count, f"Extracting ZIM metadata{lang_msg}{resume_msg}...")

        if language_filter:
            print(f"Language filter: {language_filter} (only including {language_filter} articles in metadata)")

        for i in range(start_index, article_count):
            try:
                article = zim.get_article_by_id(i)
                if article is None:
                    continue

                url = getattr(article, 'url', '') or ''
                title = getattr(article, 'title', '') or ''
                mimetype = str(getattr(article, 'mimetype', ''))

                # Only process HTML articles
                if 'text/html' not in mimetype:
                    continue

                # Skip navigation/namespace pages
                if url.startswith(('-/', 'X/', 'M/')):
                    continue

                # Apply language filter if specified
                if language_filter and not should_include_article(
                    url, title, language_filter, debug=(language_filtered < 10)
                ):
                    language_filtered += 1
                    continue

                content = article.data
                if isinstance(content, bytes):
                    content = content.decode('utf-8', errors='ignore')

                # Use unified BeautifulSoup extraction for consistency with indexing
                text = extract_text_lenient(content)
                if len(text) < 100:  # Skip very short articles
                    continue

                # Use same ID format as indexer: MD5 hash of source_id:url
                doc_id = hashlib.md5(f"{source_id}:{url}".encode()).hexdigest()
                documents[doc_id] = {
                    "title": title[:200] if title else url[:200],
                    "url": f"/zim/{source_id}/{url}",
                    "snippet": text[:500],
                    "char_count": len(text),
                    "source_id": source_id,
                    "zim_index": i,
                    "zim_url": url
                }
                processed += 1
                articles_since_checkpoint += 1

                # Extract words from title for term discovery
                # Only count words 3+ chars that aren't stopwords or existing keywords
                title_words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
                for word in title_words:
                    if word not in self.STOPWORDS and word not in existing_keywords:
                        word_freq[word] += 1

                # Progress callback
                if progress_callback and processed % 100 == 0:
                    lang_info = f", {language_filtered} filtered" if language_filtered > 0 else ""
                    progress_callback(i, article_count, f"Processed {processed} articles{lang_info}...")

                # Check for cancellation every 100 articles
                if cancel_checker and cancel_checker():
                    print(f"[metadata] Job cancelled by user at article {i}")
                    # Save checkpoint before exiting so work isn't lost
                    if checkpoint and checkpointing_available:
                        checkpoint.last_article_index = i
                        checkpoint.documents_processed = processed
                        checkpoint.progress = int((i / article_count) * 100)
                        save_checkpoint(checkpoint)
                        # Save partial work
                        partial_path = get_partial_file_path(source_id, "metadata")
                        if partial_path:
                            with open(partial_path, 'w', encoding='utf-8') as f:
                                json.dump({
                                    "source_id": source_id,
                                    "documents": documents,
                                    "language_filtered": language_filtered,
                                    "word_freq": dict(word_freq),
                                    "last_article_index": i
                                }, f)
                    zim.close()
                    return {
                        "success": False,
                        "error": "Cancelled by user",
                        "cancelled": True,
                        "document_count": processed
                    }

                # Checkpoint: every 60 seconds OR every 2000 articles
                if checkpointing_available and checkpoint:
                    time_since_checkpoint = time_module.time() - last_checkpoint_time
                    should_checkpoint = (
                        time_since_checkpoint >= CHECKPOINT_INTERVAL_SECONDS or
                        articles_since_checkpoint >= CHECKPOINT_INTERVAL_ARTICLES
                    )

                    if should_checkpoint:
                        # Update checkpoint
                        checkpoint.last_article_index = i
                        checkpoint.documents_processed = processed
                        checkpoint.progress = int((i / article_count) * 100)

                        # Save partial work to file
                        partial_path = get_partial_file_path(source_id, "metadata")
                        if partial_path:
                            checkpoint.partial_file = str(partial_path)
                            try:
                                partial_data = {
                                    "source_id": source_id,
                                    "documents": documents,
                                    "language_filtered": language_filtered,
                                    "word_freq": dict(word_freq),
                                    "last_article_index": i
                                }
                                temp_path = partial_path.with_suffix('.tmp')
                                with open(temp_path, 'w', encoding='utf-8') as f:
                                    json.dump(partial_data, f)
                                temp_path.replace(partial_path)
                            except Exception as e:
                                print(f"[metadata] Error saving partial file: {e}")

                        # Save checkpoint
                        save_checkpoint(checkpoint)
                        last_checkpoint_time = time_module.time()
                        articles_since_checkpoint = 0

            except Exception as e:
                # Track errors in checkpoint
                if checkpoint:
                    checkpoint.errors.append({"article_index": i, "error": str(e)})
                continue

        try:
            zim.close()
        except Exception:
            pass

        # Save final metadata
        metadata_file = source_path / get_metadata_file()

        # Filter discovered terms: only include words appearing 2+ times, sorted by frequency
        discovered_terms = {
            word: count for word, count in word_freq.most_common()
            if count >= 2
        }

        metadata = {
            "source_id": source_id,
            "document_count": len(documents),
            "language_filter": language_filter,
            "language_filtered_count": language_filtered,
            "discovered_terms": discovered_terms,
            "documents": documents
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        # Delete checkpoint on successful completion
        if checkpointing_available:
            delete_checkpoint(source_id, "metadata")

        lang_info = f", {language_filtered} filtered by language" if language_filtered > 0 else ""
        terms_info = f", {len(discovered_terms)} novel terms found" if discovered_terms else ""
        if progress_callback:
            progress_callback(article_count, article_count, f"Metadata generation complete{lang_info}{terms_info}")

        return {
            "success": True,
            "document_count": len(documents),
            "language_filter": language_filter,
            "language_filtered_count": language_filtered,
            "discovered_terms_count": len(discovered_terms),
            "metadata_file": str(metadata_file),
            "resumed": resumed
        }

    def _generate_html_metadata(self, source_id: str, source_path: Path,
                                progress_callback: Callable = None) -> Dict[str, Any]:
        """Generate metadata from HTML files using packager."""
        from .schemas import get_metadata_file, get_backup_manifest_file
        from .packager import generate_metadata_from_html, save_metadata

        pages_folder = source_path / "pages"
        if not pages_folder.exists():
            return {"success": False, "error": "No pages/ folder found", "document_count": 0}

        try:
            metadata = generate_metadata_from_html(str(pages_folder), source_id, save=False)

            if not metadata or not metadata.get("documents"):
                return {"success": False, "error": "No documents found in pages/", "document_count": 0}

            # Save to _metadata.json directly (don't use save_metadata which creates its own path)
            metadata_file = source_path / get_metadata_file()
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            return {
                "success": True,
                "document_count": len(metadata.get("documents", {})),
                "metadata_file": str(metadata_file)
            }
        except Exception as e:
            return {"success": False, "error": str(e), "document_count": 0}

    def _generate_pdf_metadata(self, source_id: str, source_path: Path,
                               progress_callback: Callable = None) -> Dict[str, Any]:
        """Generate metadata from PDF collection."""
        from .schemas import get_metadata_file

        collection_file = source_path / "_collection.json"
        if not collection_file.exists():
            return {"success": False, "error": "No _collection.json found", "document_count": 0}

        try:
            with open(collection_file, 'r', encoding='utf-8') as f:
                collection = json.load(f)

            documents = {}
            source_docs = collection.get("documents", {})

            for doc_id, doc_info in source_docs.items():
                documents[doc_id] = {
                    "title": doc_info.get("title", doc_id),
                    "url": doc_info.get("url", ""),
                    "snippet": doc_info.get("description", "")[:500],
                    "source_id": source_id,
                    "pdf_path": doc_info.get("path", "")
                }

            metadata_file = source_path / get_metadata_file()
            metadata = {
                "source_id": source_id,
                "document_count": len(documents),
                "documents": documents
            }

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)

            return {
                "success": True,
                "document_count": len(documents),
                "metadata_file": str(metadata_file)
            }
        except Exception as e:
            return {"success": False, "error": str(e), "document_count": 0}

    # =========================================================================
    # INDEX OPERATIONS
    # =========================================================================

    def create_index(self, source_id: str, source_type: str = None,
                     limit: int = 1000, skip_existing: bool = True,
                     progress_callback: Callable = None,
                     language_filter: str = None,
                     resume: bool = False,
                     cancel_checker: Callable = None) -> IndexResult:
        """
        Create an index for a source.

        Args:
            source_id: Source identifier
            source_type: One of: html, zim, pdf (auto-detected if not provided)
            limit: Maximum items to index
            skip_existing: Skip already-indexed items
            progress_callback: Function(current, total, message)
            language_filter: ISO language code to filter (e.g., 'en', 'es').
                           Only applies to ZIM files. None = index all languages.
            resume: If True, attempt to resume from checkpoint (ZIM only)
            cancel_checker: Function that returns True if job was cancelled

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

        # For ZIM, skip_existing=False means clear_existing=True (force reindex)
        clear_existing = not skip_existing

        if source_type == "html":
            return self._index_html(source_id, limit, skip_existing, progress_callback)
        elif source_type == "zim":
            return self._index_zim(source_id, limit, progress_callback, language_filter,
                                   clear_existing, resume, cancel_checker)
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
                   progress_callback: Callable,
                   language_filter: str = None,
                   clear_existing: bool = False,
                   resume: bool = False,
                   cancel_checker: Callable = None) -> IndexResult:
        """Index a ZIM file with optional checkpoint support"""
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
                backup_folder=str(self.get_source_path(source_id)),
                language_filter=language_filter,
                clear_existing=clear_existing,
                resume=resume,
                cancel_checker=cancel_checker
            )

            return IndexResult(
                success=result.get("success", False),
                source_id=source_id,
                indexed_count=result.get("indexed_count", 0),
                total_chars=result.get("total_chars", 0),
                language_filtered=result.get("language_filtered", 0),
                resumed=result.get("resumed", False),
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

        Uses validate_source_files() from schemas.py for core file checks,
        then adds additional validation for license, tags, and content.

        Checks:
        - Has backup files (HTML/ZIM/PDF)
        - Has required schema files (manifest, metadata/index, vectors)
        - Has embeddings with content
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

        # Load manifest file if source_config not provided
        manifest_path = source_path / get_manifest_file()
        if source_config is None and manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    source_config = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load manifest for validation: {e}")
                source_config = {}
        else:
            source_config = source_config or {}

        # Normalize file names first (e.g., rename ZIM files to standard format)
        # This runs early so validation sees the correct file names
        normalize_result = self.normalize_filenames(source_id)
        if normalize_result["renamed"]:
            for r in normalize_result["renamed"]:
                print(f"Normalized: {r['old']} -> {r['new']}")

        # Use validate_source_files() from schemas.py for core file checks
        schema_validation = validate_source_files(str(source_path), source_id)

        # Map schema validation results to our ValidationResult
        result.has_manifest = schema_validation["has_manifest"]
        result.has_metadata_file = schema_validation["has_metadata"]
        result.has_index_file = schema_validation["has_index"]
        result.has_vectors_file = schema_validation["has_vectors"]
        result.has_backup_manifest = schema_validation["has_backup_manifest"]
        result.has_backup = schema_validation["has_backup"]

        # Add schema validation issues with actionable hints
        for issue in schema_validation.get("issues", []):
            # Map schema issues to specific steps
            if "manifest" in issue.lower():
                result.add_issue(issue, action="step2_config", action_label="Configure Source", step=2)
            elif "metadata" in issue.lower():
                result.add_issue(issue, action="step3_metadata", action_label="Generate Metadata", step=3)
            elif "vectors" in issue.lower():
                result.add_issue(issue, action="step5_index", action_label="Create Index", step=5)
            else:
                result.issues.append(issue)
            result.is_valid = False

        # Set legacy fields for compatibility
        result.has_metadata = result.has_metadata_file or result.has_index_file
        result.has_index = result.has_index_file

        # Check backup exists (additional check using our detection)
        source_type = self._detect_source_type(source_id)
        if not source_type and not result.has_backup:
            if "No backup found" not in str(result.issues):
                result.add_issue(
                    "No backup found (no HTML, ZIM, or PDF files)",
                    action="step1_backup",
                    action_label="Add Backup Files",
                    step=1
                )
            result.is_valid = False
        elif source_type:
            result.has_backup = True

        # Verify metadata file has content (not just exists)
        metadata_doc_count = 0
        metadata_file = source_path / get_metadata_file()
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    docs_data = json.load(f)
                metadata_doc_count = len(docs_data.get("documents", {}))
                if metadata_doc_count == 0:
                    result.has_metadata = False
                    result.has_metadata_file = False
                    result.add_issue(
                        "Metadata file exists but is empty",
                        action="step3_metadata",
                        action_label="Generate Metadata",
                        step=3
                    )
                    result.is_valid = False
            except Exception:
                result.has_metadata = False
                result.has_metadata_file = False
                result.add_issue(
                    "Metadata file exists but could not be read",
                    action="step3_metadata",
                    action_label="Generate Metadata",
                    step=3
                )
                result.is_valid = False

        # Check vectors file has content
        vectors_doc_count = 0
        vectors_file = source_path / get_vectors_file()
        if vectors_file.exists():
            result.has_embeddings = True
            try:
                with open(vectors_file, 'r', encoding='utf-8') as f:
                    embed_data = json.load(f)
                vectors_doc_count = len(embed_data.get("vectors", {}))
                if vectors_doc_count == 0:
                    vectors_doc_count = embed_data.get("document_count", 0)
                if vectors_doc_count == 0:
                    result.has_embeddings = False
                    result.has_vectors_file = False
                    result.add_issue(
                        "Vectors file exists but is empty",
                        action="step5_index",
                        action_label="Create Index",
                        step=5
                    )
                    result.is_valid = False
            except Exception:
                result.has_embeddings = False
                result.has_vectors_file = False
                result.add_issue(
                    "Vectors file exists but could not be read",
                    action="step5_index",
                    action_label="Create Index",
                    step=5
                )
                result.is_valid = False
        else:
            result.has_embeddings = False

        # Check for metadata/index count mismatch
        if metadata_doc_count > 0 and vectors_doc_count > 0:
            if vectors_doc_count < metadata_doc_count:
                coverage = (vectors_doc_count / metadata_doc_count) * 100
                if coverage < 90:  # Less than 90% coverage is a warning
                    result.add_warning(
                        f"Index incomplete: {vectors_doc_count:,} indexed of {metadata_doc_count:,} documents ({coverage:.1f}%)",
                        action="step5_force_reindex",
                        action_label="Force Re-index with higher limit",
                        step=5
                    )
                    if coverage < 50:  # Less than 50% is an issue
                        result.add_issue(
                            f"Index severely incomplete: only {coverage:.1f}% of documents indexed",
                            action="step5_force_reindex",
                            action_label="Force Re-index All Documents",
                            step=5
                        )
                        result.is_valid = False

        # Check license
        license_val = source_config.get("license", "")
        if license_val and license_val.lower() not in ["unknown", ""]:
            result.has_license = True
            result.license_verified = source_config.get("license_verified", False)
            if not result.license_verified:
                result.add_warning(
                    "License not verified",
                    action="step2_verify_license",
                    action_label="Verify License",
                    step=2
                )
        else:
            result.has_license = False
            result.add_issue(
                "No license specified",
                action="step2_config",
                action_label="Set License in Configure Source",
                step=2
            )
            result.is_valid = False

            # Try auto-detection
            detected = self.detect_license(source_id)
            if detected:
                result.detected_license = detected
                result.add_warning(
                    f"Auto-detected license: {detected} (needs verification)",
                    action="step2_apply_license",
                    action_label=f"Apply detected license: {detected}",
                    step=2
                )

        # Check tags - check source_config first, then _collection.json for PDF sources
        tags = source_config.get("tags", []) or source_config.get("topics", [])
        print(f"[validate] Tags from config: {tags} (type: {type(tags).__name__})")

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
            print(f"[validate] Source already has tags: {tags}")
        else:
            result.has_tags = False
            result.add_warning(
                "No tags specified",
                action="step2_add_tags",
                action_label="Add Tags",
                step=2
            )
            print(f"[validate] No tags found, calling suggest_tags()")

            # Suggest tags
            suggested = self.suggest_tags(source_id)
            print(f"[validate] suggest_tags returned: {suggested}")
            if suggested:
                result.suggested_tags = suggested
                result.add_warning(
                    f"Suggested tags: {', '.join(suggested)}",
                    action="step2_apply_tags",
                    action_label="Apply Suggested Tags",
                    step=2
                )

        # Current schema version (v3 uses underscore-prefixed files)
        result.schema_version = 3

        # Determine if production ready (all critical checks pass)
        # Using schema validation: manifest + (metadata or index) + vectors
        result.production_ready = (
            result.has_backup and
            result.has_manifest and
            (result.has_metadata_file or result.has_index_file) and
            result.has_vectors_file and
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

        SCHEMA v3 FILES:
        - _manifest.json (source-level metadata)
        - _metadata.json (document metadata)
        - _vectors.json (vectors only, no content duplication)

        RECOMMENDED (warnings only):
        - License is verified
        - Has tags for categorization

        Args:
            source_id: Source identifier
            source_config: Source configuration from _manifest.json
            require_v2: Deprecated parameter, kept for compatibility

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

        # Check metadata file first
        metadata_file = source_path / get_metadata_file()
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
        manifest = source_path / get_backup_manifest_file()
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
        print(f"[suggest_tags] Called for source: {source_id}")
        print(f"[suggest_tags] Backup path: {self.backup_path}")
        print(f"[suggest_tags] Looking for metadata in: {source_path}")
        suggested = set()
        tag_scores = {}  # Track keyword match frequency for ranking

        # For PDF sources, check _collection.json topics first
        collection_file = source_path / "_collection.json"
        if collection_file.exists():
            print(f"[suggest_tags] Found _collection.json")
            try:
                with open(collection_file, 'r', encoding='utf-8') as f:
                    collection_data = json.load(f)
                topics = collection_data.get("collection", {}).get("topics", [])
                for topic in topics:
                    suggested.add(topic)
                print(f"[suggest_tags] Got {len(topics)} topics from collection")
            except Exception as e:
                print(f"[suggest_tags] Error reading collection: {e}")

        # Check metadata for titles
        metadata_file = source_path / get_metadata_file()
        print(f"[suggest_tags] Checking for: {metadata_file}")

        if metadata_file.exists():
            print(f"[suggest_tags] Found metadata file: {metadata_file}")
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                documents = metadata.get("documents", {})
                doc_count = len(documents) if isinstance(documents, dict) else 0
                print(f"[suggest_tags] Metadata has {doc_count} documents")

                if not documents:
                    print(f"[suggest_tags] Metadata file exists but has no documents: {metadata_file}")
                else:
                    # Scan titles and snippets for keyword frequency
                    all_text = ""
                    sample_titles = []
                    for doc_id, doc_info in documents.items():
                        title = doc_info.get("title", "")
                        snippet = doc_info.get("snippet", "")
                        all_text += " " + title.lower() + " " + snippet.lower()
                        if len(sample_titles) < 5:
                            sample_titles.append(title[:50])

                    print(f"[suggest_tags] Sample titles: {sample_titles}")

                    # Count keyword matches for each tag (frequency-based ranking)
                    for tag, keywords in self.TOPIC_KEYWORDS.items():
                        score = 0
                        for keyword in keywords:
                            # Count occurrences of this keyword
                            count = all_text.count(keyword)
                            if count > 0:
                                score += count
                        if score > 0:
                            tag_scores[tag] = score
                            suggested.add(tag)

                    # Log top scores
                    print(f"[suggest_tags] Tag scores (top 10): {dict(sorted(tag_scores.items(), key=lambda x: x[1], reverse=True)[:10])}")
                    print(f"[suggest_tags] Found {len(suggested)} suggested tags from content")

            except Exception as e:
                print(f"[suggest_tags] Error reading metadata: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[suggest_tags] Metadata file not found at {metadata_file}")
            # List what files ARE in the source path
            if source_path.exists():
                files = [f.name for f in source_path.iterdir() if f.is_file()]
                print(f"[suggest_tags] Files in {source_path}: {files[:10]}")

        # Check source_id for priority tags (these should always be included)
        source_lower = source_id.lower()
        priority_tags = set()
        for tag, keywords in self.TOPIC_KEYWORDS.items():
            for keyword in keywords:
                if keyword in source_lower:
                    priority_tags.add(tag)
                    break

        # Combine: priority tags first, then highest-scoring content tags
        # Priority tags from source_id always included at the front
        if tag_scores:
            # Use frequency-ranked tags from content scan
            sorted_content_tags = sorted(
                [t for t in tag_scores.keys() if t not in priority_tags],
                key=lambda t: tag_scores[t],
                reverse=True
            )
            content_tags = sorted_content_tags
        else:
            # Fallback to unordered set if no content was scanned
            content_tags = [t for t in suggested if t not in priority_tags]

        all_tags = list(priority_tags) + content_tags

        print(f"[suggest_tags] Priority tags from source_id: {priority_tags}")
        print(f"[suggest_tags] Total tags: {len(all_tags)}, returning top 10")

        return all_tags[:10]  # Return top 10 (priority + highest-scoring content)

    def analyze_metadata_for_tags(self, source_id: str, update_metadata: bool = False) -> Dict[str, Any]:
        """
        Analyze existing metadata to discover novel terms without regenerating.

        This is a quick alternative to regenerating metadata when you just want
        to discover terms from an existing source.

        Args:
            source_id: Source identifier
            update_metadata: If True, save discovered_terms back to metadata file

        Returns:
            Dict with:
                - discovered_terms: dict of {term: count} sorted by frequency
                - total_terms: number of unique terms found
                - updated_metadata: bool indicating if file was updated
        """
        import re
        from collections import Counter
        from .schemas import get_metadata_file, normalize_tag

        source_path = self.get_source_path(source_id)
        metadata_file = source_path / get_metadata_file()

        if not metadata_file.exists():
            return {
                "success": False,
                "error": f"Metadata file not found for {source_id}",
                "discovered_terms": {},
                "total_terms": 0
            }

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Check if already has discovered_terms
            if "discovered_terms" in metadata and not update_metadata:
                return {
                    "success": True,
                    "discovered_terms": metadata["discovered_terms"],
                    "total_terms": len(metadata["discovered_terms"]),
                    "already_analyzed": True,
                    "updated_metadata": False
                }

            # Build existing keywords set
            existing_keywords = set()
            for keywords in self.TOPIC_KEYWORDS.values():
                for kw in keywords:
                    existing_keywords.update(kw.lower().split())

            # Scan document titles
            word_freq = Counter()
            documents = metadata.get("documents", {})

            for doc_id, doc_info in documents.items():
                title = doc_info.get("title", "")
                # Extract words 3+ chars, not in stopwords or existing keywords
                title_words = re.findall(r'\b[a-zA-Z]{3,}\b', title.lower())
                for word in title_words:
                    # Normalize the word (plurals, verb forms, synonyms)
                    normalized = normalize_tag(word)
                    if normalized not in self.STOPWORDS and normalized not in existing_keywords:
                        word_freq[normalized] += 1

            # Filter to terms appearing 2+ times
            discovered_terms = {
                word: count for word, count in word_freq.most_common()
                if count >= 2
            }

            # Optionally update the metadata file
            updated = False
            if update_metadata:
                metadata["discovered_terms"] = discovered_terms
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
                updated = True

            return {
                "success": True,
                "discovered_terms": discovered_terms,
                "total_terms": len(discovered_terms),
                "documents_scanned": len(documents),
                "updated_metadata": updated
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "discovered_terms": {},
                "total_terms": 0
            }

    def discover_novel_used_tags(self, source_ids: List[str] = None) -> Dict[str, Any]:
        """
        Discover tags used by sources that are not in TOPIC_KEYWORDS.

        This helps global admins identify new tags that users have chosen
        which could be added to the canonical TOPIC_KEYWORDS list.

        Args:
            source_ids: List of source IDs to scan. If None, scans all sources
                       in the backup folder.

        Returns:
            Dict with:
                - novel_tags: dict of {tag: [list of source_ids using it]}
                - known_tags: dict of {tag: [list of source_ids using it]}
                - sources_scanned: number of sources checked
                - sources_with_tags: number of sources that had tags
                - report: formatted string report for display
        """
        from .schemas import get_manifest_file

        # Get known tags from TOPIC_KEYWORDS
        known_tag_names = set(self.TOPIC_KEYWORDS.keys())

        # Track tag usage
        novel_tags = {}  # tag -> [source_ids]
        known_tags = {}  # tag -> [source_ids]
        sources_scanned = 0
        sources_with_tags = 0
        errors = []

        # Determine which sources to scan
        if source_ids is None:
            # Scan all sources in backup folder
            if not self.backup_path.exists():
                return {
                    "novel_tags": {},
                    "known_tags": {},
                    "sources_scanned": 0,
                    "sources_with_tags": 0,
                    "report": "Backup folder does not exist",
                    "errors": ["Backup folder does not exist"]
                }

            source_ids = [
                d.name for d in self.backup_path.iterdir()
                if d.is_dir() and not d.name.startswith('.')
            ]

        # Scan each source
        for source_id in source_ids:
            source_path = self.get_source_path(source_id)
            manifest_file = source_path / get_manifest_file()

            if not manifest_file.exists():
                continue

            sources_scanned += 1

            try:
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)

                tags = manifest.get("tags", [])
                if not tags:
                    # Also check "topics" for PDF sources
                    tags = manifest.get("topics", [])

                if tags:
                    sources_with_tags += 1

                    for tag in tags:
                        tag_lower = tag.lower().strip()
                        if not tag_lower:
                            continue

                        if tag_lower in known_tag_names:
                            if tag_lower not in known_tags:
                                known_tags[tag_lower] = []
                            known_tags[tag_lower].append(source_id)
                        else:
                            if tag_lower not in novel_tags:
                                novel_tags[tag_lower] = []
                            novel_tags[tag_lower].append(source_id)

            except Exception as e:
                errors.append(f"{source_id}: {str(e)}")

        # Sort novel tags by usage count (most used first)
        sorted_novel = dict(
            sorted(novel_tags.items(), key=lambda x: len(x[1]), reverse=True)
        )

        # Generate report
        report_lines = []
        report_lines.append("Novel Tags Not in TOPIC_KEYWORDS:")
        report_lines.append("-" * 50)

        if sorted_novel:
            # Find max tag length for alignment
            max_tag_len = max(len(tag) for tag in sorted_novel.keys())

            for tag, sources in sorted_novel.items():
                source_list = ", ".join(sources[:5])
                if len(sources) > 5:
                    source_list += f", ... (+{len(sources) - 5} more)"
                report_lines.append(
                    f"{tag:<{max_tag_len}}  (used by {len(sources)} source(s): {source_list})"
                )
        else:
            report_lines.append("No novel tags found - all tags are in TOPIC_KEYWORDS")

        report_lines.append("")
        report_lines.append(f"Sources scanned: {sources_scanned}")
        report_lines.append(f"Sources with tags: {sources_with_tags}")
        report_lines.append(f"Novel tags found: {len(sorted_novel)}")
        report_lines.append(f"Known tags in use: {len(known_tags)}")

        if errors:
            report_lines.append("")
            report_lines.append(f"Errors ({len(errors)}):")
            for err in errors[:5]:
                report_lines.append(f"  - {err}")

        return {
            "novel_tags": sorted_novel,
            "known_tags": known_tags,
            "sources_scanned": sources_scanned,
            "sources_with_tags": sources_with_tags,
            "report": "\n".join(report_lines),
            "errors": errors
        }

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


# =============================================================================
# CLOUD SOURCE MANAGEMENT - Install sources from R2
# =============================================================================

MASTER_METADATA_KEY = "backups/_master.json"


def list_cloud_sources() -> Dict[str, Any]:
    """
    List available sources from R2 cloud storage.

    Reads the master metadata file (backups/_master.json) for efficiency.
    Falls back to scanning if master file doesn't exist.

    Returns:
        Dict with:
        - sources: List of source info dicts
        - total: Total source count
        - error: Error message if any
    """
    from .cloud.r2 import get_backups_storage
    import tempfile

    result = {"sources": [], "total": 0, "error": None}

    try:
        r2 = get_backups_storage()
        if not r2.is_configured():
            result["error"] = "R2 storage not configured"
            return result

        # Try to download master metadata file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if r2.download_file(MASTER_METADATA_KEY, tmp_path):
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    master_data = json.load(f)

                # Convert master data to source list format
                sources = []
                for source_id, source_info in master_data.get("sources", {}).items():
                    sources.append({
                        "source_id": source_id,
                        "name": source_info.get("name", source_id),
                        "description": source_info.get("description", ""),
                        "license": source_info.get("license", "Unknown"),
                        "license_verified": source_info.get("license_verified", False),
                        "tags": source_info.get("tags", []),
                        "document_count": source_info.get("total_docs", source_info.get("count", 0)),
                        "total_docs": source_info.get("total_docs", source_info.get("count", 0)),
                        "total_chars": source_info.get("total_chars", 0),
                        "version": source_info.get("version", "1.0.0"),
                        "has_vectors": source_info.get("has_vectors", True),
                        "has_backup": source_info.get("has_backup", False),
                        "backup_type": "cloud",
                        "total_size_bytes": source_info.get("total_size_bytes", 0),
                        "total_size_mb": source_info.get("total_size_mb", 0),
                        "last_updated": source_info.get("last_updated", "")
                    })

                result["sources"] = sources
                result["total"] = len(sources)
                return result
            else:
                print("[list_cloud_sources] Master metadata not found, falling back to scan")
        finally:
            import os as os_module
            if os.path.exists(tmp_path):
                os_module.unlink(tmp_path)

        # Fallback: scan for _manifest.json files (slower but works without master file)
        result = _scan_cloud_sources_fallback(r2)

    except Exception as e:
        result["error"] = str(e)

    return result


def _scan_cloud_sources_fallback(r2) -> Dict[str, Any]:
    """Fallback method: scan R2 for _manifest.json files (slower)."""
    import tempfile

    result = {"sources": [], "total": 0, "error": None}
    files = r2.list_files("backups/")
    manifest_files = [f for f in files if f["key"].endswith("/_manifest.json")]

    sources = []
    for manifest_info in manifest_files:
        key = manifest_info["key"]
        parts = key.split("/")
        if len(parts) >= 3:
            source_id = parts[1]

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                tmp_path = tmp.name

            if r2.download_file(key, tmp_path):
                try:
                    with open(tmp_path, 'r', encoding='utf-8') as f:
                        manifest = json.load(f)

                    sources.append({
                        "source_id": source_id,
                        "name": manifest.get("name", source_id),
                        "description": manifest.get("description", ""),
                        "license": manifest.get("license", "Unknown"),
                        "license_verified": manifest.get("license_verified", False),
                        "tags": manifest.get("tags", []),
                        "document_count": manifest.get("total_docs", 0),
                        "total_docs": manifest.get("total_docs", 0),
                        "version": manifest.get("version", "1.0.0"),
                        "has_vectors": manifest.get("has_vectors", False),
                        "has_backup": manifest.get("has_backup", False),
                        "backup_type": "cloud",
                        "last_updated": manifest.get("last_indexed", "")
                    })
                except Exception as e:
                    print(f"[scan_fallback] Error parsing manifest for {source_id}: {e}")
                finally:
                    import os as os_module
                    os_module.unlink(tmp_path)

    result["sources"] = sources
    result["total"] = len(sources)
    return result


def update_cloud_master_metadata(source_id: str, source_info: Dict[str, Any]) -> bool:
    """
    Update the master metadata file in R2 with a new/updated source.

    Called after uploading a source pack to R2.

    Args:
        source_id: Source ID being added/updated
        source_info: Source metadata to add

    Returns:
        True if successful
    """
    from .cloud.r2 import get_backups_storage
    import tempfile

    try:
        r2 = get_backups_storage()
        if not r2.is_configured():
            print("[update_master] R2 not configured")
            return False

        # Download existing master or start fresh
        master_data = {"sources": {}, "last_updated": ""}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            if r2.download_file(MASTER_METADATA_KEY, tmp_path):
                with open(tmp_path, 'r', encoding='utf-8') as f:
                    master_data = json.load(f)
        except Exception:
            pass  # Start with empty master

        # Update with new source info
        master_data["sources"][source_id] = {
            "name": source_info.get("name", source_id),
            "description": source_info.get("description", ""),
            "license": source_info.get("license", "Unknown"),
            "license_verified": source_info.get("license_verified", False),
            "tags": source_info.get("tags", []),
            "total_docs": source_info.get("total_docs", 0),
            "count": source_info.get("total_docs", 0),  # Alias for compatibility
            "total_chars": source_info.get("total_chars", 0),
            "version": source_info.get("version", "1.0.0"),
            "has_vectors": source_info.get("has_vectors", True),
            "has_backup": source_info.get("has_backup", False),
            "total_size_bytes": source_info.get("total_size_bytes", 0),
            "total_size_mb": source_info.get("total_size_mb", 0),
            "last_updated": datetime.now().isoformat()
        }
        master_data["last_updated"] = datetime.now().isoformat()

        # Write updated master
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(master_data, f, indent=2)

        success = r2.upload_file(tmp_path, MASTER_METADATA_KEY)
        if success:
            print(f"[update_master] Updated master metadata with {source_id}")
        return success

    except Exception as e:
        print(f"[update_master] Error: {e}")
        return False


def download_source_pack(
    source_id: str,
    dest_path: Optional[Path] = None,
    include_backup: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> Dict[str, Any]:
    """
    Download a source pack from R2 cloud storage.

    Args:
        source_id: Source ID to download
        dest_path: Destination path (defaults to BACKUP_PATH/{source_id})
        include_backup: Whether to download backup files (pages/, ZIM, etc.)
        progress_callback: Optional callback(filename, current, total)

    Returns:
        Dict with:
        - success: bool
        - source_id: str
        - dest_path: str
        - files_downloaded: list
        - error: str if any
    """
    from .cloud.r2 import get_backups_storage

    result = {
        "success": False,
        "source_id": source_id,
        "dest_path": "",
        "files_downloaded": [],
        "error": None
    }

    try:
        r2 = get_backups_storage()
        if not r2.is_configured():
            result["error"] = "R2 storage not configured"
            return result

        # Determine destination path
        if dest_path is None:
            backup_path = os.getenv("BACKUP_PATH", "")
            if not backup_path:
                result["error"] = "BACKUP_PATH not configured"
                return result
            dest_path = Path(backup_path) / source_id

        dest_path = Path(dest_path)
        dest_path.mkdir(parents=True, exist_ok=True)
        result["dest_path"] = str(dest_path)

        # List files in source folder
        source_prefix = f"backups/{source_id}/"
        files = r2.list_files(source_prefix)

        if not files:
            result["error"] = f"Source '{source_id}' not found in cloud storage"
            return result

        # Filter files to download
        files_to_download = []
        for f in files:
            filename = f["key"].replace(source_prefix, "")

            # Always download schema files
            if filename.startswith("_"):
                files_to_download.append(f)
            # Optionally download backup files
            elif include_backup:
                files_to_download.append(f)

        total_files = len(files_to_download)
        total_bytes = sum(f.get("size", 0) for f in files_to_download)
        downloaded = []
        skipped = []
        bytes_downloaded = 0

        for i, file_info in enumerate(files_to_download):
            key = file_info["key"]
            filename = key.replace(source_prefix, "")
            local_path = dest_path / filename
            remote_size = file_info.get("size", 0)

            # Create subdirectories if needed
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Smart download: skip if file exists with matching size
            if local_path.exists():
                local_size = local_path.stat().st_size
                if local_size == remote_size and remote_size > 0:
                    skipped.append(filename)
                    bytes_downloaded += remote_size
                    if progress_callback:
                        progress_callback(f"Skipped {filename} (exists)", i + 1, total_files)
                    continue

            if progress_callback:
                size_mb = remote_size / (1024*1024) if remote_size else 0
                progress_callback(f"Downloading {filename} ({size_mb:.1f} MB)", i + 1, total_files)

            if r2.download_file(key, str(local_path)):
                downloaded.append(filename)
                bytes_downloaded += remote_size
            else:
                print(f"[download_source_pack] Failed to download: {key}")

        result["files_skipped"] = skipped
        result["total_size_mb"] = round(total_bytes / (1024*1024), 2)

        result["files_downloaded"] = downloaded
        # Success if we downloaded or skipped files (skipped = already had them)
        result["success"] = len(downloaded) > 0 or len(skipped) > 0

    except Exception as e:
        result["error"] = str(e)

    return result


def install_source_to_chromadb(
    source_id: str,
    source_path: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, Any]:
    """
    Import vectors from a downloaded source pack into local ChromaDB.

    Reads _vectors.json and _index.json to add documents to ChromaDB.

    Args:
        source_id: Source ID to install
        source_path: Path to source folder (defaults to BACKUP_PATH/{source_id})
        progress_callback: Optional callback(current, total)

    Returns:
        Dict with:
        - success: bool
        - source_id: str
        - documents_added: int
        - error: str if any
    """
    from .vectordb import get_vector_store

    result = {
        "success": False,
        "source_id": source_id,
        "documents_added": 0,
        "error": None
    }

    try:
        # Determine source path
        if source_path is None:
            backup_path = os.getenv("BACKUP_PATH", "")
            if not backup_path:
                result["error"] = "BACKUP_PATH not configured"
                return result
            source_path = Path(backup_path) / source_id

        source_path = Path(source_path)

        # Check required files exist
        vectors_file = source_path / get_vectors_file()
        index_file = source_path / get_index_file()

        if not vectors_file.exists():
            result["error"] = f"Vectors file not found: {vectors_file}"
            return result

        if not index_file.exists():
            result["error"] = f"Index file not found: {index_file}"
            return result

        # Load vectors and index
        print(f"[install_source] Loading vectors from {vectors_file}")
        with open(vectors_file, 'r', encoding='utf-8') as f:
            vectors_data = json.load(f)

        print(f"[install_source] Loading index from {index_file}")
        with open(index_file, 'r', encoding='utf-8') as f:
            index_data = json.load(f)

        vectors = vectors_data.get("vectors", {})
        documents = index_data.get("documents", {})

        if not vectors:
            result["error"] = "No vectors found in vectors file"
            return result

        print(f"[install_source] Found {len(vectors)} vectors, {len(documents)} documents")

        # Get local ChromaDB store (force local mode)
        store = get_vector_store(mode="local")

        # Prepare documents for import
        docs_to_add = []
        embeddings = []

        total = len(vectors)
        for i, (doc_id, vector) in enumerate(vectors.items()):
            doc_info = documents.get(doc_id, {})

            if not doc_info:
                print(f"[install_source] Warning: No document info for {doc_id}")
                continue

            doc = {
                "id": doc_id,
                "content": doc_info.get("content", ""),
                "title": doc_info.get("title", "Unknown"),
                "url": doc_info.get("url", ""),
                "source": source_id,
                "categories": doc_info.get("categories", []),
                "doc_type": doc_info.get("doc_type", "article"),
                "content_hash": doc_info.get("content_hash", ""),
                "scraped_at": doc_info.get("scraped_at", "")
            }

            docs_to_add.append(doc)
            embeddings.append(vector)

            if progress_callback and (i + 1) % 100 == 0:
                progress_callback(i + 1, total)

        # Add to ChromaDB
        if docs_to_add:
            print(f"[install_source] Adding {len(docs_to_add)} documents to ChromaDB")
            added = store.add_documents(docs_to_add, embeddings=embeddings)
            result["documents_added"] = added
            result["success"] = True
            print(f"[install_source] Successfully added {added} documents")

    except Exception as e:
        import traceback
        traceback.print_exc()
        result["error"] = str(e)

    return result


def install_source_from_cloud(
    source_id: str,
    include_backup: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None
) -> Dict[str, Any]:
    """
    Full pipeline: Download source pack from R2 and install to local ChromaDB.

    Args:
        source_id: Source ID to install
        include_backup: Whether to download backup files
        progress_callback: Optional callback(stage, current, total)

    Returns:
        Dict with:
        - success: bool
        - source_id: str
        - download_result: dict
        - install_result: dict
        - error: str if any
    """
    result = {
        "success": False,
        "source_id": source_id,
        "download_result": None,
        "install_result": None,
        "error": None
    }

    # Stage 1: Download
    def download_progress(filename, current, total):
        if progress_callback:
            progress_callback(f"Downloading {filename}", current, total)

    download_result = download_source_pack(
        source_id=source_id,
        include_backup=include_backup,
        progress_callback=download_progress
    )
    result["download_result"] = download_result

    if not download_result["success"]:
        result["error"] = download_result.get("error", "Download failed")
        return result

    # Stage 2: Install to ChromaDB
    def install_progress(current, total):
        if progress_callback:
            progress_callback("Installing vectors", current, total)

    install_result = install_source_to_chromadb(
        source_id=source_id,
        source_path=Path(download_result["dest_path"]),
        progress_callback=install_progress
    )
    result["install_result"] = install_result

    if not install_result["success"]:
        result["error"] = install_result.get("error", "Install failed")
        return result

    result["success"] = True
    return result
