"""
Source Pack Registry

Manages the catalog of available source packs, their health status,
and determines which packs are ready for distribution.

Uses _manifest.json for source identity.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

from .schemas import get_manifest_file


class PackTier(Enum):
    """Quality tiers for source packs"""
    PERSONAL = "personal"      # Private, any state
    COMMUNITY = "community"    # Public, 80%+ indexed, licenses reported
    OFFICIAL = "official"      # Featured, 100% verified, backups complete


@dataclass
class SourcePack:
    """Represents a downloadable source pack"""
    source_id: str
    name: str
    description: str
    base_url: str
    license: str
    license_url: str
    license_verified: bool
    document_count: int
    total_chars: int
    topics: List[str]
    tags: List[str]
    tier: PackTier
    last_updated: str

    # Health indicators
    has_backup: bool = False
    backup_type: Optional[str] = None  # "zim", "html", "pdf"
    backup_size_mb: float = 0.0
    backup_downloadable: bool = False  # True if server can serve the backup

    # Completeness scores
    index_completeness: float = 1.0  # Percentage of pages indexed
    metadata_completeness: float = 1.0  # Percentage with full metadata

    # Download options available
    can_download_metadata: bool = True  # Metadata is always available
    can_download_vectors: bool = False  # Future: vector export
    can_rescrape: bool = True  # User can re-scrape using source config

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        d = asdict(self)
        d['tier'] = self.tier.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SourcePack':
        """Create from dictionary"""
        data = data.copy()
        if isinstance(data.get('tier'), str):
            data['tier'] = PackTier(data['tier'])
        return cls(**data)


class SourcePackRegistry:
    """
    Manages the catalog of source packs available for download.

    Discovers sources from:
    - _manifest.json files in backup folder
    - _master.json in backup folder (document counts, topics)
    - Backup folder scanning (for offline availability)
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or Path(__file__).parent.parent
        self.backup_folder = self._get_backup_folder()
        self.metadata_path = self._get_metadata_path()

    def _get_backup_folder(self) -> str:
        """Get backup folder from local_config or BACKUP_PATH env var"""
        import os
        # Try local_config first (user's GUI setting)
        try:
            from admin.local_config import get_local_config
            config = get_local_config()
            backup_folder = config.get_backup_folder()
            if backup_folder:
                return backup_folder
        except ImportError:
            pass
        return os.getenv("BACKUP_PATH", "")

    def _get_metadata_path(self) -> Path:
        """Get path to _master.json in backup folder"""
        if self.backup_folder:
            return Path(self.backup_folder) / "_master.json"
        raise ValueError("No backup folder configured")

    def _load_sources_config(self) -> Dict[str, Any]:
        """Load source definitions by discovering _manifest.json files in backup folder"""
        backup_path = self._get_backup_folder()
        sources = {}

        if backup_path:
            backup_folder = Path(backup_path)
            if backup_folder.exists():
                for source_folder in backup_folder.iterdir():
                    if source_folder.is_dir():
                        source_id = source_folder.name
                        # Check for _manifest.json
                        source_file = source_folder / get_manifest_file()
                        if source_file.exists():
                            try:
                                with open(source_file, 'r', encoding='utf-8') as f:
                                    sources[source_id] = json.load(f)
                            except Exception:
                                pass

        return {"sources": sources}

    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata master index"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"sources": {}}

    def _check_backup_status(self, source_id: str, source_config: Dict) -> tuple:
        """
        Check if backup files exist for a source.
        Returns (has_backup, backup_type, size_mb, backup_available_locally)

        Note: Backups may exist but not be available for download from a remote server.
        This checks LOCAL availability only.
        """
        backup_path = self._get_backup_folder()
        if not backup_path:
            return False, None, 0.0

        backup_dir = Path(backup_path)
        if not backup_dir.exists():
            return False, None, 0.0

        # Check for ZIM file
        zim_patterns = [
            backup_dir / "zim" / f"{source_id}.zim",
            backup_dir / "ZIM" / f"{source_id}.zim",
            backup_dir / f"{source_id}.zim",
        ]
        for zim_path in zim_patterns:
            if zim_path.exists():
                size_mb = zim_path.stat().st_size / (1024 * 1024)
                return True, "zim", round(size_mb, 2)

        # Check for HTML folder
        html_patterns = [
            backup_dir / "html" / source_id,
            backup_dir / "HTML" / source_id,
            backup_dir / source_id,
        ]
        for html_path in html_patterns:
            if html_path.exists() and html_path.is_dir():
                # Calculate folder size
                try:
                    total_size = sum(
                        f.stat().st_size for f in html_path.rglob('*') if f.is_file()
                    )
                    size_mb = total_size / (1024 * 1024)
                    return True, "html", round(size_mb, 2)
                except (PermissionError, OSError):
                    return True, "html", 0.0

        return False, None, 0.0

    def _determine_tier(self, source_config: Dict, has_backup: bool) -> PackTier:
        """
        Determine the quality tier for a source based on completeness.

        Official: license_verified in _manifest.json (backup is bonus, not required)
        Community: license reported but not verified
        Personal: No license info
        """
        license_verified = source_config.get("license_verified", False)
        has_license = source_config.get("license", "Unknown") != "Unknown"

        if license_verified:
            # Verified license = OFFICIAL (backup availability is separate)
            return PackTier.OFFICIAL
        elif has_license:
            # Has license but not verified = COMMUNITY
            return PackTier.COMMUNITY
        else:
            # No license = PERSONAL (not downloadable)
            return PackTier.PERSONAL

    def get_all_packs(self) -> List[SourcePack]:
        """
        Get all source packs (all tiers).
        """
        sources_config = self._load_sources_config()
        metadata = self._load_metadata()

        packs = []

        # Build packs from sources that are in BOTH config and metadata
        config_sources = sources_config.get("sources", {})
        metadata_sources = metadata.get("sources", {})

        for source_id, config in config_sources.items():
            meta = metadata_sources.get(source_id, {})

            if not meta:
                # Source is in config but not indexed yet
                continue

            has_backup, backup_type, backup_size = self._check_backup_status(source_id, config)
            tier = self._determine_tier(config, has_backup)

            pack = SourcePack(
                source_id=source_id,
                name=config.get("name", source_id),
                description=config.get("notes", ""),
                base_url=config.get("base_url", ""),
                license=config.get("license", "Unknown"),
                license_url=config.get("license_url", ""),
                license_verified=config.get("license_verified", False),
                document_count=meta.get("count", 0),
                total_chars=meta.get("chars", 0),
                topics=meta.get("topics", []),
                tags=config.get("tags", []),
                tier=tier,
                last_updated=meta.get("last_sync", ""),
                has_backup=has_backup,
                backup_type=backup_type,
                backup_size_mb=backup_size,
                backup_downloadable=has_backup,  # If backup exists locally, it's downloadable
                can_rescrape=bool(config.get("base_url")),  # Can rescrape if we have the URL
            )
            packs.append(pack)

        return packs

    def get_official_packs(self) -> List[SourcePack]:
        """
        Get only OFFICIAL tier packs (ready for distribution).
        These have verified licenses and complete backups.
        """
        return [p for p in self.get_all_packs() if p.tier == PackTier.OFFICIAL]

    def get_downloadable_packs(self) -> List[SourcePack]:
        """
        Get packs that are ready for download (OFFICIAL or COMMUNITY).
        Excludes PERSONAL tier.
        """
        return [
            p for p in self.get_all_packs()
            if p.tier in (PackTier.OFFICIAL, PackTier.COMMUNITY)
        ]

    def get_pack(self, source_id: str) -> Optional[SourcePack]:
        """Get a specific pack by source ID"""
        for pack in self.get_all_packs():
            if pack.source_id == source_id:
                return pack
        return None

    def get_pack_health_report(self) -> Dict[str, Any]:
        """
        Generate a health report for all sources.
        Shows what's complete vs incomplete.
        """
        sources_config = self._load_sources_config()
        metadata = self._load_metadata()

        config_sources = set(sources_config.get("sources", {}).keys())
        indexed_sources = set(metadata.get("sources", {}).keys())

        # Sources in metadata but not in config
        unconfigured = indexed_sources - config_sources

        # Sources in config but not indexed
        unindexed = config_sources - indexed_sources

        packs = self.get_all_packs()

        return {
            "total_sources": len(config_sources | indexed_sources),
            "configured": len(config_sources),
            "indexed": len(indexed_sources),
            "unconfigured_sources": list(unconfigured),
            "unindexed_sources": list(unindexed),
            "tier_breakdown": {
                "official": len([p for p in packs if p.tier == PackTier.OFFICIAL]),
                "community": len([p for p in packs if p.tier == PackTier.COMMUNITY]),
                "personal": len([p for p in packs if p.tier == PackTier.PERSONAL]),
            },
            "packs": [p.to_dict() for p in packs]
        }
