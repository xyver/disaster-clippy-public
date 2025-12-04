"""
Source Packs System

Enables users to browse, download, and install curated content packs
from the parent server into their local Clippy instance.
"""

from .registry import SourcePackRegistry, SourcePack, PackTier
from .api import router as sourcepacks_router

__all__ = ['SourcePackRegistry', 'SourcePack', 'PackTier', 'sourcepacks_router']
