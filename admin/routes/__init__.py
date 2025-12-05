"""
Admin Routes Module

FastAPI routes for the admin panel.
"""

from .sources import router as sourcepacks_router
from .source_tools import router as source_tools_router
from .packs import router as packs_router
from .jobs import router as jobs_router

__all__ = [
    'sourcepacks_router',
    'source_tools_router',
    'packs_router',
    'jobs_router'
]
