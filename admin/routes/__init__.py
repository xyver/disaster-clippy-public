"""
Admin Routes Module

FastAPI routes for the admin panel.
"""

from .sources import router as sourcepacks_router
from .source_tools import router as source_tools_router
from .packs import router as packs_router
from .jobs import router as jobs_router
from .job_builder import router as job_builder_router
from .search_test import router as search_test_router
from .localizations import router as localizations_router

__all__ = [
    'sourcepacks_router',
    'source_tools_router',
    'packs_router',
    'jobs_router',
    'job_builder_router',
    'search_test_router',
    'localizations_router'
]
