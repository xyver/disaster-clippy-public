"""
Local User Admin Panel
Provides settings management for end users of their offline disaster preparedness system
"""

from .app import router
from .local_config import get_local_config, LocalConfig

__all__ = ['router', 'get_local_config', 'LocalConfig']
