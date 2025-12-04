"""
Backup Tools Module

Tools for backing up websites, newsletters, and other content sources.
"""

from .html import HTMLBackupScraper
from .substack import SubstackBackup

__all__ = ['HTMLBackupScraper', 'SubstackBackup']
