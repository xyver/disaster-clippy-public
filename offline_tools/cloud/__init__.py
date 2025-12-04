"""
Cloud Storage Module

Handles backup file storage on Cloudflare R2 (S3-compatible).
"""

from .r2 import R2Storage, get_r2_storage

__all__ = ['R2Storage', 'get_r2_storage']
