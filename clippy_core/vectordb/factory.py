"""
Factory utilities for vector stores.

Helper functions for creating and configuring vector stores.
"""

import os
from pathlib import Path


def get_chroma_path(dimension: int = 768, backup_path: str = None, language: str = "en") -> str:
    """
    Get the ChromaDB path for a specific dimension and language.

    Args:
        dimension: Embedding dimension (384, 768, 1024, 1536)
        backup_path: Base backup path (auto-detected if None)
        language: Language code (e.g., "en", "es")

    Returns:
        Path to the dimension/language-specific ChromaDB directory
    """
    if backup_path is None:
        backup_path = os.getenv("BACKUP_PATH", "data")

    if language and language != "en":
        return os.path.join(backup_path, f"chroma_db_{dimension}_{language}")

    return os.path.join(backup_path, f"chroma_db_{dimension}")


def get_dimension_for_model(model_name: str) -> int:
    """
    Get embedding dimension for a model name.

    Args:
        model_name: Model identifier

    Returns:
        Embedding dimension
    """
    model_dimensions = {
        # OpenAI
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        # Local models
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "intfloat/e5-large-v2": 1024,
        "intfloat-e5-large-v2": 1024,
    }

    return model_dimensions.get(model_name, 768)
