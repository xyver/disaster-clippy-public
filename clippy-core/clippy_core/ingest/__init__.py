"""
Ingestion: turn documents into cited chunks and write them to an index.

    Document (pages of text + metadata)
        -> Chunker.chunk(doc) -> [Chunk, ...]
        -> SQLiteHybridStore.add_chunks(...)

The Chunker is the main customisation point for a new domain. See
chunkers.py for the built-in ones and HANDOFF.md for how to write your own.
"""

from .documents import Document, Page
from .pdf import extract_pages
from .chunkers import Chunker, PageChunker, RegexSectionChunker, load_chunker
from .pipeline import build_index, load_manifest

__all__ = ["Document", "Page", "extract_pages", "Chunker", "PageChunker",
           "RegexSectionChunker", "load_chunker", "build_index", "load_manifest"]
