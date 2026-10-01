"""PDF text extraction, page by page. Uses PyMuPDF if installed, else pypdf."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .documents import Page

SCANNED_CHARS_PER_PAGE = 40   # below this on average, the PDF is probably scanned


def extract_pages(path: str | Path) -> List[Page]:
    path = Path(path)
    try:
        import pymupdf  # PyMuPDF >= 1.24
    except ImportError:
        try:
            import fitz as pymupdf  # older PyMuPDF
        except ImportError:
            pymupdf = None

    if pymupdf is not None:
        with pymupdf.open(str(path)) as doc:
            return [Page(i, page.get_text("text") or "") for i, page in enumerate(doc, start=1)]

    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("Install a PDF library: pip install pymupdf  (or: pip install pypdf)") from e
    reader = PdfReader(str(path))
    return [Page(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]


def looks_scanned(pages: List[Page]) -> bool:
    """True if the PDF has little or no text layer and likely needs OCR."""
    if not pages:
        return True
    return sum(len(p.text.strip()) for p in pages) / len(pages) < SCANNED_CHARS_PER_PAGE
