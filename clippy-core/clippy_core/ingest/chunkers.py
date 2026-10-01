"""
Chunkers: split a Document into citable Chunks.

Built in
--------
PageChunker          Generic. Packs paragraphs into ~target_chars chunks and
                     tracks which pages each chunk spans. Works on any PDF.
RegexSectionChunker  Structure-aware. Starts a new section wherever a line
                     matches ``pattern`` (e.g. "Rule 611 ..."), so every chunk
                     belongs to exactly one citable section. Long sections are
                     split into parts that keep the same section_id.

Writing your own
----------------
Subclass Chunker and implement ``chunk(doc) -> list[Chunk]``. Use
``self.make_chunk(...)`` so ids, urls and the metadata conventions (see
schemas.py) stay consistent. Point the manifest at it:

    chunker:
      class: my_project.chunking:MyChunker
      params: {some_option: 3}
"""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from ..schemas import Chunk
from .documents import Document

Line = Tuple[int, str]   # (page number, line text)


# Lines that start a new paragraph even without a blank line before them:
# "1. ", "2) ", "(a) ", "a) ", "- ", "• "
DEFAULT_PARAGRAPH_START = r"^\s*(\(?\d{1,3}[.)]|\(?[a-z][.)]|\([ivx]+\)|[-•▪])\s+"


class Chunker:
    """
    Base class.

    drop_lines       regexes for lines to discard (running headers, footers, page numbers)
    paragraph_start  regex for lines that begin a new paragraph (numbered items, bullets)
    """

    name = "base"

    def __init__(self, drop_lines: Optional[Sequence[str]] = None,
                 paragraph_start: Optional[str] = DEFAULT_PARAGRAPH_START):
        self.drop_patterns = [re.compile(p) for p in (drop_lines or [])]
        self.paragraph_start = re.compile(paragraph_start) if paragraph_start else None

    def chunk(self, doc: Document) -> List[Chunk]:  # pragma: no cover - interface
        raise NotImplementedError

    # ---- helpers for subclasses -----------------------------------------

    def lines(self, doc: Document) -> List[Line]:
        """All non-dropped lines of the document, tagged with their page number."""
        out: List[Line] = []
        for page in doc.pages:
            for raw in page.text.splitlines():
                line = raw.rstrip()
                if any(p.search(line) for p in self.drop_patterns):
                    continue
                out.append((page.num, line))
        return out

    def make_chunk(self, doc: Document, index: int, text: str, page_start: int, page_end: int,
                   extra: Optional[Dict[str, Any]] = None) -> Chunk:
        meta = doc.base_metadata()
        meta.update({"page_start": page_start, "page_end": page_end, "chunker": self.name})
        if extra:
            meta.update(extra)
        chunk_id = f"{doc.source_id}/{doc.doc_id}@{doc.edition or 'na'}#{index:05d}"
        return Chunk(id=chunk_id, content=text.strip(), source_id=doc.source_id,
                     title=doc.title, url=doc.page_url(page_start), metadata=meta)

    def paragraphs(self, lines: Iterable[Line]) -> List[Tuple[int, int, str]]:
        """Group lines into paragraphs on blank lines and paragraph starts. Returns (page_start, page_end, text)."""
        paras, buf, start, end = [], [], None, None
        for page, line in lines:
            if buf and self.paragraph_start and self.paragraph_start.match(line):
                paras.append((start, end, " ".join(buf)))
                buf, start, end = [], None, None
            if line.strip():
                if start is None:
                    start = page
                end = page
                buf.append(line.strip())
            elif buf:
                paras.append((start, end, " ".join(buf)))
                buf, start, end = [], None, None
        if buf:
            paras.append((start, end, " ".join(buf)))
        return paras

    @staticmethod
    def pack(paras: List[Tuple[int, int, str]], target: int, overlap: int = 0
             ) -> List[Tuple[int, int, str]]:
        """Pack paragraphs into pieces of about ``target`` chars. Oversized paragraphs are split."""
        pieces: List[Tuple[int, int, str]] = []
        expanded: List[Tuple[int, int, str]] = []
        for ps, pe, text in paras:
            if len(text) <= target:
                expanded.append((ps, pe, text))
                continue
            sentences = re.split(r"(?<=[.;:!?])\s+", text)
            cur = ""
            for s in sentences:
                while len(s) > target:           # no sentence breaks at all: hard split
                    if cur:
                        expanded.append((ps, pe, cur))
                        cur = ""
                    expanded.append((ps, pe, s[:target]))
                    s = s[target:]
                if cur and len(cur) + len(s) + 1 > target:
                    expanded.append((ps, pe, cur))
                    cur = s
                else:
                    cur = f"{cur} {s}".strip()
            if cur:
                expanded.append((ps, pe, cur))

        cur_text, cur_start, cur_end = "", None, None
        for ps, pe, text in expanded:
            if cur_text and len(cur_text) + len(text) + 2 > target:
                pieces.append((cur_start, cur_end, cur_text))
                tail = cur_text[-overlap:].split(" ", 1)[-1] if overlap else ""
                cur_text, cur_start = (tail, cur_end) if tail else ("", None)
            if cur_start is None:
                cur_start = ps
            cur_end = pe
            cur_text = f"{cur_text}\n\n{text}".strip()
        if cur_text:
            pieces.append((cur_start, cur_end, cur_text))
        return pieces


class PageChunker(Chunker):
    """Generic chunker for any document. ~target_chars per chunk, page-tracked, with overlap."""

    name = "page"

    def __init__(self, target_chars: int = 1500, overlap_chars: int = 200,
                 separate_pages: bool = False,
                 drop_lines: Optional[Sequence[str]] = None,
                 paragraph_start: Optional[str] = DEFAULT_PARAGRAPH_START):
        super().__init__(drop_lines, paragraph_start)
        self.target_chars = target_chars
        self.overlap_chars = overlap_chars
        self.separate_pages = separate_pages

    def chunk(self, doc: Document) -> List[Chunk]:
        if self.separate_pages:
            pieces = [piece for page in doc.pages for piece in self.pack(
                self.paragraphs(self.lines(replace(doc, pages=[page]))),
                self.target_chars, self.overlap_chars)]
        else:
            pieces = self.pack(self.paragraphs(self.lines(doc)), self.target_chars, self.overlap_chars)
        return [self.make_chunk(doc, i, text, ps, pe) for i, (ps, pe, text) in enumerate(pieces)]


class RegexSectionChunker(Chunker):
    """
    One section per heading match. ``pattern`` is matched against each line.

    Named groups:
        id     -> section_id     (required; falls back to group 1, then the whole match)
        title  -> section_title  (optional)

    Example for numbered rules like "Rule 611 Short Program - Singles":
        pattern: '^(?P<id>Rule\\s+\\d+[A-Z]?)\\b[\\s.:–-]*(?P<title>.*)$'

    Text before the first heading becomes a "front matter" section
    (set ``keep_front_matter=False`` to drop it).
    """

    name = "regex_section"

    def __init__(self, pattern: str, max_chars: int = 3000, flags: str = "",
                 keep_front_matter: bool = True, front_matter_id: str = "Front matter",
                 drop_lines: Optional[Sequence[str]] = None,
                 paragraph_start: Optional[str] = DEFAULT_PARAGRAPH_START):
        super().__init__(drop_lines, paragraph_start)
        re_flags = 0
        for f in flags.upper():
            re_flags |= {"I": re.IGNORECASE, "M": re.MULTILINE, "S": re.DOTALL}.get(f, 0)
        self.pattern = re.compile(pattern, re_flags)
        self.max_chars = max_chars
        self.keep_front_matter = keep_front_matter
        self.front_matter_id = front_matter_id

    def _heading(self, line: str) -> Optional[Tuple[str, str]]:
        m = self.pattern.search(line.strip())
        if not m:
            return None
        groups = m.groupdict()
        sid = groups.get("id") or (m.group(1) if m.groups() else m.group(0))
        title = (groups.get("title") or "").strip()
        return " ".join(sid.split()), title

    def sections(self, doc: Document) -> List[Dict[str, Any]]:
        """Split into sections: [{'id', 'title', 'lines': [(page, line), ...]}]."""
        sections: List[Dict[str, Any]] = []
        current = {"id": self.front_matter_id, "title": "", "lines": [], "front": True}
        for page, line in self.lines(doc):
            head = self._heading(line) if line.strip() else None
            if head:
                if current["lines"] and (not current.get("front") or self.keep_front_matter):
                    sections.append(current)
                current = {"id": head[0], "title": head[1], "lines": [(page, line)]}
            else:
                current["lines"].append((page, line))
        if current["lines"] and (not current.get("front") or self.keep_front_matter):
            sections.append(current)
        return [s for s in sections if any(l.strip() for _, l in s["lines"])]

    def chunk(self, doc: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        for section in self.sections(doc):
            pieces = self.pack(self.paragraphs(section["lines"]), self.max_chars, overlap=0)
            for part, (ps, pe, text) in enumerate(pieces, 1):
                extra = {"section_id": section["id"], "section_title": section["title"]}
                if len(pieces) > 1:
                    extra.update({"part": part, "parts": len(pieces)})
                chunks.append(self.make_chunk(doc, len(chunks), text, ps, pe, extra))
        return chunks


def load_chunker(spec: Union[None, str, Dict[str, Any], Chunker], base: Optional[Path] = None) -> Chunker:
    """
    Build a chunker from a manifest spec:
        None                                   -> PageChunker()
        "page" | "regex_section"               -> built-in chunkers
        "package.module:ClassName"             -> importable module
        "path/to/file.py:ClassName"            -> file, relative to ``base`` (the manifest folder)
        {"class": <any of the above>, "params": {...}}  -> ClassName(**params)
    """
    if spec is None:
        return PageChunker()
    if isinstance(spec, Chunker):
        return spec
    params: Dict[str, Any] = {}
    if isinstance(spec, dict):
        params = dict(spec.get("params") or {})
        spec = spec.get("class") or "page"
    short = {"page": PageChunker, "regex_section": RegexSectionChunker}
    if spec in short:
        return short[spec](**params)
    if ":" not in spec:
        raise ValueError(f"Chunker spec must be 'module:Class' or 'file.py:Class', got {spec!r}")
    target, class_name = spec.rsplit(":", 1)
    if target.endswith(".py"):
        path = Path(target)
        if not path.is_absolute() and base is not None:
            path = Path(base) / path
        if not path.exists():
            raise FileNotFoundError(f"Chunker file not found: {path}")
        import importlib.util
        module_spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[path.stem] = module
        module_spec.loader.exec_module(module)
    else:
        try:
            module = importlib.import_module(target)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Could not import chunker module {target!r}. Put it next to the manifest, "
                f"install it, or reference the file directly as 'my_chunker.py:{class_name}'.") from e
    return getattr(module, class_name)(**params)
