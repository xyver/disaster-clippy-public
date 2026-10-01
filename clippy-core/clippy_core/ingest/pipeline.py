"""
Build an index from a manifest (sources.yaml) or a list of PDF paths.

Manifest format (paths are relative to the manifest file):

    index: build/rules.sqlite           # where to write the index
    embedding:                          # optional; see embeddings.py
      provider: auto                    # auto | openai | local | hash
      model: ""
    chunker:                            # default chunker for all documents
      class: regex_section              # or "page", or "my_module:MyChunker"
      params: {pattern: '^(?P<id>Rule\\s+\\d+)\\b\\s*(?P<title>.*)$'}
    defaults:                           # merged into every document entry
      source_id: skating-rules
    documents:
      - path: pdfs/tech-rules-2025.pdf
        doc_id: tech-rules              # stable id, same across editions
        title: Technical Rules
        edition: "2025-26"
        url: https://example.org/tech-rules-2025.pdf   # public link for citations
        metadata: {discipline: singles}                 # anything extra, filterable
        pages: "5-480"                  # optional: only index these pages (skip cover, TOC, index)
        chunker: {...}                  # optional per-document override
    config:                             # optional ClippyConfig overrides for ask/serve
      prompt_path: prompt.md

The directory containing the manifest is added to sys.path, so a custom
chunker module can sit right next to it.
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import yaml

from ..embeddings import Embedder, get_embedder
from ..vectordb.sqlite_hybrid import SQLiteHybridStore
from .chunkers import Chunker, load_chunker
from .documents import Document
from .pdf import extract_pages, looks_scanned


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "doc"


def load_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))
    data["_base"] = base
    data["_path"] = path
    if data.get("index"):
        data["index"] = str((base / data["index"]).resolve())
    cfg = data.get("config") or {}
    if cfg.get("prompt_path"):
        cfg["prompt_path"] = str((base / cfg["prompt_path"]).resolve())
    data["config"] = cfg
    if not data.get("documents"):
        raise ValueError(f"{path}: manifest has no 'documents' list")
    return data


def manifest_from_files(files: Sequence[Union[str, Path]], source_id: str = "docs",
                        index: str = "index.sqlite", chunker: Any = None) -> Dict[str, Any]:
    """A minimal in-memory manifest for `clippy build some/*.pdf`."""
    docs = []
    for f in files:
        f = Path(f).resolve()
        docs.append({"path": str(f), "doc_id": _slug(f.stem), "title": f.stem.replace("_", " ")})
    return {"index": str(Path(index).resolve()), "chunker": chunker, "defaults": {"source_id": source_id},
            "documents": docs, "config": {}, "_base": Path.cwd(), "_path": None}


@dataclass
class BuildReport:
    index: str
    embedder: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def total_chunks(self) -> int:
        return sum(d["chunks"] for d in self.documents)

    def summary(self) -> str:
        lines = [f"Index: {self.index}", f"Embedder: {self.embedder}"]
        for d in self.documents:
            lines.append(f"  {d['doc_id']} {d.get('edition') or ''}: {d['pages']} pages -> {d['chunks']} chunks")
        lines.append(f"Total: {self.total_chunks} chunks in {self.seconds:.1f}s")
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def parse_page_ranges(spec: Any, total: int) -> set:
    """ "5-480", "1,3,10-20", 12, or None (all pages) -> set of 1-based page numbers."""
    if spec in (None, "", "all"):
        return set(range(1, total + 1))
    pages: set = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a), (int(b) if b.strip() else total) + 1))
        elif part:
            pages.add(int(part))
    return {p for p in pages if 1 <= p <= total}


def _document(entry: Dict[str, Any], defaults: Dict[str, Any], base: Path) -> Document:
    e = {**defaults, **entry}
    metadata = dict(defaults.get("metadata") or {})
    metadata.update(entry.get("metadata") or {})
    path = Path(e["path"])
    if not path.is_absolute():
        path = (base / path).resolve()
    return Document(
        doc_id=str(e.get("doc_id") or _slug(path.stem)),
        path=path,
        title=str(e.get("title") or path.stem),
        source_id=str(e.get("source_id") or "docs"),
        edition=str(e.get("edition") or ""),
        url=str(e.get("url") or ""),
        metadata=metadata,
    )


def build_index(manifest: Dict[str, Any], rebuild: bool = False, embedder: Optional[Embedder] = None,
                progress: Optional[Callable[[str], None]] = None) -> BuildReport:
    """Extract, chunk, embed and store every document in the manifest."""
    say = progress or (lambda msg: None)
    started = time.time()
    index_path = Path(manifest.get("index") or "index.sqlite")
    if rebuild and index_path.exists():
        index_path.unlink()

    if embedder is None:
        emb_cfg = manifest.get("embedding") or {}
        embedder = get_embedder(emb_cfg.get("provider", "auto"), emb_cfg.get("model", ""))
    store = SQLiteHybridStore(index_path, embedder=embedder)
    report = BuildReport(index=str(index_path), embedder=embedder.name)
    if embedder.name.startswith("hash"):
        report.warnings.append("Using 'hash' embeddings (no OpenAI key or sentence-transformers found). "
                               "Keyword search is full strength; semantic search is word-overlap only.")

    base = Path(manifest.get("_base") or Path.cwd())
    default_chunker = load_chunker(manifest.get("chunker"), base)
    defaults = manifest.get("defaults") or {}

    for entry in manifest["documents"]:
        doc = _document(entry, defaults, base)
        if not doc.path.exists():
            report.warnings.append(f"{doc.doc_id}: file not found: {doc.path}")
            continue
        chunker: Chunker = load_chunker(entry["chunker"], base) if entry.get("chunker") else default_chunker
        say(f"Reading {doc.path.name}")
        doc.pages = extract_pages(doc.path)
        keep = parse_page_ranges(entry.get("pages"), len(doc.pages))
        doc.pages = [p for p in doc.pages if p.num in keep]
        if looks_scanned(doc.pages):
            report.warnings.append(f"{doc.doc_id}: very little text; the PDF may be scanned and need OCR")
        chunks = chunker.chunk(doc)
        say(f"  {len(doc.pages)} pages -> {len(chunks)} chunks ({chunker.name}); embedding…")
        # Replace any previous chunks of this document+edition, keep other editions
        removed = [c for c in store.conn.execute(
            "SELECT id FROM chunks WHERE source_id=? AND json_extract(metadata,'$.doc_id')=? "
            "AND COALESCE(json_extract(metadata,'$.edition'),'')=?",
            (doc.source_id, doc.doc_id, doc.edition))]
        if removed:
            with store.conn:
                store._delete_ids(r["id"] for r in removed)
        store.add_chunks(chunks)
        report.documents.append({"doc_id": doc.doc_id, "edition": doc.edition,
                                 "pages": len(doc.pages), "chunks": len(chunks)})

    store.set_meta("built_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
    if manifest.get("_path"):
        store.set_meta("manifest", str(manifest["_path"]))
    store.close()
    report.seconds = time.time() - started
    return report
