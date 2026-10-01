"""
SQLiteHybridStore - a one-file index with keyword + semantic search.

- Keyword search: SQLite FTS5 with BM25 ranking (porter stemming), weighted
  so matches in ``section_id`` > ``title`` > ``content``. Exact terms like
  "Rittberger", "Level 4" or a rule number are found reliably.
- Semantic search: cosine similarity over embeddings stored in the same file,
  computed with numpy.
- Hybrid (default): both lists fused with Reciprocal Rank Fusion (RRF).

The whole index is one ``.sqlite`` file: easy to rebuild per edition, copy,
version, or ship inside an app. Needs no server.

Scale: vectors are held in memory for semantic search. Comfortable up to
roughly 100-200k chunks. Beyond that, swap in a dedicated vector backend
(pgvector adapter included; sqlite-vec, Chroma or LanceDB are easy additions)
behind the same ``search()`` signature.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..embeddings import Embedder, embedder_from_name
from ..schemas import Chunk, SearchResult, SourceInfo
from .base import SyncVectorStoreBase

SCHEMA_VERSION = "1"
RRF_K = 60
_TOKEN = re.compile(r"[\w][\w'.-]*[\w]|[\w]", re.UNICODE)


class IndexMismatchError(RuntimeError):
    pass


class SQLiteHybridStore(SyncVectorStoreBase):
    def __init__(self, path: str | Path, embedder: Optional[Embedder] = None,
                 api_key: Optional[str] = None, create: bool = True,
                 keyword_only: bool = False):
        self.path = Path(path)
        if not create and not self.path.exists():
            raise FileNotFoundError(f"Index not found: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

        stored = self.get_meta("embedder")
        if keyword_only and embedder is not None:
            raise ValueError("keyword_only cannot be combined with an embedder")
        if embedder is None:
            if not stored:
                raise IndexMismatchError(
                    f"{self.path} has no embedder recorded. Build it first, or pass an embedder.")
            if not keyword_only:
                embedder = embedder_from_name(stored, api_key=api_key)
        elif stored and stored != embedder.name:
            raise IndexMismatchError(
                f"Index {self.path} was built with {stored!r}, not {embedder.name!r}. "
                "Rebuild the index or query it with the same embedder.")
        self.embedder = embedder
        self.keyword_only = keyword_only
        if not stored:
            self.set_meta("embedder", embedder.name)
            self.set_meta("dimension", str(embedder.dimension))
            self.set_meta("schema_version", SCHEMA_VERSION)
            self.set_meta("created_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._matrix: Optional[np.ndarray] = None
        self._rowids: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ schema

    def _create_schema(self) -> None:
        c = self.conn
        c.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS chunks (
                rowid INTEGER PRIMARY KEY,
                id TEXT UNIQUE NOT NULL,
                source_id TEXT NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                url TEXT,
                metadata TEXT,
                embedding BLOB
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                title, section, content, tokenize='porter unicode61'
            );
        """)
        c.commit()

    def get_meta(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
        self.conn.commit()

    # ------------------------------------------------------------------ writes

    def add_chunks(self, chunks: Sequence[Chunk], batch_size: int = 64,
                   progress=None) -> int:
        """Insert or replace chunks (matched by id). Returns number written."""
        if self.keyword_only:
            raise IndexMismatchError("A keyword-only store cannot embed new chunks")
        written = 0
        for i in range(0, len(chunks), batch_size):
            batch = list(chunks[i:i + batch_size])
            vectors = self.embedder.embed([self._embed_text(ch) for ch in batch])
            with self.conn:
                for ch, vec in zip(batch, vectors):
                    self._delete_ids([ch.id])
                    cur = self.conn.execute(
                        "INSERT INTO chunks(id, source_id, title, content, url, metadata, embedding) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (ch.id, ch.source_id, ch.title, ch.content, ch.url,
                         json.dumps(ch.metadata, ensure_ascii=False),
                         np.asarray(vec, dtype=np.float32).tobytes()))
                    self.conn.execute(
                        "INSERT INTO chunks_fts(rowid, title, section, content) VALUES (?, ?, ?, ?)",
                        (cur.lastrowid, ch.title or "",
                         " ".join(str(ch.metadata.get(k, "")) for k in ("section_id", "section_title")),
                         ch.content))
            written += len(batch)
            if progress:
                progress(written, len(chunks))
        self._matrix = None
        return written

    @staticmethod
    def _embed_text(ch: Chunk) -> str:
        head = " ".join(str(ch.metadata.get(k, "")) for k in ("section_id", "section_title"))
        return f"{ch.title}\n{head}\n{ch.content}".strip()

    def _delete_ids(self, ids: Iterable[str]) -> None:
        for chunk_id in ids:
            row = self.conn.execute("SELECT rowid FROM chunks WHERE id=?", (chunk_id,)).fetchone()
            if row:
                self.conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (row["rowid"],))
                self.conn.execute("DELETE FROM chunks WHERE rowid=?", (row["rowid"],))

    def delete_source(self, source_id: str) -> int:
        rows = self.conn.execute("SELECT id FROM chunks WHERE source_id=?", (source_id,)).fetchall()
        with self.conn:
            self._delete_ids(r["id"] for r in rows)
        self._matrix = None
        return len(rows)

    def delete_doc(self, doc_id: str) -> int:
        rows = self.conn.execute(
            "SELECT id FROM chunks WHERE json_extract(metadata, '$.doc_id') = ?", (doc_id,)).fetchall()
        with self.conn:
            self._delete_ids(r["id"] for r in rows)
        self._matrix = None
        return len(rows)

    # ------------------------------------------------------------------ filters

    @staticmethod
    def _where(sources: Optional[List[str]], filters: Optional[Dict[str, Any]],
               alias: str = "c") -> Tuple[str, list]:
        clauses, params = [], []
        if sources:
            clauses.append(f"{alias}.source_id IN ({','.join('?' * len(sources))})")
            params += list(sources)
        for key, value in (filters or {}).items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid filter key: {key!r}")
            values = value if isinstance(value, (list, tuple, set)) else [value]
            clauses.append(
                f"CAST(json_extract({alias}.metadata, '$.{key}') AS TEXT) IN ({','.join('?' * len(values))})")
            params += [str(v) for v in values]
        return (" AND ".join(clauses) or "1=1"), params

    # ------------------------------------------------------------------ search

    def search(self, query: str, n_results: int = 10, sources: Optional[List[str]] = None,
               filters: Optional[Dict[str, Any]] = None, mode: str = "hybrid") -> List[SearchResult]:
        mode = (mode or "hybrid").lower()
        if mode == "keyword":
            return self.search_keyword(query, n_results, sources, filters)
        if mode == "semantic":
            return self.search_semantic(query, n_results, sources, filters)

        pool = max(n_results * 5, 50)
        kw = self._keyword_ranked(query, pool, sources, filters)
        sem = self._semantic_ranked(query, pool, sources, filters)
        fused: Dict[int, float] = {}
        for ranked in (kw, sem):
            for rank, (rowid, _score) in enumerate(ranked):
                fused[rowid] = fused.get(rowid, 0.0) + 1.0 / (RRF_K + rank + 1)
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:n_results]
        return self._results(top)

    def search_keyword(self, query: str, n_results: int = 10, sources: Optional[List[str]] = None,
                       filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        return self._results(self._keyword_ranked(query, n_results, sources, filters))

    def search_semantic(self, query: str, n_results: int = 10, sources: Optional[List[str]] = None,
                        filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        return self._results(self._semantic_ranked(query, n_results, sources, filters))

    @staticmethod
    def fts_query(query: str) -> str:
        """Turn free text into a safe FTS5 query: every term quoted, OR-joined."""
        terms = [t.replace('"', "") for t in _TOKEN.findall(query)]
        return " OR ".join(f'"{t}"' for t in terms if t)

    def _keyword_ranked(self, query, limit, sources, filters) -> List[Tuple[int, float]]:
        match = self.fts_query(query)
        if not match:
            return []
        where, params = self._where(sources, filters)
        # bm25() is lower-is-better; columns: title, section, content
        rows = self.conn.execute(
            f"SELECT c.rowid AS rowid, bm25(chunks_fts, 2.0, 4.0, 1.0) AS rank "
            f"FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
            f"WHERE chunks_fts MATCH ? AND {where} ORDER BY rank LIMIT ?",
            [match, *params, limit]).fetchall()
        return [(r["rowid"], -float(r["rank"])) for r in rows]

    def _load_matrix(self) -> None:
        rows = self.conn.execute("SELECT rowid, embedding FROM chunks ORDER BY rowid").fetchall()
        dim = self.embedder.dimension
        self._rowids = np.array([r["rowid"] for r in rows], dtype=np.int64)
        self._matrix = (np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                        if rows else np.zeros((0, dim), dtype=np.float32))

    def _semantic_ranked(self, query, limit, sources, filters) -> List[Tuple[int, float]]:
        if self.keyword_only:
            raise IndexMismatchError("Semantic search needs the embedder used to build this index")
        if self._matrix is None:
            self._load_matrix()
        if not len(self._rowids):
            return []
        scores = self._matrix @ self.embedder.embed_one(query)
        if sources or filters:
            where, params = self._where(sources, filters)
            allowed = {r["rowid"] for r in self.conn.execute(
                f"SELECT c.rowid FROM chunks c WHERE {where}", params)}
            mask = np.fromiter((rid in allowed for rid in self._rowids), dtype=bool,
                               count=len(self._rowids))
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:limit]
        return [(int(self._rowids[i]), float(scores[i])) for i in order if np.isfinite(scores[i])]

    def _results(self, ranked: List[Tuple[int, float]]) -> List[SearchResult]:
        if not ranked:
            return []
        ids = [rid for rid, _ in ranked]
        rows = {r["rowid"]: r for r in self.conn.execute(
            f"SELECT rowid, id, source_id, title, content, url, metadata FROM chunks "
            f"WHERE rowid IN ({','.join('?' * len(ids))})", ids)}
        out = []
        for rid, score in ranked:
            r = rows.get(rid)
            if r is None:
                continue
            meta = json.loads(r["metadata"] or "{}")
            out.append(SearchResult(
                id=r["id"], content=r["content"], source_id=r["source_id"], title=r["title"] or "",
                url=r["url"] or "", score=score, doc_type=meta.get("doc_type", "article"), metadata=meta))
        return out

    # ------------------------------------------------------------------ info

    def get_sources(self) -> List[SourceInfo]:
        rows = self.conn.execute(
            "SELECT source_id, COUNT(*) AS n FROM chunks GROUP BY source_id ORDER BY source_id").fetchall()
        return [SourceInfo(id=r["source_id"], name=r["source_id"], count=r["n"]) for r in rows]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        r = self.conn.execute("SELECT * FROM chunks WHERE id=?", (doc_id,)).fetchone()
        if not r:
            return None
        return {"id": r["id"], "source_id": r["source_id"], "title": r["title"], "content": r["content"],
                "url": r["url"], "metadata": json.loads(r["metadata"] or "{}")}

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def info(self) -> Dict[str, Any]:
        meta = {r["key"]: r["value"] for r in self.conn.execute("SELECT key, value FROM meta")}
        docs = self.conn.execute(
            "SELECT json_extract(metadata,'$.doc_id') AS doc, json_extract(metadata,'$.edition') AS ed, "
            "COUNT(*) AS n FROM chunks GROUP BY doc, ed ORDER BY doc").fetchall()
        return {"path": str(self.path), "chunks": self.count(), **meta,
                "documents": [{"doc_id": d["doc"], "edition": d["ed"], "chunks": d["n"]} for d in docs]}

    def close(self) -> None:
        self.conn.close()
