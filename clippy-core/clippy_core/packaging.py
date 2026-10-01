"""Producer-side export of a built SQLite index as a transferable Clippy pack."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .ingest.pipeline import _document
from .pack import PACK_FORMAT, PACK_FORMAT_VERSION, _check_database, _check_manifest, inspect_pack


def _file_info(path: Path) -> Dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def export_pack(manifest: Dict[str, Any], output: str | Path, *, include_pdfs: bool = False) -> Path:
    """Release a consistent index snapshot after checking every declared document."""
    pack_info = manifest.get("pack") or {}
    required = ("id", "name", "short_name", "description", "tags")
    if any(not pack_info.get(key) for key in required):
        raise ValueError(f"Manifest pack section requires: {', '.join(required)}")
    if not isinstance(pack_info["tags"], list) or not all(isinstance(t, str) for t in pack_info["tags"]):
        raise ValueError("Pack tags must be a list of strings")
    if len(str(pack_info["short_name"])) > 40:
        raise ValueError("Pack short_name must be 40 characters or less")

    index_path = Path(manifest["index"]).resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Build the index before exporting: {index_path}")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Pack archive already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".clippy-export-", dir=output.parent) as temporary:
        staged = Path(temporary)
        snapshot = staged / "index.sqlite"
        with closing(sqlite3.connect(str(index_path))) as source, closing(sqlite3.connect(str(snapshot))) as target:
            source.backup(target)
        with closing(sqlite3.connect(str(snapshot))) as conn:
            rows = conn.execute(
                "SELECT source_id, json_extract(metadata,'$.doc_id'), "
                "COALESCE(json_extract(metadata,'$.edition'), ''), COUNT(*) "
                "FROM chunks GROUP BY 1, 2, 3"
            ).fetchall()
            actual = {(source_id, doc_id, edition): count
                      for source_id, doc_id, edition, count in rows}
            meta = dict(conn.execute("SELECT key, value FROM meta"))
        if not actual:
            raise ValueError("Index has no passages")
        _check_database(snapshot, sum(actual.values()))

        base = Path(manifest.get("_base") or Path.cwd())
        defaults = manifest.get("defaults") or {}
        documents = []
        expected = set()
        source_pdfs = {}
        for entry in manifest["documents"]:
            doc = _document(entry, defaults, base)
            key = (doc.source_id, doc.doc_id, doc.edition)
            if key in expected:
                raise ValueError(f"Duplicate source/document/edition in manifest: {key}")
            expected.add(key)
            if actual.get(key, 0) <= 0:
                raise ValueError(f"Declared document is absent from index: {key}")
            record = {"source_id": doc.source_id, "doc_id": doc.doc_id,
                      "edition": doc.edition, "title": doc.title, "url": doc.url,
                      "passage_count": actual[key]}
            if include_pdfs:
                if not doc.path.is_file():
                    raise FileNotFoundError(f"PDF unavailable for inclusion: {doc.path}")
                info = _file_info(doc.path)
                member = f"documents/{info['sha256']}.pdf"
                record["pdf_member"] = member
                source_pdfs[member] = (doc.path, info)
            documents.append(record)
        if set(actual) != expected:
            raise ValueError(f"Index contains undeclared documents: {sorted(set(actual) - expected)}")

        files = {"index.sqlite": _file_info(snapshot)}
        files.update({member: info for member, (_, info) in source_pdfs.items()})
        packed = {
            "format": PACK_FORMAT,
            "format_version": PACK_FORMAT_VERSION,
            "pack_id": str(pack_info["id"]),
            "name": str(pack_info["name"]),
            "short_name": str(pack_info["short_name"]),
            "description": str(pack_info["description"]),
            "tags": pack_info["tags"],
            "license": str(pack_info.get("license") or "unspecified"),
            "origin": str(pack_info.get("origin") or "project"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "index_file": "index.sqlite",
            "index_schema_version": meta.get("schema_version", ""),
            "embedder": meta.get("embedder", ""),
            "embedding_dimension": int(meta.get("dimension") or 0),
            "source_ids": sorted({key[0] for key in actual}),
            "passage_count": sum(actual.values()),
            "documents": documents,
            "files": files,
        }
        _check_manifest(packed)
        archive = staged / "pack.clippypack"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as pack:
            pack.writestr("pack.json", json.dumps(packed, indent=2, ensure_ascii=False) + "\n")
            pack.write(snapshot, "index.sqlite")
            for member, (path, _) in sorted(source_pdfs.items()):
                pack.write(path, member)
        inspect_pack(archive)
        archive.rename(output)
    return output
