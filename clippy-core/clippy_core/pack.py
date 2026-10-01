"""Install and open a finished Clippy index pack without producer tooling."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


PACK_FORMAT = "clippy-sqlite-pack"
PACK_FORMAT_VERSION = 1
_PACK_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
_PDF_MEMBER = re.compile(r"^documents/[0-9a-f]{64}\.pdf$")
_CHUNK = 1024 * 1024


def _check_manifest(manifest: Dict[str, Any]) -> None:
    if manifest.get("format") != PACK_FORMAT or manifest.get("format_version") != PACK_FORMAT_VERSION:
        raise ValueError("Unsupported Clippy pack format")
    pack_id = manifest.get("pack_id")
    if not isinstance(pack_id, str) or not _PACK_ID.fullmatch(pack_id):
        raise ValueError("Invalid pack_id")
    if manifest.get("index_file") != "index.sqlite":
        raise ValueError("Pack must contain index.sqlite")
    files = manifest.get("files")
    if not isinstance(files, dict) or "index.sqlite" not in files:
        raise ValueError("Pack file checksums are missing")
    for name, info in files.items():
        if name != "index.sqlite" and not _PDF_MEMBER.fullmatch(name):
            raise ValueError(f"Unexpected pack member: {name}")
        if not isinstance(info, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(info.get("sha256", ""))):
            raise ValueError(f"Invalid checksum for {name}")
        if not isinstance(info.get("size_bytes"), int) or info["size_bytes"] < 0:
            raise ValueError(f"Invalid size for {name}")
    if not isinstance(manifest.get("source_ids"), list) or not manifest["source_ids"]:
        raise ValueError("Pack has no source IDs")
    if not isinstance(manifest.get("passage_count"), int) or manifest["passage_count"] <= 0:
        raise ValueError("Pack has no passages")


def _check_database(path: Path, expected_count: int) -> None:
    with closing(sqlite3.connect(str(path))) as conn:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Pack SQLite index failed integrity_check")
        count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if count != expected_count:
            raise ValueError(f"Pack passage count mismatch: {count} != {expected_count}")


def _copy_verified(source, destination: Path, expected: Dict[str, Any]) -> None:
    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as out:
        while chunk := source.read(_CHUNK):
            out.write(chunk)
            digest.update(chunk)
            size += len(chunk)
            if size > expected["size_bytes"]:
                raise ValueError(f"Pack member exceeds declared size: {destination.name}")
    if size != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"Pack member failed checksum: {destination.name}")


def inspect_pack(archive: str | Path) -> Dict[str, Any]:
    """Validate archive structure and checksums; return its public manifest."""
    with zipfile.ZipFile(archive) as pack:
        names = pack.namelist()
        if len(names) != len(set(names)) or "pack.json" not in names:
            raise ValueError("Pack members are missing or duplicated")
        if pack.getinfo("pack.json").file_size > _CHUNK:
            raise ValueError("Pack manifest is too large")
        manifest = json.loads(pack.read("pack.json"))
        _check_manifest(manifest)
        if set(names) != {"pack.json", *manifest["files"]}:
            raise ValueError("Pack members do not match manifest")
        for name, expected in manifest["files"].items():
            digest = hashlib.sha256()
            size = 0
            with pack.open(name) as member:
                while chunk := member.read(_CHUNK):
                    digest.update(chunk)
                    size += len(chunk)
                    if size > expected["size_bytes"]:
                        raise ValueError(f"Pack member exceeds declared size: {name}")
            if size != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
                raise ValueError(f"Pack member failed checksum: {name}")
        return manifest


@dataclass(frozen=True)
class InstalledPack:
    path: Path
    manifest: Dict[str, Any]

    def open_store(self, *, keyword_only: bool = True, api_key: str | None = None):
        """Open this prepared index; keyword mode needs no embedding model."""
        from .vectordb import SQLiteHybridStore

        return SQLiteHybridStore(self.path / "index.sqlite", create=False,
                                 keyword_only=keyword_only, api_key=api_key)


def open_pack(directory: str | Path) -> InstalledPack:
    """Validate an installed pack before opening it for chat or search."""
    path = Path(directory).resolve()
    manifest = json.loads((path / "pack.json").read_text(encoding="utf-8"))
    _check_manifest(manifest)
    for name, expected in manifest["files"].items():
        file_path = path / name
        with file_path.open("rb") as source:
            digest = hashlib.sha256()
            size = 0
            while chunk := source.read(_CHUNK):
                digest.update(chunk)
                size += len(chunk)
        if size != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
            raise ValueError(f"Installed pack member failed checksum: {name}")
    _check_database(path / "index.sqlite", manifest["passage_count"])
    return InstalledPack(path, manifest)


def install_pack(archive: str | Path, library: str | Path) -> InstalledPack:
    """Verify and install a pack under ``library/<pack_id>``; refuse overwrite."""
    manifest = inspect_pack(archive)
    root = Path(library).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / manifest["pack_id"]
    if destination.exists():
        raise FileExistsError(f"Pack already installed: {destination}")
    with tempfile.TemporaryDirectory(prefix=".clippy-install-", dir=root) as temporary:
        staged = Path(temporary)
        with zipfile.ZipFile(archive) as pack:
            (staged / "pack.json").write_bytes(pack.read("pack.json"))
            for name, expected in manifest["files"].items():
                with pack.open(name) as source:
                    _copy_verified(source, staged / name, expected)
        _check_database(staged / "index.sqlite", manifest["passage_count"])
        staged.rename(destination)
    return InstalledPack(destination, manifest)
