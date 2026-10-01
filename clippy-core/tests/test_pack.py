import zipfile

import pytest

from clippy_core.pack import inspect_pack, install_pack, open_pack
from clippy_core.packaging import export_pack


def _pack_manifest(manifest):
    manifest["pack"] = {
        "id": "sample-rules", "name": "Sample Sport Rules", "short_name": "Sample Rules",
        "description": "Two fictional rulebook editions", "tags": ["rules", "sample"],
        "license": "CC0-1.0", "origin": "project",
    }
    return manifest


def test_export_install_and_search_two_documents(manifest, built_index, tmp_path):
    archive = export_pack(_pack_manifest(manifest), tmp_path / "sample.clippypack")
    with zipfile.ZipFile(archive) as packed:
        assert set(packed.namelist()) == {"pack.json", "index.sqlite"}
    description = inspect_pack(archive)
    assert description["pack_id"] == "sample-rules"
    assert len(description["documents"]) == 2
    assert {entry["edition"] for entry in description["documents"]} == {"2024-25", "2025-26"}
    installed = install_pack(archive, tmp_path / "library")
    assert open_pack(installed.path).manifest == description
    store = installed.open_store()
    try:
        matches = store.search("short program", n_results=5, mode="keyword")
        assert matches
        assert matches[0].metadata["edition"] in {"2024-25", "2025-26"}
    finally:
        store.close()
    assert open_pack(installed.path).manifest == description
    with pytest.raises(FileExistsError):
        install_pack(archive, tmp_path / "library")


def test_optional_pdfs_and_release_gate(manifest, built_index, tmp_path):
    _pack_manifest(manifest)
    archive = export_pack(manifest, tmp_path / "with-pdfs.clippypack", include_pdfs=True)
    description = inspect_pack(archive)
    assert len([name for name in description["files"] if name.startswith("documents/")]) == 2
    assert all("pdf_member" in entry for entry in description["documents"])

    incomplete = {**manifest, "documents": manifest["documents"][:1]}
    with pytest.raises(ValueError, match="undeclared documents"):
        export_pack(incomplete, tmp_path / "incomplete.clippypack")

    missing = {**manifest, "documents": [*manifest["documents"],
                {**manifest["documents"][0], "doc_id": "missing"}]}
    with pytest.raises(ValueError, match="absent from index"):
        export_pack(missing, tmp_path / "missing.clippypack")


def test_rejects_tampered_index(manifest, built_index, tmp_path):
    archive = export_pack(_pack_manifest(manifest), tmp_path / "original.clippypack")
    tampered = tmp_path / "tampered.clippypack"
    with zipfile.ZipFile(archive) as original, zipfile.ZipFile(tampered, "w") as changed:
        changed.writestr("pack.json", original.read("pack.json"))
        changed.writestr("index.sqlite", b"bad index")
    with pytest.raises(ValueError, match="checksum"):
        inspect_pack(tampered)
    with pytest.raises(ValueError, match="checksum"):
        install_pack(tampered, tmp_path / "library")


def test_runtime_export_contains_consumer_only_pack_support(tmp_path):
    from export_runtime import export_runtime

    output = export_runtime(tmp_path / "runtime")
    assert (output / "clippy_core" / "pack.py").exists()
    assert not (output / "clippy_core" / "packaging.py").exists()
    assert not (output / "clippy_core" / "ingest").exists()
