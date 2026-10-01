import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample"

# Let this standalone suite run from either clippy-core/ or its parent repo.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_sample_maker():
    spec = importlib.util.spec_from_file_location("make_sample_pdfs", SAMPLE / "make_sample_pdfs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def sample_pdfs(tmp_path_factory):
    maker = _load_sample_maker()
    out = tmp_path_factory.mktemp("pdfs")
    a, b = out / "rules-2024-25.pdf", out / "rules-2025-26.pdf"
    maker.make_pdf(a, "2024-25", maker.RULES_2024)
    maker.make_pdf(b, "2025-26", maker.RULES_2025)
    return a, b


PATTERN = r"^(?P<id>Rule\s+\d+[A-Z]?)\b[\s.:–-]*(?P<title>.*)$"
DROP = [r"^Sample Sport Technical Rules \d{4}-\d{2}$", r"^Page \d+$"]


@pytest.fixture
def manifest(sample_pdfs, tmp_path):
    a, b = sample_pdfs
    return {
        "index": str(tmp_path / "idx.sqlite"),
        "embedding": {"provider": "hash"},
        "chunker": {"class": "regex_section", "params": {"pattern": PATTERN, "drop_lines": DROP}},
        "defaults": {"source_id": "sample"},
        "documents": [
            {"path": str(a), "doc_id": "rules", "title": "Sample Rules", "edition": "2024-25",
             "url": "https://example.org/rules-2024-25.pdf"},
            {"path": str(b), "doc_id": "rules", "title": "Sample Rules", "edition": "2025-26"},
        ],
        "_base": tmp_path,
        "_path": None,
    }


@pytest.fixture
def built_index(manifest):
    from clippy_core.ingest.pipeline import build_index
    build_index(manifest, rebuild=True)
    return manifest["index"]
