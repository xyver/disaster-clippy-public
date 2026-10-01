from clippy_core.ingest.chunkers import PageChunker, RegexSectionChunker, load_chunker
from clippy_core.ingest.documents import Document, Page
from clippy_core.ingest.pdf import extract_pages, looks_scanned
from clippy_core.ingest.pipeline import parse_page_ranges

from conftest import DROP, PATTERN


def doc_from(pdf, edition="2025-26"):
    d = Document(doc_id="rules", path=pdf, title="Sample Rules", source_id="s", edition=edition,
                 url="https://example.org/r.pdf")
    d.pages = extract_pages(pdf)
    return d


def test_extract_pages_numbers_from_one(sample_pdfs):
    pages = extract_pages(sample_pdfs[1])
    assert [p.num for p in pages] == list(range(1, len(pages) + 1))
    assert "Rule 101" in pages[0].text
    assert not looks_scanned(pages)


def test_regex_chunker_one_chunk_per_rule(sample_pdfs):
    chunks = RegexSectionChunker(PATTERN, drop_lines=DROP).chunk(doc_from(sample_pdfs[1]))
    ids = [c.metadata["section_id"] for c in chunks]
    assert ids[0] == "Front matter"
    assert ids[1:] == ["Rule 101", "Rule 102", "Rule 201", "Rule 202", "Rule 301", "Rule 302",
                       "Rule 401", "Rule 501"]
    r202 = chunks[ids.index("Rule 202")]
    assert r202.metadata["section_title"] == "Short Program - Duration"
    assert "2 minutes 50 seconds" in r202.content
    assert r202.metadata["page_start"] == 2
    assert r202.url == "https://example.org/r.pdf#page=2"
    assert r202.metadata["edition"] == "2025-26"
    # running headers and page footers are dropped
    assert all("Page " not in c.content and "Technical Rules 2025-26" not in c.content for c in chunks)


def test_numbered_paragraphs_stay_separate(sample_pdfs):
    chunks = RegexSectionChunker(PATTERN, drop_lines=DROP).chunk(doc_from(sample_pdfs[1]))
    r201 = next(c for c in chunks if c.metadata.get("section_id") == "Rule 201")
    assert "\n\n2. A maximum of three jump elements" in r201.content


def test_long_rule_is_split_into_parts_with_same_id(sample_pdfs):
    chunks = RegexSectionChunker(PATTERN, drop_lines=DROP, max_chars=300).chunk(doc_from(sample_pdfs[1]))
    parts = [c for c in chunks if c.metadata.get("section_id") == "Rule 301"]
    assert len(parts) > 1
    assert [p.metadata["part"] for p in parts] == list(range(1, len(parts) + 1))
    assert all(len(p.content) <= 300 for p in parts)
    assert len({c.id for c in chunks}) == len(chunks)


def test_front_matter_can_be_dropped(sample_pdfs):
    chunks = RegexSectionChunker(PATTERN, keep_front_matter=False).chunk(doc_from(sample_pdfs[1]))
    assert chunks[0].metadata["section_id"] == "Rule 101"


def test_us_skating_heading_pattern_ignores_standalone_rule_references():
    from pathlib import Path
    from clippy_core.ingest.pipeline import load_manifest

    manifest_path = Path(__file__).resolve().parents[1] / "examples/skating/sources.yaml"
    manifest = load_manifest(manifest_path)
    chunker = load_chunker(manifest["chunker"], base=manifest_path.parent)
    doc = Document(doc_id="rulebook", path=manifest_path, title="Rules", source_id="rules")
    doc.pages = [Page(1, "2026-27 U.S. Figure Skating Rulebook\n"
                         "4535 for Rules on required officials.\n"
                         "9042\t Permissible Time Allowance - Short Programs\n"
                         "Senior short program: 2:50 maximum time")]
    chunks = chunker.chunk(doc)
    assert [c.metadata["section_id"] for c in chunks] == ["Front matter", "9042"]


def test_page_chunker_tracks_pages_and_size(sample_pdfs):
    chunks = PageChunker(target_chars=400, overlap_chars=50).chunk(doc_from(sample_pdfs[0]))
    assert len(chunks) > 3
    assert all(c.metadata["page_start"] <= c.metadata["page_end"] for c in chunks)
    assert max(c.metadata["page_end"] for c in chunks) == 3
    assert all(len(c.content) <= 400 + 60 for c in chunks)


def test_page_chunker_can_keep_diagram_pages_separate(sample_pdfs):
    chunks = PageChunker(target_chars=1500, separate_pages=True).chunk(doc_from(sample_pdfs[0]))
    assert chunks
    assert all(c.metadata["page_start"] == c.metadata["page_end"] for c in chunks)
    assert {c.metadata["page_start"] for c in chunks} == {1, 2, 3}


def test_oversized_paragraph_without_sentences_is_hard_split():
    d = Document(doc_id="x", path=None, title="X", source_id="s")
    d.pages = [Page(1, "word" * 1000)]
    chunks = PageChunker(target_chars=500, overlap_chars=0).chunk(d)
    assert len(chunks) >= 8 and all(len(c.content) <= 500 for c in chunks)


def test_load_chunker_specs(tmp_path):
    assert isinstance(load_chunker(None), PageChunker)
    assert isinstance(load_chunker("page"), PageChunker)
    c = load_chunker({"class": "regex_section", "params": {"pattern": PATTERN, "max_chars": 999}})
    assert isinstance(c, RegexSectionChunker) and c.max_chars == 999
    (tmp_path / "mychunk.py").write_text(
        "from clippy_core.ingest.chunkers import PageChunker\nclass Mine(PageChunker):\n    name='mine'\n")
    assert load_chunker("mychunk.py:Mine", base=tmp_path).name == "mine"


def test_parse_page_ranges():
    assert parse_page_ranges(None, 5) == {1, 2, 3, 4, 5}
    assert parse_page_ranges("2-", 5) == {2, 3, 4, 5}
    assert parse_page_ranges("1,3-4", 5) == {1, 3, 4}
    assert parse_page_ranges(9, 5) == set()
