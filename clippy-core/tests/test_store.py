import pytest

from clippy_core.embeddings import HashEmbedder
from clippy_core.schemas import Chunk
from clippy_core.vectordb import IndexMismatchError, SQLiteHybridStore


def make_store(tmp_path):
    store = SQLiteHybridStore(tmp_path / "s.sqlite", embedder=HashEmbedder(256))
    store.add_chunks([
        Chunk("a", "The Rittberger is an edge jump taken off from a back outside edge.", "s",
              "Book", "", {"section_id": "Rule 1", "edition": "2024-25", "doc_id": "book"}),
        Chunk("b", "A fall results in a deduction of one point.", "s", "Book", "",
              {"section_id": "Rule 2", "edition": "2025-26", "doc_id": "book"}),
        Chunk("c", "Costumes must be modest. Props are not permitted.", "other", "Guide", "",
              {"section_id": "Rule 3", "edition": "2025-26", "doc_id": "guide"}),
    ])
    return store


def test_keyword_finds_exact_term(tmp_path):
    store = make_store(tmp_path)
    assert store.search_keyword("rittberger")[0].id == "a"
    assert store.search("How much is deducted for a fall?")[0].id == "b"


def test_section_id_is_searchable(tmp_path):
    store = make_store(tmp_path)
    assert store.search_keyword("Rule 3")[0].id == "c"


def test_filters_and_sources(tmp_path):
    store = make_store(tmp_path)
    ids = {r.id for r in store.search("rule", filters={"edition": "2025-26"})}
    assert ids == {"b", "c"}
    ids = {r.id for r in store.search("rule", sources=["other"], mode="semantic")}
    assert ids == {"c"}
    ids = {r.id for r in store.search("rule", filters={"edition": ["2024-25", "2025-26"]}, mode="keyword")}
    assert ids == {"a", "b", "c"}


def test_fts_query_is_safe():
    q = SQLiteHybridStore.fts_query('what\'s "Rule 611" (a) AND OR NEAR* -x')
    assert q.count('"') % 2 == 0 and "*" not in q


def test_weird_queries_do_not_crash(tmp_path):
    store = make_store(tmp_path)
    for q in ['"', "(", "AND", "***", "", "   "]:
        store.search(q)


def test_replace_and_delete(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks([Chunk("a", "Replaced text about loops.", "s", "Book", "", {"doc_id": "book"})])
    assert store.count() == 3
    assert store.search_keyword("loops")[0].id == "a"
    assert store.delete_doc("book") == 2
    assert store.count() == 1


def test_reopen_uses_stored_embedder_and_rejects_mismatch(tmp_path):
    make_store(tmp_path).close()
    reopened = SQLiteHybridStore(tmp_path / "s.sqlite")
    assert reopened.embedder.name == "hash-256"
    assert reopened.search("fall")
    reopened.close()
    with pytest.raises(IndexMismatchError):
        SQLiteHybridStore(tmp_path / "s.sqlite", embedder=HashEmbedder(512))


def test_prepared_index_opens_for_keyword_search_without_loading_embedder(tmp_path, monkeypatch):
    make_store(tmp_path).close()
    import clippy_core.vectordb.sqlite_hybrid as sqlite_module

    monkeypatch.setattr(sqlite_module, "embedder_from_name", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("keyword search loaded an embedding model")))
    store = SQLiteHybridStore(tmp_path / "s.sqlite", create=False, keyword_only=True)
    assert store.search("fall", mode="keyword")[0].id == "b"
    with pytest.raises(IndexMismatchError, match="Semantic search needs"):
        store.search("fall", mode="semantic")
    store.close()


def test_citation_format(tmp_path):
    r = make_store(tmp_path).search_keyword("fall")[0]
    r.metadata.update({"doc_title": "Tech Rules", "page_start": 4, "page_end": 5})
    assert r.citation() == "Tech Rules 2025-26 · Rule 2 · pp. 4-5"
