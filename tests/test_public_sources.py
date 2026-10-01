"""Public catalog routes must not query Pinecone for source discovery."""

import importlib
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient


def test_public_source_routes_use_catalog(monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setenv("VECTOR_DB_MODE", "pinecone")
    monkeypatch.setattr(
        app_module,
        "get_public_catalog_sources",
        lambda: [{"source_id": "test-rules", "name": "Test Rules", "doc_count": 12}],
    )
    monkeypatch.setattr(
        app_module,
        "get_vector_store",
        lambda: (_ for _ in ()).throw(AssertionError("source discovery queried Pinecone")),
    )
    monkeypatch.setattr(
        app_module,
        "_source_cache",
        {"data": [], "ids": set(), "expires": None},
    )

    client = TestClient(app_module.app)
    sources = client.get("/sources")
    assert sources.status_code == 200
    assert sources.json()["sources"]["test-rules"]["count"] == 12
    assert sources.json()["sources"]["test-rules"]["has_1536"] is True

    simple_sources = client.get("/api/v1/sources")
    assert simple_sources.status_code == 200
    assert simple_sources.json()["sources"] == [
        {"id": "test-rules", "name": "Test Rules", "count": 12}
    ]
    welcome = client.get("/welcome")
    assert welcome.status_code == 200
    assert welcome.json()["stats"]["total_documents"] == 12
    assert "currently empty" not in welcome.json()["message"]


def test_public_sources_fail_closed_without_catalog(monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setenv("VECTOR_DB_MODE", "pinecone")
    monkeypatch.setattr(app_module, "get_public_catalog_sources", lambda: [])
    monkeypatch.setattr(
        app_module,
        "get_vector_store",
        lambda: (_ for _ in ()).throw(AssertionError("source discovery queried Pinecone")),
    )
    monkeypatch.setattr(
        app_module,
        "_source_cache",
        {"data": [], "ids": set(), "expires": None},
    )

    client = TestClient(app_module.app)
    assert client.get("/sources").json()["sources"] == {}
    assert client.get("/api/v1/sources").json()["sources"] == []


def test_hosted_mode_forces_pinecone_without_local_fallback(monkeypatch):
    from admin.local_config import get_local_config
    from offline_tools.vectordb import factory

    monkeypatch.setenv("VECTOR_DB_MODE", "pinecone")
    local_config = get_local_config()
    monkeypatch.setitem(local_config.config, "offline_mode", "offline_only")
    assert local_config.get_offline_mode() == "online_only"

    calls = []
    monkeypatch.setattr(factory, "get_vector_store", lambda **kwargs: calls.append(kwargs) or "cloud-store")
    assert factory.get_vector_store_for_search(fallback=True) == "cloud-store"
    assert calls == [{"mode": "pinecone"}]


def test_empty_stream_still_completes():
    app_module = importlib.import_module("app")
    client = TestClient(app_module.app)
    response = client.post("/api/v1/chat/stream", json={"message": ""})
    assert response.status_code == 200
    assert "data: [DONE]" in response.text


def test_presence_question_does_not_search():
    app_module = importlib.import_module("app")
    client = TestClient(app_module.app)
    response = client.post("/chat", json={"message": "do you talk still?"})
    assert response.status_code == 200
    assert response.json()["articles"] == []
    assert response.json()["response"].startswith("Yes, I'm here")

    stream = client.post("/api/v1/chat/stream", json={"message": "are you there?"})
    assert stream.status_code == 200
    assert "[ARTICLES][]" in stream.text
    assert "[DONE]" in stream.text


def test_parallel_page_loads_wait_for_catalog_refresh(monkeypatch):
    app_module = importlib.import_module("app")
    monkeypatch.setattr(app_module, "_public_catalog_cache", {"catalog": None, "expires": None})
    calls = []

    def refresh():
        calls.append(True)
        time.sleep(0.05)
        app_module._public_catalog_cache.update({
            "catalog": {"sources": [{"source_id": "test-rules"}]},
            "expires": datetime.now(timezone.utc) + timedelta(minutes=1),
        })

    monkeypatch.setattr(app_module, "_refresh_public_catalog_cache", refresh)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: app_module.get_public_catalog(), range(2)))
    assert len(calls) == 1
    assert all(result["sources"][0]["source_id"] == "test-rules" for result in results)


def test_skating_question_excludes_other_discipline_rules():
    app_module = importlib.import_module("app")
    def result(source, discipline):
        return {"metadata": {"source": source, "discipline": discipline}}

    articles = [
        result("usfs-rulebook-2026-27", "synchronized"),
        result("usfs-rulebook-2026-27", "singles"),
        result("usfs-rulebook-2026-27", ""),
        result("appropedia", ""),
    ]
    filtered = app_module.filter_skating_results_by_discipline(
        "senior synchronized skating short program time", articles
    )
    assert filtered == [articles[0], articles[2], articles[3]]
    assert app_module.filter_skating_results_by_discipline(
        "Compare singles and synchronized skating", articles
    ) == articles


def test_rule_context_uses_section_id_not_result_number():
    app_module = importlib.import_module("app")
    context = app_module.format_articles_for_context([{
        "metadata": {
            "title": "1400 | U.S. Figure Skating Rulebook",
            "section_id": "1400",
            "source": "usfs-rulebook-2026-27",
            "url": "https://example.test/rules.pdf#page=97",
            "categories": [],
        },
        "content": "A fall is defined in this rule.",
        "score": 0.8,
    }])
    assert "Reference 1:" in context
    assert "Rule/Section: 1400" in context
    assert "Article #1" not in context


def test_catalog_rebuild_preserves_hosted_reference_only():
    from offline_tools.cloud.catalog import _preserve_hosted_references

    entries = [{"source_id": "ready-gov"}]
    existing = {"sources": [
        {"source_id": "ready-gov", "reference_only": False},
        {"source_id": "usfs-rulebook-2026-27", "reference_only": True},
        {"source_id": "old-pack", "reference_only": False},
    ]}
    result = _preserve_hosted_references(entries, existing)
    assert [item["source_id"] for item in result] == ["ready-gov", "usfs-rulebook-2026-27"]
