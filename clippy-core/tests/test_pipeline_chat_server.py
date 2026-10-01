import asyncio

import pytest

from clippy_core import ChatService, ClippyConfig
from clippy_core.evaluation import evaluate
from clippy_core.ingest.pipeline import build_index
from clippy_core.llm import LLMService
from clippy_core.schemas import ResponseMethod
from clippy_core.vectordb import SQLiteHybridStore

GOLDEN = {"k": 5, "questions": [
    {"q": "How many jump elements are allowed in the short program?", "expect": {"section_id": "Rule 201"}},
    {"q": "maximum short program length", "filters": {"edition": "2025-26"},
     "expect": {"section_id": "Rule 202", "edition": "2025-26"}},
    {"q": "costume violation deduction", "expect_text": "deduction of 2.0"},
    {"q": "what counts as a fall", "expect": {"section_id": "Rule 501"}},
]}


def test_build_reports_and_rebuild_is_idempotent(manifest):
    r1 = build_index(manifest, rebuild=True)
    assert [d["chunks"] for d in r1.documents] == [9, 9]
    r2 = build_index(manifest)              # re-run without rebuild: replaces, doesn't duplicate
    store = SQLiteHybridStore(manifest["index"])
    assert store.count() == r1.total_chunks == r2.total_chunks
    assert {d["edition"] for d in store.info()["documents"]} == {"2024-25", "2025-26"}


def test_same_document_id_in_different_sources_does_not_replace_chunks(manifest):
    original = manifest["documents"][1]
    manifest["defaults"]["metadata"] = {"publisher": "Example"}
    manifest["documents"].append({**original, "source_id": "second-source"})
    build_index(manifest, rebuild=True)
    store = SQLiteHybridStore(manifest["index"])
    assert store.count() == 27
    counts = dict(store.conn.execute("SELECT source_id, COUNT(*) FROM chunks GROUP BY source_id"))
    assert counts == {"sample": 18, "second-source": 9}
    assert all(r.metadata["publisher"] == "Example" for r in store.search("short program"))


def test_missing_file_is_a_warning(manifest):
    manifest["documents"].append({"path": "nope.pdf", "doc_id": "nope"})
    report = build_index(manifest, rebuild=True)
    assert any("not found" in w for w in report.warnings)


def test_eval_on_sample(built_index):
    report = evaluate(SQLiteHybridStore(built_index), GOLDEN, mode="hybrid")
    assert report.hit_at_k == 1.0
    assert 0 < report.mrr <= 1.0
    assert "hit@1" in report.summary()


def test_chat_without_llm_is_extractive(built_index):
    config = ClippyConfig(index_path=built_index, llm_provider="none")
    chat = ChatService(SQLiteHybridStore(built_index), config=config)
    resp = chat.chat_sync("short program duration", filters={"edition": "2025-26"})
    assert resp.method == ResponseMethod.SIMPLE
    assert "[1]" in resp.text and "Rule 202" in resp.text
    assert all(r.metadata["edition"] == "2025-26" for r in resp.search_results)


class FakeLLM(LLMService):
    def __init__(self, config):
        super().__init__(config)
        self.provider = "fake"
        self.seen = None

    async def generate_async(self, messages, system_prompt=None):
        self.seen = (messages[-1].content, system_prompt)
        return "The short program may last 2 minutes 50 seconds [1]."


def test_chat_sends_numbered_cited_passages_and_prompt(built_index, tmp_path):
    prompt = tmp_path / "p.md"
    prompt.write_text("You are a test prompt.")
    config = ClippyConfig(index_path=built_index, prompt_path=str(prompt))
    llm = FakeLLM(config)
    chat = ChatService(SQLiteHybridStore(built_index), config=config, llm_service=llm)
    resp = chat.chat_sync("short program duration", filters={"edition": "2025-26"})
    user_msg, system = llm.seen
    assert system == "You are a test prompt."
    assert user_msg.count("[1] Sample Rules 2025-26 · Rule 202") == 1
    assert '"""' in user_msg
    assert resp.method == ResponseMethod.CLOUD_LLM
    assert resp.to_dict()["citations"][0]["citation"].startswith("Sample Rules 2025-26")


def test_server_endpoints(built_index):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from clippy_core.server import create_app

    config = ClippyConfig(index_path=built_index, llm_provider="none")
    client = TestClient(create_app(config, SQLiteHybridStore(built_index)))
    assert "Rules search" in client.get("/").text
    r = client.get("/api/search", params={"q": "fall", "filter": "edition:2024-25"}).json()
    assert r["results"] and all(x["metadata"]["edition"] == "2024-25" for x in r["results"])
    assert r["results"][0]["url"].startswith("https://example.org/rules-2024-25.pdf#page=")
    r = client.post("/api/ask", json={"question": "props", "filters": {"edition": "2025-26"}}).json()
    assert r["method"] == "simple" and r["results"]
    local = [x for x in r["results"] if x["url"].startswith("/files/")]
    assert local, "2025-26 has no public url, so links should point at /files/"
    assert client.get(local[0]["url"].split("#")[0]).headers["content-type"] == "application/pdf"
    assert client.get("/files/unknown").status_code == 404


def test_cli_smoke(built_index, capsys):
    from clippy_core.cli import main
    main(["search", "Rittberger OR fall", "--index", built_index, "-k", "2"])
    main(["ask", "costume deduction", "--index", built_index, "--provider", "none", "-f", "edition=2025-26"])
    out = capsys.readouterr().out
    assert "Rule 501" in out and "deduction of 2.0" in out
