# clippy-core

Cited search and chat over your own documents. It turns PDFs into a one-file index, searches it with keyword and semantic ranking together, and returns answers that cite the document, edition, section and page, with a link to the exact page.

It's extracted from [Disaster Clippy](https://github.com/xyver/disaster-clippy-public) as a small, portable core. The first tested external use is a U.S. figure skating rules search.

## Use a prepared index

A consuming app can receive a prepared `.sqlite` index and use only the search/chat runtime. Keyword search needs no local embedding model; written answers use a server-side OpenAI or Anthropic API key.

```bash
pip install -e ".[openai]"  # or [anthropic]; no PDF or local-model dependencies
```

```python
from clippy_core import ChatService, ClippyConfig
from clippy_core.vectordb import SQLiteHybridStore

config = ClippyConfig(index_path="rules.sqlite", search_mode="keyword", llm_provider="openai")
store = SQLiteHybridStore(config.index_path, create=False, keyword_only=True)
answer = ChatService(store, config=config).chat_sync("What counts as a fall?")
print(answer.text, answer.search_results)
```

If your application already searches its own database (for example Supabase
pgvector), pass its prepared passages to `answer_sync` instead. Keep private
user or property details in `host_context`; they personalize the answer but
are not cited as source evidence.

```python
from clippy_core import ChatService
from clippy_core.schemas import SearchResult

passages = [SearchResult.from_pgvector(row) for row in search_rows]
answer = ChatService(config=config).answer_sync(
    "What does this rule require?", passages, max_evidence=5,
    host_context="Discipline: singles; season: 2026-27",
)
print(answer.text, answer.search_results)  # render these exact references
```

The host must authenticate the user and filter `search_rows` before passing
them to core. Citation validation and carrying prior-turn passages into the
latest reference list are still planned; see
[`docs/portable-core-strategy.md`](../docs/portable-core-strategy.md).

Each project can point `ClippyConfig.index_path` at its own prepared index or
pass its own database results to `answer_sync`. `sources=` and `filters=` narrow
search within an index; the host decides which indexes and source IDs a user
may access. Pass `sources=` with host-retrieved passages too: the core rejects
passages outside that explicit selection, and an empty selection searches
nothing. Keep personal documents separate from official source packs.

The host can also inject any model adapter that implements `LLMBackend`:
`provider`, `generate_async(messages, system_prompt)`, and
`generate_stream_async(messages, system_prompt)`. This works for a host's local
model server or a different cloud provider without adding that provider to
the core package. The exported runtime bundles no local inference engine.

Keep the model key in the host app's backend. The host app should authenticate callers, limit usage, and enforce a spending budget before calling chat. `clippy_core.server` is a local demonstration API with no such controls. Hybrid/semantic search needs the same query embedder that built the index; a prepared index built with local sentence-transformers embeddings still needs that model for semantic queries. Keyword mode opens that index without it.

To copy just this consumer runtime into another project, run `python export_runtime.py --output /path/to/clippy-runtime`. The export contains the read-and-chat modules and their package metadata; it omits ingestion, source tools, evaluation, the CLI, and the unauthenticated demo server. The target path must not already exist.

## Quick start

```bash
pip install -e ".[pdf,server]"            # from this clippy-core directory; add ,anthropic or ,openai for LLM answers

python examples/sample/make_sample_pdfs.py                     # two fictional rulebooks
clippy build   -m examples/sample/sources.yaml --rebuild       # PDFs -> build/sample-rules.sqlite
clippy search  "short program length" -m examples/sample/sources.yaml -f edition=2025-26
clippy ask     "What is the costume deduction?" -m examples/sample/sources.yaml
clippy eval    examples/sample/golden.yaml -m examples/sample/sources.yaml --compare
clippy serve   -m examples/sample/sources.yaml                 # http://127.0.0.1:8000
```

It works with no API keys: keyword search runs at full strength, and `ask` returns the cited passages instead of a written answer. Add keys for more:

| Env var | Effect |
|---|---|
| `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` | `ask` writes an answer that quotes and cites the rules |
| `OPENAI_API_KEY` at build time | real semantic embeddings when the manifest selects `openai` or `auto` (or install sentence-transformers for local embeddings) |

The sample manifest explicitly selects `hash` embeddings so it runs offline and deterministically. The [2026-27 U.S. rulebook example](examples/skating/README.md) uses local sentence-transformers embeddings; install `.[pdf,local]` for it. Hash embeddings provide word overlap, not strong semantic matching.

Set `CLIPPY_MODEL_CACHE` to keep downloaded local embedding models in a chosen directory. Keep the model available at query time; the index records the model name and refuses queries made with a different embedder.

## Commands

| Command | What it does |
|---|---|
| `clippy build -m sources.yaml [--rebuild]` | Extract, chunk, embed, and write the index |
| `clippy build docs/*.pdf --index x.sqlite` | Quick build with the generic chunker, no manifest |
| `clippy preview -m sources.yaml [--doc ID] [--full]` | Show how documents will chunk, without embedding. Use it constantly while writing a chunker. |
| `clippy info -m sources.yaml` | What's in the index: documents, editions, embedder |
| `clippy search "…" [-f key=value] [--mode keyword\|semantic\|hybrid]` | Ranked passages with citations |
| `clippy ask "…" [-f key=value]` | Cited answer |
| `clippy eval golden.yaml [--compare]` | Retrieval scores (hit@1, hit@k, MRR) and the misses |
| `clippy serve` | Minimal web page and JSON API |

## Python

```python
from clippy_core import ChatService, ClippyConfig
from clippy_core.vectordb import SQLiteHybridStore

config = ClippyConfig(index_path="build/sample-rules.sqlite", prompt_path="examples/sample/prompt.md")
chat = ChatService(SQLiteHybridStore(config.index_path), config=config)

resp = chat.chat_sync("How long can the short program be?", filters={"edition": "2025-26"})
print(resp.text)
for r in resp.search_results:
    print(r.citation(), r.url)
```

## Layout

```
clippy_core/
  config.py          ClippyConfig: providers, index path, search and context limits
  schemas.py         Chunk, SearchResult (with .citation()), metadata conventions
  embeddings.py      openai | local (sentence-transformers) | hash (no deps)
  llm.py             Anthropic / OpenAI, sync + async + streaming
  context.py         How passages are numbered and shown to the LLM
  chat.py            ChatService: search -> cited context -> answer
  evaluation.py      Golden-set retrieval scoring
  cli.py, server.py  `clippy` command and minimal FastAPI app
  prompts/default.md Default system prompt
  ingest/            PDF extraction, chunkers, manifest-driven build pipeline
  vectordb/          SQLiteHybridStore (default), PgVectorStore (optional)
examples/
  sample/            Fictional rulebooks, manifest, prompt, golden set (used by tests)
  skating/           Tested rulebook manifest, custom chunker, prompt, golden set
tests/               pytest suite (runs offline in about a second)
```

**Building on this? Read [HANDOFF.md](HANDOFF.md).**

## License

MIT. Documents you index keep their own licenses.
