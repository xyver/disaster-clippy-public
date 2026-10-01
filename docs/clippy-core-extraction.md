# clippy-core extraction

`clippy-core/` is the portable Python distribution of Disaster Clippy's cited search and chat ideas. It is checked into this public repo as a self-contained project with its own `pyproject.toml`, `clippy_core` package, examples, tests, README, and handoff guide. A developer can copy that directory or install it with `pip install -e ./clippy-core`.

The first intended external use is a search and chat system for figure skating rulebooks. The package accepts PDFs, records document and edition metadata, builds a local SQLite index, searches it, and supplies cited passages to an optional LLM. It has no imports from Disaster Clippy's `app.py`, `admin/`, or `offline_tools/`.

## Current boundary

| Area | Responsibility |
| --- | --- |
| `clippy-core/clippy_core/` | Portable PDF ingestion, chunking, embeddings, hybrid retrieval, citations, chat, evaluation, CLI, and a small local server |
| `app.py`, `admin/`, `offline_tools/` | Existing Disaster Clippy hosted and local runtime, source packs, admin tools, and cloud integrations |
| `clippy-core/examples/` | Fictional sample and a figure skating manifest/chunker template |
| `clippy-core/tests/` | Offline package tests |

The hosted app still uses its existing search implementation. Adopting `clippy_core` inside the app requires parity checks for source filters, Pinecone/Chroma adapters, streaming, and source-pack behavior. The extracted package currently supplies SQLite hybrid search and an optional PgVector adapter; it does not contain a Pinecone adapter.

## Two meanings of “core”

The County Map private distribution docs (`docs/distribution model/disaster-clippy/core_strategy.md`) call the **core engine** the full public app/runtime installed by a small wrapper. That wrapper installs, updates, and launches the existing app, then offers packs and optional capabilities. This remains one maintained app codebase with multiple delivery paths.

`clippy-core` is a **developer library** extracted from that codebase's search and chat concepts. It can be installed into another project without the Disaster Clippy UI, admin, wrapper, or source-pack store. It is not the wrapper's engine artifact. The SQLite index built for a particular rulebook is a content artifact owned by the consuming project, not part of the library distribution.

This distinction also preserves the distribution model's optional layers: regular app users do not need PDF authoring dependencies or local models just to use installed source packs.

## What the package adds

- PDF text extraction with page tracking and a scanned-PDF warning.
- Generic page chunking and configurable rule/section chunking; projects can supply their own chunker.
- Source, document, edition, section, and page metadata on each chunk.
- A single SQLite file with FTS5 keyword ranking, stored embeddings, and reciprocal-rank hybrid search.
- Hash embeddings for an offline demo, or OpenAI and sentence-transformers embeddings for real semantic retrieval.
- Cited extractive answers without an LLM; optional OpenAI or Anthropic answer generation.
- CLI commands for preview, build, search, ask, info, eval, and a local API server.
- Golden-question retrieval evaluation and sample PDFs.
- A verified `.clippypack` export and install path for finished SQLite indexes; see [Portable pack workflow](portable-pack-workflow.md).

See [`clippy-core/README.md`](../clippy-core/README.md) for the install and quick start, and [`clippy-core/HANDOFF.md`](../clippy-core/HANDOFF.md) for metadata conventions and domain customization.

## Figure skating sequence

1. The 451-page 2026-27 U.S. Figure Skating Rulebook has been inspected. Its text layer is searchable, and the rules and diagram sections are mapped in `clippy-core/examples/skating/sources.yaml`.
2. The rulebook has been built as a 1,888-chunk SQLite index with local `all-MiniLM-L6-v2` embeddings. An ignored local manifest points at the supplied PDF; the PDF and index are not committed.
3. Eight verified rule questions are recorded in `clippy-core/examples/skating/golden.yaml`. Expand them with real user questions, especially where several disciplines use similar wording.
4. Add the separate competition and test requirement charts that the publisher maintains outside the rulebook PDF.
5. A separate, temporary hosted reference of the same 1,888 chunks now uses OpenAI embeddings in the official Pinecone index. It is listed in the public catalog as `usfs-rulebook-2026-27` and links to the publisher's PDF. It is not a downloadable pack and does not change the portable SQLite index.
6. Test a real LLM answer path and check citation grounding before relying on generated skating answers.

Scanned pages need OCR before this package can index them. The existing Disaster Clippy OCR tooling is a possible producer of searchable PDF copies, but the portable package does not import it.

## Validation and limits

The package's 37 offline tests pass in this repo. The fictional sample builds an 18-chunk SQLite index, and the 451-page U.S. rulebook builds into 1,888 chunks with local embeddings. On eight verified questions, hybrid retrieval finds the expected rule in its top five results for all eight; top-result accuracy is four of eight. This is an initial retrieval baseline, not a complete quality evaluation. Live LLM calls, OpenAI embeddings, and the PgVector adapter remain untested in this integration.

The package server is a local development surface with no authentication or rate limiting. It should be wrapped by a project-specific application before public deployment.

## Prepared-data runtime boundary

The next consumer target is narrower than the current developer package: the consuming app receives a prepared index, searches it, formats citations, and sends bounded context to a model API through its own backend. The app owns authentication, rate limits, quotas, provider keys, and spending controls. It does not need PDF extraction, source onboarding, or a local LLM. The `clippy_core` chat interface already accepts an injected LLM service for an app-owned API proxy.

The first supported path is keyword search against the prepared SQLite index with `SQLiteHybridStore(..., create=False, keyword_only=True)` and an OpenAI or Anthropic chat provider. It works with the locally embedded skating index without installing sentence-transformers at query time. Hybrid search still requires a query embedder compatible with the index's stored vectors. To run hybrid search without a local model, the producer must prepare vectors with an API embedding model, and the consuming backend must use that same model for queries.

`python clippy-core/export_runtime.py --output <new-folder>` now produces a consumer-only folder with search/chat and pack installation modules plus minimal package metadata. It excludes PDF ingestion, pack creation, build/eval commands, examples, the CLI, and the unauthenticated demo server. `clippy-core/` remains the producer checkout; its base install leaves PDF and local-model dependencies optional. A cost-controlled API adapter is still owned by the consuming app, where authentication, quota policy, and provider credentials belong. The existing source tools stay in the producer side of Disaster Clippy.

## Migration rule

Keep this as a parallel, reusable package while the Disaster Clippy app continues to use its existing paths. Move app call sites onto the package only after behavior and deployment parity are demonstrated. Preserve one package implementation and make app-specific adapters above it rather than copying its code into a second runtime.

## Citation continuity across chat turns

The hosted app retrieves up to 15 candidates and reduces them to five before building both the current model context and the visible reference cards. It also passes recent conversation history to the model. A model can therefore repeat a rule and link from an earlier answer even when that rule is absent from the current five cards. This happened in the skating demo: an answer mentioned Rules 8373 and 2712, while its current cards listed other rules.

For the portable core and its host apps, treat history as conversational context, not as current evidence. If an answer relies on a passage from a previous turn, retrieve that passage again by its stable source/document/section identity, include its text in the current bounded context, and display its link with the latest answer. Apply the currently selected collection filter before carrying it forward. If the passage cannot be recovered or is outside the selected collection, the answer should say it cannot verify that claim from the current sources. The current answer's citation links must resolve to passages shown in its own reference list. Preserve a bounded reference count by replacing a lower-value current hit when a prior passage is needed; do not silently add hidden evidence.

This is a requirement for future implementation and evaluation, not a guarantee of the current hosted app. Test it with follow-up questions that cite a rule found in an earlier answer, and with a new chat containing the same question but no history.

See [Sheltrium integration review](sheltrium-integration-review.md) for an external consumer that has the same history-versus-current-evidence issue.

The shared host/runtime boundary informed by Sheltrium, Disaster Clippy, and the planned GoFigure app is in [Portable clippy core](portable-core-strategy.md).
