# Handoff: building on clippy-core

This is the baseline for a cited rules search. It's meant to be taken over by people who know the rules better than the people who built it. Everything generic works already. The parts that need domain knowledge are small, isolated, and listed below.

## 1. What you're getting

For the current producer-to-consumer handoff, see [Portable pack workflow](../docs/portable-pack-workflow.md). Build and evaluate in this checkout, export a `.clippypack`, and give the other app the archive plus the consumer runtime. Additional skating PDFs belong as entries in one manifest; the pack exporter checks that each was indexed.

- **One-file index.** PDFs become a single `.sqlite` file with keyword (BM25) and semantic search, fused. No server or cloud database needed. Rebuild it each season, version it, or ship it.
- **Citations everywhere.** Every chunk knows its document, edition, section (rule number) and pages. Answers cite `[n]`, and links open the PDF at the right page.
- **Works without keys.** With no LLM, you get the matching passages. With an Anthropic or OpenAI key, you get written answers that quote the rule text.
- **An evaluation harness.** A golden question set scores retrieval, so every change is measured.
- **Tests.** 27 tests covering chunking, search, filters, build, chat, API, and CLI. `pytest` runs offline in about a second.

## 2. How it fits together

```
sources.yaml ──► extract_pages(pdf) ──► Chunker.chunk(doc) ──► SQLiteHybridStore.add_chunks()
 (documents,        page-by-page text      one chunk per rule,      FTS5 + embeddings,
  editions, urls,                           metadata attached        one .sqlite file
  chunker choice)                                                         │
                                                                          ▼
question ──► ChatService.search() ──► ContextFormatter ──► LLM (+ prompt.md) ──► answer with [n] citations
             hybrid, filterable        numbered passages      or extractive fallback
             (edition, discipline…)    with citation + link
```

## 3. Where you customise (and where you don't need to)

| You want to… | Change this | Code? |
|---|---|---|
| Split the rulebooks correctly | `chunker:` in `sources.yaml`: `pattern`, `drop_lines`, `max_chars`. Then `pages:` per document. | No |
| Split in ways a regex can't | `examples/skating/skating_chunker.py` (subclass `RegexSectionChunker`) | Yes, one file |
| Add a rulebook or a new season | Add an entry under `documents:` with the same `doc_id` and a new `edition` | No |
| Change how answers are written | `examples/skating/prompt.md` | No |
| Change what "good" means | `examples/skating/golden.yaml` | No |
| Add filters (discipline, level…) | Put the field in chunk `metadata` (manifest `metadata:` or your chunker), then `-f discipline=singles` | Maybe |
| Change passage layout for the LLM | Subclass `ContextFormatter` and pass `formatter=` to ChatService | Yes |
| Use another LLM (e.g. Ollama offline) | Subclass `LLMService` and pass `llm_service=` | Yes |
| Use another vector backend | Implement `search(query, n_results, sources, filters, mode)`. Nothing else in the pipeline depends on the backend. | Yes |

**You shouldn't need to touch:** the store, the pipeline, ChatService, or the CLI. If you find you do, that's a design gap worth raising.

## 4. Metadata conventions

Citations, filters and eval all rely on these keys (see `clippy_core/schemas.py`):

| Key | Meaning | Example |
|---|---|---|
| `doc_id` | Stable id of the rulebook, same across editions | `isu-technical-rules` |
| `doc_title` | Human title | `US Figure Skating Rulebook` (verify exact PDF title) |
| `edition` | Season, edition, or version | `2025-26` |
| `section_id` | The citable unit | `Rule <number>` (verify against the PDF) |
| `section_title` | Its heading | The section heading as printed in the PDF |
| `page_start`, `page_end` | 1-based PDF pages | `147`, `148` |
| `url` | Public PDF link. The page anchor is added automatically. | `https://…/rules.pdf` |

Any other key (discipline, level, federation…) passes through and is filterable.

## 5. The working loop for US figure skating rulebooks

1. **Add the PDF** to `examples/skating/pdfs/` or make an ignored local manifest pointing at its absolute path. The checked-in manifest records the inspected 2026-27 U.S. rulebook and its publisher URL; see [examples/skating/README.md](examples/skating/README.md).
2. **Preview, don't build:** `clippy preview -m examples/skating/sources.yaml --limit 40`. Check that:
   - each chunk is exactly one rule, with the right `section_id` and title
   - the table of contents isn't producing fake sections. If it is, set `pages: "N-"` to skip it.
   - running headers, footers, and page numbers are gone. If not, add them to `drop_lines`. A leftover header attaches to the previous rule and stretches its page range.
   - numbered paragraphs (`1.`, `a)`) are kept separate
3. **Build:** `clippy build -m examples/skating/sources.yaml --rebuild`
4. **Expand the eight verified golden questions** with real user questions and the rules that answer them. Aim for 20–50 across disciplines.
5. **Measure:** `clippy eval examples/skating/golden.yaml -m examples/skating/sources.yaml --compare`. See [examples/skating/README.md](examples/skating/README.md) for the tested 2026-27 U.S. rulebook layout and local-model workflow.
6. Fix the misses (chunking, synonyms, prompt), rebuild, and re-run eval. Repeat.

## 6. Ideas the domain team is best placed to do

In rough order of value:

1. **Paragraph-level citations.** Split each rule on its numbered paragraphs and set a verified paragraph-level `section_id`, so answers point at the exact paragraph.
2. **Amendments and communications.** Index them as their own documents (their own `doc_id`), and record which rules they modify. Then show "amended by…" next to a rule.
3. **Element vocabulary.** Skaters search `3Lz`, `triple Lutz`, and `Lutz` interchangeably. Add a synonyms field in the chunker, or expand queries before search, so keyword search matches all of them.
4. **Edition awareness.** Default filters to the current season, and add a "what changed between editions" view (same `doc_id` and `section_id`, different `edition`).
5. **Tables.** Some rules live in tables (levels, values). PDF text extraction flattens them. Check a few by hand. PyMuPDF has table extraction if needed.

## 7. Known limits and things not yet tested

- **Live LLM calls are untested.** The Anthropic and OpenAI code is adapted from Disaster Clippy and exercised in tests with a fake LLM, but it hasn't made a real API call in this build. Run one `clippy ask` with a key first. The default Anthropic model is set in `clippy_core/llm.py` (`DEFAULT_MODELS`); override with `LLM_MODEL`.
- **Embedding coverage.** The 2026-27 U.S. rulebook was indexed with the local `all-MiniLM-L6-v2` sentence-transformers model. OpenAI embeddings remain untested in this build.
- **`PgVectorStore`** is carried over from the February 2026 extraction, untested, and semantic-only.
- **`hash` embeddings** are a placeholder. They match word overlap, not meaning. Set `OPENAI_API_KEY` or install sentence-transformers for real use, and rebuild. The index records its embedder and refuses mismatched queries.
- **Scanned PDFs** aren't OCR'd. The build warns when a PDF has almost no text. Disaster Clippy's `offline_tools/ocr.py` can produce `.ocr.pdf` copies.
- **Complex layouts.** Two-column pages, headings split across lines, and tables may need chunker work. Check with `preview`.
- **Scale.** Semantic vectors are held in memory. That's fine up to roughly 100–200k chunks. A few rulebooks are a few thousand.
- **The server** has no auth or rate limiting. It's a local and dev starting point, not a public deployment.
- **Copyright.** Rulebooks belong to their publishers. Linking to the official PDF with page anchors and showing short quoted passages is the safe pattern. Check the federation's terms before hosting the PDFs yourself.

## 8. Relationship to Disaster Clippy

This is the `clippy_core/` extraction described in Disaster Clippy's `docs/clippy-core-extraction.md`. It's restored from commit `6e67f72` (February 2026) and extended with the SQLite hybrid store, embeddings, ingestion, citation formatting, CLI, eval, and tests. The fallbacks into `offline_tools` are gone, so it has no dependency on Disaster Clippy. Disaster Clippy itself still runs on its own code. It can adopt this core later if parity is proven.
