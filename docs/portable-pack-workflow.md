# Portable Clippy packs

Disaster Clippy can prepare source documents and hand another project a finished search index. The receiving project needs the small `clippy-core` runtime and a `.clippypack` archive. It does not need the PDF extraction scripts, chunkers, embedding model used at build time, or Disaster Clippy's admin tools.

## Boundaries

| Layer | Owner | Contents |
|---|---|---|
| Producer | Disaster Clippy or another pack builder | PDFs, manifest, PDF extraction, chunker, embedding process, retrieval evaluation |
| Finished pack | Producer hands to consumer | `pack.json`, `index.sqlite`, optionally original PDFs |
| Consumer runtime | GoFigure, Sheltrium, or another app | Pack verification and install, passage search, bounded cited answer generation |
| Host app | Each project | User accounts, access to packs/documents, model provider and key, usage limits, chat UI |

The live Disaster Clippy site's official collections remain cloud Pinecone packs. A `.clippypack` is a transferable local SQLite artifact for a receiving app or user's own library. Installing one does not change the live site's official collection settings.

## Build and hand off

From `clippy-core/`:

```bash
pip install -e ".[pdf,local]"
clippy preview -m examples/skating/sources.yaml --limit 40
clippy build -m examples/skating/sources.yaml --rebuild
clippy eval examples/skating/golden.yaml -m examples/skating/sources.yaml --compare
clippy pack-export -m examples/skating/sources.yaml -o build/skating.clippypack
clippy pack-inspect build/skating.clippypack
```

The manifest's `pack:` section supplies the pack ID, display name, short name, description, tags, origin, and license label. `pack-export` snapshots the SQLite index, checks that every declared document has passages and that the index has no undeclared documents, records counts and checksums, and creates the archive. It refuses to overwrite an existing archive. By default, only the prepared index is shared; `--include-pdfs` adds deduplicated original PDFs when the producer intends to distribute them. The SQLite index itself contains extracted source text, so the publisher's distribution terms still apply.

For the coming skating collection, add each new PDF under `documents:` with its own `path`, stable `doc_id`, `source_id`, `title`, `edition`, and public `url`. Keep document IDs distinct when different PDFs are parts of the same season. One manifest and one SQLite index can hold all five PDFs; a separate `pack:` block describes the collection as a whole. Preview each PDF's layout and select or adjust its chunker before rebuilding. `pack-export` will fail if one is missing from the built index, including when the PDF was missing or scanned with no extractable text. Keep the real PDFs outside Git.

## Receive and use

Copy the `.clippypack` file to the receiving project. If it has the full `clippy-core` package, the CLI can inspect and install it:

```bash
clippy pack-inspect skating.clippypack
clippy pack-install skating.clippypack --library ./packs
```

An app with only the exported consumer runtime installs it through Python:

```python
from clippy_core import ChatService, ClippyConfig, install_pack

pack = install_pack("skating.clippypack", "./packs")
store = pack.open_store(keyword_only=True)
config = ClippyConfig(search_mode="keyword", llm_provider="openai")
answer = ChatService(store, config=config).chat_sync("What counts as a fall?")
print(answer.text, answer.search_results)
```

The install checks the archive's member names and SHA-256 checksums, checks SQLite integrity and passage count, and places it under `packs/<pack_id>`. It refuses to replace an existing pack. The host chooses which installed packs a user can search and controls the chat model: OpenAI, Anthropic, or an adapter for another local or cloud provider. Keyword search needs no query embedding model. Semantic or hybrid search still needs a compatible query embedder at runtime; the archive records the build embedder and dimension.

`python export_runtime.py --output <new-folder>` creates the consumer package. It includes pack install/open code, search and chat, but no `pack-export`, PDF ingestion, producer CLI, source tools, or server. The host can also supply its own retrieved passages to `ChatService.answer_sync` when its documents live in another store.

## Versioning and updates

`pack_id` names an installed artifact. Use a new ID for a new season or revision that should coexist with the old one, for example `usfs-rules-2027-28`. An installer can select which pack IDs are available to each user. Replacement or migration of an existing ID is a host operation; the installer deliberately refuses to overwrite it.
