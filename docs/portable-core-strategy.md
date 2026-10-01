# Portable clippy core: shared answer runtime

Status: design contract, 2026-10-01. The host-retrieved-passages path, separate host context, injected provider interface, and explicit source-boundary check are implemented; durable mounted-corpus manifests, citation carry-forward, turn metadata, and streaming reference events remain future work.

## What the experiments established

Disaster Clippy proves a public collection browser: official source packs live in Pinecone, users select collections, and a chat turn combines a bounded set of search passages with a model API response. Its existing hosted route does not call `clippy_core` yet.

Sheltrium proves the user-account pattern: a homeowner has a private advisor in their workspace, property/location selects relevant sources, and profile/property data personalizes the answer. Its active `/api/chat` route uses Supabase pgvector and OpenAI directly; the copied February 2026 `clippy_core` folder is not on that route. See [Sheltrium integration review](sheltrium-integration-review.md).

The skating experiment proves that a large, structured PDF can become a prepared, page-linked index. GoFigure should consume that index and add skating-specific context such as discipline, event, level, and rulebook edition in its own application. A rulebook passage is source evidence; a user's profile is personalization context. Keep those distinct in the prompt and response.

## Boundary and ownership

| Component | Owns | Does not own |
| --- | --- | --- |
| Producer tools | PDF extraction, OCR handoff, chunking, metadata validation, embedding, index build, retrieval evaluation | User accounts or live chat bills |
| Portable runtime | Search over a supplied prepared store **or** acceptance of already retrieved passages; bounded context; answer generation through an injected local/cloud provider or configured cloud client; citation provenance and turn results | Scrapers, PDF upload flows, public source curation, authentication, billing, UI |
| Host app | Authentication, tenant/property access, selected packs and project documents, user context, chat persistence, chosen model provider and its credentials/runtime, rate limits, token and spending budgets, transport, UI | Parsing or reindexing source PDFs during a chat turn |

Keep the producer and runtime in the same upstream `clippy-core` project, but export the consumer-only runtime as a separate folder or wheel. A host should depend on a versioned package or exported artifact, never copy a stale source tree and assume its active route uses it. The runtime bundles no local inference engine; a host can inject one through `LLMBackend`. Keyword search must remain usable without a local embedding model; semantic search requires a query embedder compatible with the prepared vectors.

## Host-facing contract to implement

One turn should have the same structure for all three applications:

```text
host authenticates caller and mounts allowed official and project documents
  -> host loads private user/property/skater context, if any
  -> core searches a supplied prepared store OR accepts host-retrieved SearchResult passages
  -> core chooses the bounded evidence set and rehydrates any cited passage needed from history
  -> core formats evidence and separate personalization context
  -> host-selected local or cloud model generates the answer
  -> core returns answer, exact visible references, model/usage metadata, and grounding status
  -> host persists the turn and renders it
```

The public API should support both `chat(search_store=..., ...)` and `answer(results=..., ...)` without making a host adapt its database to SQLite. The latter is the small seam Sheltrium needs for its existing `search_sources` RPC; the former is the easiest GoFigure path with the prepared skating index. Both paths should return one consistent turn object. Search adapters for Pinecone and Supabase can live outside the minimal runtime package.

Inputs should be explicit: question, selected/authorized source IDs, optional metadata filters (for example edition or discipline), prior turns with cited passage IDs, host context data, an evidence limit, and a model adapter/configuration. The host must apply authorization before passing passages or private context. A previous assistant answer is useful for resolving a follow-up but is not itself a source document.

The result should contain answer text, the exact passages shown to the model, stable passage IDs, display names and links, any inability to verify a claim, and model identity/token usage when the provider returns them. Streaming should expose reference metadata with the turn instead of yielding only text chunks; the host can then render the same citations for streamed and non-streamed answers. No API keys, full private prompts, or secrets belong in the public response.

## Passage and pack contract

A prepared passage needs stable `passage_id`, `source_id`, `doc_id`, `edition`, content, title, section ID, page range, and canonical URL. Source packs also need the short name and 3–7 topic tags used by the Disaster Clippy collection UI. Host-specific fields (Sheltrium location/risk applicability, GoFigure discipline/event/level) belong in metadata or host filters, not hard-coded core branches. A source version or checksum makes citation carry-forward unambiguous after a rulebook edition changes.

Each user or project may mount its own prepared documents alongside permitted official packs. Record origin and owner scope in the host's corpus manifest, and do not promote a private document into the official catalog. The host can point the runtime at a project-specific SQLite index or pass already retrieved passages; the same answer contract applies. The producer tools prepare new PDFs outside the chat path. Source IDs and document IDs must remain stable within an owner scope so a later turn can rehydrate a cited passage.

For each answer, the references shown in the UI must be the passages in that answer's evidence set. If a follow-up relies on a rule from an earlier turn, retrieve it by stable ID, check that its source is still selected and authorized, include its text in the current bounded context, and show its link again in the latest References list. If it cannot be retrieved, do not cite it as verified. Candidate retrieval can be wider than the evidence limit (as Disaster Clippy does with 15 candidates and five displayed passages), but the model must never receive hidden candidate passages. A citation validator should detect IDs or links outside the current evidence set before publishing an answer.

## Three host configurations

| Host | Retrieval and evidence | Personalization | Model and transport |
| --- | --- | --- | --- |
| Disaster Clippy | Official Pinecone packs, selection filters, five visible passages | Collection preferences and chat history | Hosted API client, SSE; existing app remains on its current route until parity is demonstrated |
| Sheltrium | Supabase `search_sources` results, property-eligible source IDs | Authenticated profile/property/risk/equipment/survey data | Host-owned API client, JSON initially; preserve property authorization and cost controls |
| GoFigure | Prepared skating SQLite index first, optionally a remote store later; filter by rulebook edition and discipline | Skater/event context supplied by GoFigure | Host-owned API client, chosen chat transport; no PDF build or local model in the app |

The existing Disaster Clippy Railway `OPENAI_MODEL` setting and Sheltrium's hard-coded `CHAT_MODEL` are independent. Model switching belongs in each host configuration. A host can inject an `LLMBackend` implemented with OpenAI, Anthropic, another cloud API, or a local model server. The core package does not need to install Ollama or ship model weights. A host should be able to log the provider-returned model ID and usage for a turn without exposing prompts or keys. Context and output budgets may differ by model, especially for local runtimes.

## Lessons from Global Map Research mode

Global Map's Research design separates three states: sources available in the broader catalog, sources mounted in a workspace, and artifacts active for the current analysis. Clippy should make the equivalent boundary visible: **available documents**, **mounted corpus**, and **passages used in this answer**. Search must stay inside the mounted corpus, and the answer must cite only its current evidence set. Source discovery can suggest additions, but mounting another document is an explicit host action rather than an automatic side effect of a model response.

For the first host integration, a normalized list of selected source/document IDs is enough. The current runtime checks an explicit `sources=` boundary and fails closed if the store or host supplies a passage outside it; the host remains responsible for determining which IDs the user is allowed to select. A future saved-corpus feature can add two identities: a session/user-facing `corpus_id` and a content-derived `corpus_fingerprint` based on sorted membership plus document/index versions and hashes. The fingerprint excludes user identity, question text, and chat history; the host keeps ownership and access policy on the corpus handle. Answers can record that fingerprint to detect when a saved reference points to an older rulebook edition.

The model should see a compact corpus manifest and only the passages needed for the current question, rather than every page of a 450-page PDF. A longer research session can keep structured working state (current question, comparison, selected documents, cited passage IDs) separately from raw chat history. Summarize older conversational turns when needed, but re-fetch source passages before citing them. Research-mode notes, briefs, and exports belong to the host's workspace; the core supplies traceable turn evidence.

These ideas come from Global Map's `research_mode.md`, `research_mcp_source_bound_corpus.md`, and `LLM_CONTEXT_MODEL.md` in `county-map-private/docs`. They are a reusable scope and provenance pattern, not a dependency on Global Map's geographic data or UI.

## Migration order and checks

1. **Freeze an evidence contract.** Add stable passage IDs, owner/origin scope, a mounted-corpus manifest, and a turn result envelope; specify selected-source and edition filters. Use the current skating golden questions and Sheltrium's FORTIFIED queries as examples.
2. **Add the host-retrieved-passages path.** `ChatService.answer` now accepts supplied `SearchResult` objects and separate host context, while `ChatService.chat` keeps the prepared-SQLite search path. Both return the same reference format for non-streamed answers. A host adapter still must map its database rows and authorize them.
3. **Add citation continuity.** Rehydrate a passage cited in history when needed, include it in the current evidence budget, and reject unsupported citations. Test a follow-up in the same chat and the same question in a fresh chat.
4. **Make model and cost visibility explicit.** The injected `LLMBackend` seam is implemented; add provider-reported model and token usage to the turn result. Let the host set a per-turn output cap and enforce account budgets before generation. Test each chosen cloud or local adapter against its context and streaming limits.
5. **Build thin host adapters.** GoFigure should consume a prepared skating index first. Sheltrium can keep its Supabase RPC and replace only its answer assembly with the core. Move Disaster Clippy's hosted route last, after selection, streaming, and citation parity checks.
6. **Evaluate end to end.** Check retrieval hit rate, answer correctness, citation-to-card agreement, edition/discipline scope, latency, and cost per successful answer on skating and FORTIFIED questions. Include account-context and unauthorized-property cases for Sheltrium.

Do not retire any working route merely because the portable API exists. The migration is complete for a host when its answer, visible references, source filters, and access controls match or improve on its current behavior.
