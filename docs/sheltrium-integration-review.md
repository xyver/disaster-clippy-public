# Sheltrium integration review

Read-only inspection of `C:\Users\Bryan\Desktop\webtest` on 2026-10-01. This records what the checked-out code does; Sheltrium's deployment settings and live traffic were not inspected.

## The active path

Sheltrium's advisor frontend (`app/js/advisor-chat.js`) sends the current question, selected source IDs, property ID, and recent chat history to `/api/chat`. Its Python route (`server/routes/chat.py`) loads account/property context from Supabase, embeds the question with `text-embedding-3-small`, calls the `search_sources` pgvector RPC for five results, builds an 8,000-character maximum source context, and calls OpenAI with a hard-coded `gpt-4o-mini` model. It returns the answer and those search results to the frontend. If `/api/chat` fails, the frontend falls back to a direct, simple Supabase text search and labels that result as testing mode.

The older `supabase/functions/chat/index.ts` implements a similar direct pipeline, but the inspected frontend calls `/api/chat`. The older `clippy_core/` folder and `clippy_wrapper.json` are not imported by the active frontend or Python chat route. The wrapper's claim that the Edge Function imports the Python folder is stale.

## Where the projects diverged

| Concern | Disaster Clippy / current `clippy-core` | Sheltrium checkout |
| --- | --- | --- |
| Product role | Public cited knowledge browser; hosted official packs in Pinecone, local packs for users | Property-aware advisor inside user, supplier, expert, and admin workspaces |
| Source selection | User-selected collections | Property/location-derived sources plus frontend selection |
| Active chat | `admin/ai_service.py` streams a response over the app endpoint; current `clippy-core` remains parallel | `server/routes/chat.py` calls OpenAI directly and returns one JSON response |
| Search | Hosted Pinecone search gets up to 15 candidates, then selects five for the prompt and cards | Supabase `search_sources` pgvector RPC requests five results directly |
| Personal context | Primarily question, selected sources, and conversation | Supabase profile, property, risk, equipment, and survey fields in the prompt |
| Model | Hosted app selects `OPENAI_MODEL` (with a `gpt-4o-mini` fallback); Luna compatibility is in the app | `CHAT_MODEL = "gpt-4o-mini"` in the Python route; changing Disaster Clippy's Railway variable does not change this |
| Portable package | Newer producer package has PDF indexing, SQLite search, optional PgVector, evaluation, and a consumer-only runtime export | Copied `clippy_core` reports version 0.1.0 and is a different, older tree; it is not the route's runtime |
| Ingestion | Disaster Clippy has separate source tools and prepared pack workflow | Sheltrium has its own scripts and Supabase source table/vector workflow |

The common `VectorStoreBase` file is byte-identical, but the copied package's `chat.py`, `config.py`, `llm.py`, `schemas.py`, and PgVector implementation differ from the current package. Folder replacement is therefore not a safe upgrade strategy. Sheltrium's `docs/chatbot.md` and `clippy_wrapper.json` describe an intended wrapper integration, while the working route uses direct OpenAI and Supabase calls.

## Integration boundary to preserve

Use the exported prepared-index runtime when a Sheltrium feature receives prepared local data. If Sheltrium keeps its Supabase `search_sources` RPC, put a small adapter at the vector-store boundary and keep property context, source selection, auth, quotas, and model credentials in Sheltrium's server. Do not copy the old Sheltrium `clippy_core` tree back into Disaster Clippy or replace it wholesale with the current package.

Both active chat paths send previous assistant answers to the model alongside current search results. Apply the [citation continuity rule](clippy-core-extraction.md#citation-continuity-across-chat-turns) in each host: a rule reused from history must be retrieved into the current context and displayed in the latest answer's references. Sheltrium currently returns the five search hits as sources, but does not validate that each citation in the generated answer belongs to them.

Before using Sheltrium's Python route for private property context, verify server-side authorization for the requested `property_id`. The inspected `/api/chat` route reads property/profile data with a Supabase service-role client and does not itself check that the caller owns that property. Frontend login alone cannot enforce that boundary.
