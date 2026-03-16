# QA Testing Guide

This folder contains quality assurance assets for testing Disaster Clippy as a product family, not just a single chat endpoint.

It now includes a lightweight runnable API suite in addition to the planning notes:

```text
QA/
  README.md
  run_api_qa.py
  product_surfaces.md
  datasets/
    api_contract_cases_v1.json
    questions_v1.json
  rubrics/
    grounded_answer_rubric.md
  reports/
    README.md
```

## What Exists Now

- `run_api_qa.py`
  - Runs API contract checks against `/api/v1/chat`, `/api/v1/chat/stream`, and `/api/v1/sources`
  - Runs a representative retrieval-style question set
  - Runs a basic session continuity check
  - Writes JSON and Markdown reports into `QA/reports/`
- `datasets/questions_v1.json`
  - First-pass representative question dataset for retrieval and answer smoke testing
- `datasets/api_contract_cases_v1.json`
  - Basic endpoint contract cases for source filtering and session-safe API behavior
- `rubrics/grounded_answer_rubric.md`
  - Manual review rubric for checking groundedness, actionability, and safety
- `product_surfaces.md`
  - Product-surface QA framing for hosted app, local runtime, advanced local admin, and the public site

## Goals

- Verify that hosted search returns relevant documents for real user questions
- Verify that responses stay grounded in retrieved context
- Verify source filtering and mode behavior do not regress
- Verify the project’s main user surfaces stay coherent:
  - hosted app
  - local runtime
  - advanced local admin
  - future source-pack/catalog surfaces
- Build a repeatable process for manual and automated QA

## Recommended QA Suite

Use 6 complementary test tracks:

1. Retrieval QA
2. Grounded Answer QA
3. API Contract QA
4. Mode Regression QA
5. Product Surface QA
6. Pack and Catalog QA

## 1) Retrieval QA

Purpose: Ensure top search results are relevant and diverse.

Test set:
- Start with 50 to 150 representative user questions
- Cover water, shelter, food, medical, power, sanitation, and emergency response
- Include simple, specific, and ambiguous queries

Per-question checks:
- At least one top result is clearly relevant
- Top 5 results are mostly on-topic
- Source diversity behavior is reasonable (no single source dominates unless filtered)

Suggested metrics:
- Recall@5 (expected source/topic appears in top 5)
- MRR (rank quality of first relevant result)

## 2) Grounded Answer QA

Purpose: Ensure responses are supported by retrieved documents.

Per-answer rubric (0-2 each):
- Faithfulness: Claims are supported by retrieved context
- Citation quality: Cited material is relevant and not fabricated
- Actionability: Steps are clear and useful in practical scenarios
- Safety: No obviously harmful advice for emergency contexts

Suggested pass condition:
- Average score >= 1.5 on each dimension
- Zero critical safety failures

## 3) API Contract QA

Purpose: Catch regressions in endpoint behavior.

Validate:
- `POST /api/v1/chat`
- `POST /api/v1/chat/stream`
- `GET /api/v1/sources`

Cases to include:
- Default all-sources behavior
- `source_mode=none` with valid `include`
- `source_mode=none` with empty `include` (expects no-sources message)
- Typos in source IDs (expects warning + fallback behavior)
- Contradictory include/exclude handling
- Session follow-up behavior using `session_id`

## 4) Mode Regression QA

Purpose: Verify consistency across connectivity modes.

Run the same question set in:
- `online_only`
- `hybrid`
- `offline_only`

Compare:
- Relevance quality
- Response completeness
- Fallback behavior in hybrid/offline scenarios
- Latency (optional, if you want performance baselines)

## 5) Product Surface QA

Purpose: Make sure Disaster Clippy still makes sense as a multi-surface product.

Test these surfaces separately:

- Hosted public app
  - users can ask useful questions immediately
  - app does not expose advanced admin workflows
- Local runtime
  - install assumptions are still valid
  - offline-first behavior remains first-class
- Advanced local admin
  - source creation, validation, translation, video processing, and future OCR workflows remain local-first
- Product/catalog site
  - language matches the real architecture
  - hosted vs local vs admin distinctions are clear

Checks to include:

- terminology consistency across docs and site pages
- no accidental leakage of maintainer-only concepts into the normal hosted path
- no misleading claims about hosted source creation

## 6) Pack and Catalog QA

Purpose: Treat source packs as a first-class release unit.

Checks to include:

- pack names and source IDs remain stable
- pack descriptions match actual content
- source pack docs and UI remain consistent with runtime behavior
- hosted and local surfaces describe the same pack model
- future catalog/profile/update manifests remain backwards-compatible

Suggested future metrics:

- pack profile completeness
- license metadata completeness
- freshness/update-history coverage
- installability and artifact integrity

## Quick Start

Run against a local app:

```powershell
python QA/run_api_qa.py --base-url http://127.0.0.1:8000
```

Run against the deployed app:

```powershell
python QA/run_api_qa.py --base-url https://disaster-clippy.up.railway.app --verify-ssl
```

Useful flags:

- `--skip-contract`
- `--skip-retrieval`
- `--skip-session`
- `--reports-dir QA/reports`

The script exits non-zero when checks fail, so it can later be used in CI or a deployment smoke test.

Current scope of `run_api_qa.py`:

- hosted/local API contract checks
- representative retrieval smoke checks
- session continuity checks

Not yet covered automatically:

- rendered site copy and docs QA
- pack catalog QA
- local admin workflow regression
- local/offline end-to-end validation

## Quick Start (Manual Baseline)

1. Run `python QA/run_api_qa.py --base-url <target>`.
2. Sample failures and borderline passes from the generated report.
3. Score answer quality with `rubrics/grounded_answer_rubric.md`.
4. Record important edge cases in `QA/reports/<date>_manual_baseline.md`.
5. Re-run the same set after search, prompt, indexing, or deployment changes.

## Example Question Dataset Schema

```json
[
  {
    "id": "q001",
    "question": "How can I purify water if I only have a pot and cloth?",
    "expected_topics": ["water", "sanitation"],
    "expected_sources": ["ready_gov_site", "appropedia"],
    "notes": "Should prioritize practical, step-based guidance."
  }
]
```

## Exit Criteria for a Release Candidate

- Retrieval Recall@5 meets your target baseline (define and track per run)
- No critical groundedness/safety failures in sampled questions
- Source filtering behavior passes all contract checks
- No major quality regressions across connection modes
- No major language drift between product docs, site pages, and runtime behavior
- No pack-model regressions in the public-facing explanation of hosted vs local use

## Near-Term Build Order

1. Expand `run_api_qa.py` into a clearer hosted/local regression harness.
2. Add a small site/docs QA pass for public wording and broken route checks.
3. Add pack/catalog QA fixtures as the site and control-plane model hardens.
4. Add local admin workflow smoke checks for source tools, translation, and video prep.
5. Later, add deeper offline-node validation for Raspberry Pi or air-gapped scenarios.

## Ideal LLM for Disaster Clippy

When benchmarking LLMs (especially offline), optimize for grounded question answering quality, not coding ability.

### What to Prioritize in Benchmarks

1. Groundedness under retrieval context
- Does the model stay within provided context instead of inventing facts?
- Does it correctly say "not enough information" when context is weak?

2. Instruction following
- Follows format constraints (clear steps, concise emergency guidance, citations when required)
- Honors source/mode constraints and safety tone

3. Practical reasoning for procedures
- Strong at multi-step how-to guidance
- Handles conditional logic ("if no fuel, then...", "if water is cloudy, then...")
- Can compare options with tradeoffs (speed vs safety vs equipment needed)

4. Robustness to noisy context
- Works when chunks are imperfect, partially redundant, or slightly off-topic
- Still extracts relevant details from mixed-quality retrieval results

5. Hallucination resistance
- Low rate of unsupported claims
- Low confidence language when uncertain

6. Latency and consistency on target hardware
- First-token and full-response times on your actual offline devices (RPi/local machine)
- Stable quality across repeated runs and prompt variants

7. Resource efficiency
- Fits RAM/VRAM constraints with acceptable quantization
- Sustained throughput without crashes or aggressive swapping

### LLM Abilities That Matter Most for Good QA

- Grounded summarization from retrieved passages
- Instruction following and constraint adherence
- Procedural reasoning (step-by-step tasks)
- Safety-aware response behavior
- Clarifying question behavior when user intent is ambiguous
- Stable formatting for actionable responses

### Abilities That Matter Less Here

- Advanced coding generation
- Software architecture reasoning
- Long code debugging performance

### Suggested Benchmark Categories for This Project

1. Product-specific eval set (highest priority)
- Your own emergency-preparedness questions + expected rubric outcomes
- Evaluate with retrieval context and without retrieval context

2. Long-context and retrieval-aligned tests
- Models should select relevant facts from multi-document context

3. Instruction and truthfulness tests
- Measure refusal to fabricate and correct uncertainty handling

4. Safety-focused scenario tests
- Medical/survival prompts where bad advice is high risk

### Practical Selection Rule

Pick the model that gives the best combined score on:
- Groundedness
- Actionability
- Safety
- Offline latency/resource fit

Then use coding ability only as a tie-breaker (if at all).
