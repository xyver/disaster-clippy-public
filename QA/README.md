# QA Testing Guide

This folder contains quality assurance assets for testing Disaster Clippy question-answer performance.

## Goals

- Verify that search returns relevant documents for real user questions
- Verify that responses stay grounded in retrieved context
- Verify source filtering and mode behavior do not regress
- Build a repeatable process for manual and automated QA

## Recommended QA Suite

Use 4 complementary test tracks:

1. Retrieval QA
2. Grounded Answer QA
3. API Contract QA
4. Mode Regression QA

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

## Folder Structure

Suggested structure as the suite grows:

```text
QA/
  README.md
  datasets/
    questions_v1.json
  rubrics/
    grounded_answer_rubric.md
  reports/
    2026-03-08_manual_baseline.md
```

## Quick Start (Manual Baseline)

1. Create `QA/datasets/questions_v1.json` with 20 representative questions.
2. Run each question through `/api/v1/chat`.
3. Score retrieval and answer quality with the rubric above.
4. Record failures and edge cases in `QA/reports/<date>_manual_baseline.md`.
5. Re-run the same set after search/prompt/indexing changes.

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
