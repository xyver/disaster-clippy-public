# Document Type Weighting System

Research and planning document for implementing document-level type classification and search weighting.

---

## Table of Contents

1. [Overview](#overview)
2. [Current State](#current-state)
3. [Proposed System](#proposed-system)
4. [PDF Document Challenges](#pdf-document-challenges)
5. [OCR for Scanned PDFs](#ocr-for-scanned-pdfs)
6. [Classifier Architecture](#classifier-architecture)
7. [Code Locations](#code-locations)
8. [Open Questions](#open-questions)
9. [Decisions Made](#decisions-made)

---

## Overview

**Goal:** Classify documents by type (guide, academic, report, etc.) at the per-document level during indexing, then use those classifications to weight search results based on user intent.

**Use Cases:**
- Researcher wants academic papers and studies -> boost academic content
- Someone responding to a situation wants reports -> boost after-action reports, case studies
- Someone building something wants hands-on guides -> boost tutorials, how-tos
- Quick reference lookup -> boost checklists, reference material

---

## Current State

### Existing Frontend Weighting (app.py)

The system already has basic query-based intent detection:

**Pipeline:**
```
User query
    |
    v
search_articles(n_results=15)     <- Retrieves 15 candidates
    |
    v
detect_doc_type_preference()      <- Analyzes QUERY for intent keywords
    |
    v
prioritize_results_by_doc_type()  <- Applies score boost/penalty
    |
    v
ensure_source_diversity()         <- Limits to 5, max 2 per source
    |
    v
Final 5 results returned
```

**Current Intent Detection Keywords (app.py lines 2070-2114):**

| Detected Intent | Keywords |
|-----------------|----------|
| Academic | "research", "study", "paper", "journal", "peer-reviewed" |
| Product | "buy", "purchase", "ready-made", "where to buy" |
| Article | "what is", "explain", "how does it work" |
| Guide | "how to", "build", "DIY", "instructions" (default) |

**Current Scoring:**
- Base score: vector similarity (0-1)
- Matching doc type: +0.15 boost
- Non-matching: -0.05 penalty

**Limitations:**
- Detects intent from query only, not from actual document content
- Documents have no stored type classification
- Limited type categories
- No confidence scoring

---

## Proposed System

### Document Type Categories

**Hands-On / Building:**
- Guide/How-To - Step-by-step instructions, tutorials, DIY builds
- Manual - Equipment/product documentation, technical specs
- Field Guide - Visual identification (plants, materials, symptoms)

**Research / Learning:**
- Academic Paper - Peer-reviewed studies, scientific research
- Article - Informational content, explainers, wiki entries
- Reference - Encyclopedic entries, data sheets, specifications
- Training Material - Curricula, courses, educational modules

**Responding / Acting:**
- After-Action Report - Post-incident analysis, lessons learned
- Case Study - Detailed analysis of specific events/implementations
- Protocol/SOP - Medical protocols, standard operating procedures
- Checklist - Quick-reference cards, condensed action lists

**Planning / Preparing:**
- Plan/Template - Emergency plans, preparedness forms
- Policy/Standard - Government guidelines, regulations, codes
- Assessment - Risk assessments, vulnerability analyses

**Other:**
- News/Update - Time-sensitive situational information
- FAQ - Q&A format troubleshooting
- Multimedia - Video tutorials, diagrams, infographics

### Per-Document Storage

Store in `_metadata.json` at document level with top 3 types and confidence:

```json
{
  "documents": {
    "doc_id_1": {
      "title": "Solar Water Heater Construction",
      "url": "/wiki/Solar_Water_Heater",
      "internal_links": [...],
      "document_types": [
        {"type": "guide", "confidence": 0.65},
        {"type": "report", "confidence": 0.25},
        {"type": "reference", "confidence": 0.10}
      ],
      "document_type_source": "content_analysis"
    }
  }
}
```

### Proposed Pipeline

```
1. Backup (get files)
2. Scan backup (backup_manifest.json)
3. Configure source (_manifest.json)
4. Generate metadata (_metadata.json) - doc listing, titles, URLs
5. [NEW] Classify documents (_classification.json or in _metadata.json)
6. Create index (_index.json, _vectors.json) - embeddings include type context
```

**Classification must happen before indexing** so that:
- Type information can be embedded in vectors
- Search can combine semantic similarity with type matching

### Classification Approaches

**Content Pattern Matching (rule-based):**

| Signal | Indicates |
|--------|-----------|
| Numbered steps, "First... Then... Finally" | Guide |
| "Abstract", "Methods", "Conclusion", citations | Academic |
| Dates, locations, "lessons learned", incident names | Report |
| Definitions, specs, lists, tables | Reference |
| "What is...", Q&A format | FAQ |
| News language, recent dates, quotes | Article |

**LLM Classification:**
- Send content sample (first ~1000 chars + headings) to LLM
- Ask for top 3 types with confidence scores
- More accurate but slower/costly
- Could batch or sample

**Hybrid:**
- Rule-based first pass
- LLM for low-confidence items
- Manual review for edge cases

### Search Integration Options

**Option A: Bake types into embedding text**
```
Prepend type context when creating embeddings:
"[DOCUMENT TYPE: guide 0.65, report 0.25] Solar water heaters..."
```
Vectors naturally capture type information.

**Option B: Metadata filtering + re-ranking (recommended)**
```
1. Vector search returns candidates
2. Load doc types from in-memory metadata dict (O(1) lookup)
3. Apply type score boost based on detected user intent
4. Re-rank and return
```

Benefits of metadata-only approach:
- Can tune weights without re-indexing
- Cleaner embeddings (type tokens don't dilute semantic meaning)
- Dictionary lookup is negligible compared to vector search time
- Enables A/B testing of different weighting schemes

**Option C: Hybrid**
```
- Light type hints in embeddings
- Full type scores in metadata
- Query-time: combine vector_similarity * 0.7 + type_match * 0.3
```

### Retrieval Pool Size

**Current:** Retrieve 15 candidates, filter to 5

**Problem:** With only 15 candidates, type weighting has limited impact. The best type-matched document might be at position 16-25 in vector similarity and never gets seen.

**Recommended:** Widen to 25-30 candidates

```
1. vector_search(n=30)                      # wider net
   |
2. similarity_floor(min=0.45)               # drop semantically weak matches
   |
3. type_boost(user_intent, DOC_TYPES)       # metadata lookup, rerank
   |
4. ensure_source_diversity(max_per_source=2)
   |
5. return top 5
```

**Performance impact:** Negligible. Going from 15 to 30 candidates adds ~5-10ms total. The LLM response generation is the bottleneck, not retrieval.

**Similarity floor rationale:** Prevents type weighting from promoting semantically weak matches. A "Guide to Basket Weaving" with 0.35 similarity shouldn't surface for "how to purify water" just because user wants guides.

### Frontend Options (future consideration)

1. **Smart chat detection** - AI detects intent from query phrasing
2. **Separate frontends** - "Guide Search" vs "Research Search" with baked-in weights
3. **Mode selector** - Single UI with dropdown to select mode
4. **Progressive** - Start general, offer refinement options

---

## PDF Document Challenges

PDFs present unique challenges compared to HTML documents.

### The Core Problems

**1. Chunking strategy varies by document size:**

| PDF Type | Size | Chunks at 4000 chars | Challenge |
|----------|------|---------------------|-----------|
| One-page article | 2-5 pages | 1-3 chunks | Easy, treat as single doc |
| Research paper | 10-50 pages | 15-75 chunks | Sections have different types |
| Technical book | 100-500 pages | 150-750 chunks | Chapters are essentially separate documents |

**2. Different sections need different classifications:**

A 50-page research paper might contain:
- Abstract/Introduction -> "academic"
- Literature Review -> "reference"
- Methods/Procedures -> "guide"
- Results/Analysis -> "report"
- Appendix with checklists -> "checklist"

Fixed 4000-char chunking ignores these boundaries.

**3. Linking to middle sections:**

When a user searches and we match chunk 47 of a 200-page PDF, how do we:
- Link them to the right location?
- Show context around the match?
- Let them navigate the full document?

### PDF Linking Options

**Browser-based PDF display supports page navigation:**

```
Local file:
  file:///path/to/document.pdf#page=47

Web-hosted:
  https://example.com/docs/document.pdf#page=47

Cloud storage (S3, etc.):
  Requires PDF viewer component (PDF.js) with page parameter
```

**Named destinations (if PDF has them):**
```
document.pdf#nameddest=chapter3
document.pdf#nameddest=methods_section
```
Most PDFs don't have named destinations unless specifically created with them.

### Proposed PDF Metadata Structure

Store chunk-to-page mapping alongside classification:

```json
{
  "pdf_chunks": {
    "chunk_abc123": {
      "parent_doc": "emergency_medicine_guide.pdf",
      "page_start": 47,
      "page_end": 49,
      "section_title": "Chapter 5: Wound Care",
      "char_offset": 184000,
      "document_types": [
        {"type": "guide", "confidence": 0.75},
        {"type": "protocol", "confidence": 0.20}
      ]
    }
  },
  "pdf_documents": {
    "emergency_medicine_guide.pdf": {
      "total_pages": 342,
      "total_chunks": 127,
      "primary_type": "manual",
      "url": "/pdfs/emergency_medicine_guide.pdf"
    }
  }
}
```

### Chunking Strategies to Consider

**Option A: Fixed size (current)**
- Simple, consistent
- Ignores document structure
- Sections split arbitrarily

**Option B: Section-aware chunking**
- Parse PDF structure (headings, page breaks)
- Chunk at logical boundaries
- Requires PDF structure analysis (not all PDFs have good structure)

**Option C: Hybrid chunking**
- Use section boundaries when detectable
- Fall back to fixed size with overlap
- Store section context even for fixed chunks

**Option D: Hierarchical indexing**
- Index at multiple granularities: document, chapter, section, chunk
- Search returns most relevant granularity
- More complex but handles varied document sizes

### Display Options for Users

**1. Embedded PDF viewer (PDF.js)**
- Opens at specific page
- User can scroll/navigate
- Works offline if PDF is local
- Requires viewer component in frontend

**2. Page image extraction**
- Pre-render pages as images during indexing
- Show relevant page(s) as preview
- Link to full PDF for context
- Storage overhead for images

**3. Text excerpt with PDF link**
- Show matched text chunk
- "View in context" links to PDF#page=N
- Simpler, relies on browser PDF handling

**4. Hybrid preview**
- Text excerpt for quick reading
- Thumbnail of the page
- Link to full PDF at correct page

### Integration with Existing Pipeline

The unified pipeline (see [source-tools.md](source-tools.md)) already handles PDFs:

```
scan_backup() -> _scan_pdf_backup()      # backup_manifest.json
generate_metadata() -> _generate_pdf_metadata()  # _metadata.json
create_index() -> _index_pdf()           # _index.json, _vectors.json
```

**Where classification fits:**

```
[Existing]
1. _scan_pdf_backup()         -> backup_manifest.json (file listing)
2. _generate_pdf_metadata()   -> _metadata.json (doc listing, titles)

[NEW - insert here]
3. _classify_documents()      -> Add document_types to _metadata.json

[Existing]
4. _index_pdf()               -> _index.json, _vectors.json (embeddings)
```

Classification must happen BEFORE indexing so type info can optionally influence embeddings.

**Extended _metadata.json for PDFs:**

```json
{
  "documents": {
    "emergency_guide_chunk_047": {
      "title": "Chapter 5: Wound Care",
      "url": "/pdfs/emergency_guide.pdf#page=47",
      "parent_pdf": "emergency_guide.pdf",
      "page_start": 47,
      "page_end": 49,
      "chunk_index": 47,
      "total_chunks": 127,
      "document_types": [
        {"type": "guide", "confidence": 0.75},
        {"type": "protocol", "confidence": 0.20}
      ]
    }
  },
  "pdf_parents": {
    "emergency_guide.pdf": {
      "total_pages": 342,
      "total_chunks": 127,
      "primary_type": "manual",
      "file_path": "pdfs/emergency_guide.pdf"
    }
  }
}
```

### Browser PDF Linking - What Actually Works

**Yes, browsers support page navigation:**

```
# Chrome, Firefox, Edge all support:
document.pdf#page=47

# Some also support zoom and named destinations:
document.pdf#page=47&zoom=100
document.pdf#nameddest=chapter5
```

**Local files (offline mode):**
```
file:///D:/disaster-backups/pdfs/guide.pdf#page=47
```
Works in browser if opened directly, but may need a local server to serve it.

**Cloud-hosted (R2/S3):**
```
https://your-bucket.r2.dev/pdfs/guide.pdf#page=47
```
Works directly - browser loads PDF, jumps to page.

**Embedded viewer (PDF.js):**
```html
<iframe src="/pdfjs/web/viewer.html?file=/pdfs/guide.pdf#page=47"></iframe>
```
More control, works offline, consistent across browsers.

### Recommended PDF Serving Approach

Similar to how ZIM and HTML backups have dedicated servers:

| Source Type | Server | Endpoint |
|-------------|--------|----------|
| ZIM | `admin/zim_server.py` | `/zim/{source_id}/{path}` |
| HTML Backup | `admin/backup_server.py` | `/backup/{source_id}/{path}` |
| PDF | `admin/pdf_server.py` (new) | `/pdf/{source_id}/{filename}#page=N` |

**PDF server features needed:**
- Serve PDF files with correct MIME type
- Accept page parameter for deep linking
- Option to embed PDF.js viewer for consistent experience
- "Back to Search" navigation (like ZIM/HTML servers)

### Open Questions for PDFs

1. **Section detection** - Can we reliably detect chapter/section boundaries in arbitrary PDFs?

2. **Per-chunk vs per-document classification** - Should a 200-page book have one classification or per-chapter?

3. **Storage location** - PDFs in cloud vs local affects linking strategy

4. ~~**Chunk overlap** - Should chunks overlap to preserve context at boundaries?~~ DECIDED: 300 char overlap

5. **Large document handling** - At what size should we treat chapters as separate documents?

6. ~~**OCR PDFs** - Scanned documents have different chunking challenges than text-native PDFs~~ DOCUMENTED: See "OCR for Scanned PDFs" section above

7. **PDF.js vs native** - Should we embed PDF.js for consistency, or rely on browser native PDF viewing?

8. ~~**Chunk-to-page accuracy** - Current 4000-char chunking doesn't track page boundaries.~~ IMPLEMENTING: Page-aware extraction

---

### PDF Implementation Decisions (Dec 2024)

**Decided:**
1. **Extract PDFs from ZIMs to child sources** - e.g., `appropedia_pdfs/` - faster for linking than serving from ZIM
2. **300-char chunk overlap** - preserves context at chunk boundaries
3. **Page-aware extraction** - track page_start, page_end for each chunk
4. **Browser-navigable URLs** - use `#page=N` instead of `#chunk-N`
5. **pdf_server.py** - new server following backup_server.py pattern

**Implementation Files:**
- `offline_tools/indexer.py` - PDFIndexer class (new methods added)
  - `_extract_text_with_pages()` - preserves page boundaries
  - `_extract_enhanced_metadata()` - includes page_count, file_size
  - `_chunk_pages_with_overlap()` - 300-char overlap, page tracking
  - `index()` - updated to use new methods and `#page=N` URLs
- `admin/pdf_server.py` - serves PDFs with browser page navigation
- `offline_tools/schemas.py` - PDF-specific fields in DocumentMetadata
- `app.py` - pdf_router registered for `/pdf/{source_id}/{filename}` endpoint

**New Metadata Fields for PDF Chunks:**
```python
{
    "parent_pdf": "emergency_guide.pdf",
    "page_start": 47,
    "page_end": 49,
    "chunk_index": 3,
    "total_chunks": 127,
    "total_pages": 342,
}
```

**URL Pattern Change:**
- OLD: `/api/pdf/{source_id}/{filename}#chunk-3`
- NEW: `/pdf/{source_id}/{filename}#page=47`

---

### OCR for Scanned PDFs

Scanned PDFs (book scans, historical documents) have no text layer and return 0 characters from extraction. These require OCR preprocessing before indexing.

**See:** [ROADMAP.md](../ROADMAP.md#ocr-for-scanned-pdfs) for implementation plan, recommended tools (OCRmyPDF, Tesseract, pdftoppm), and usage examples.

---

## Classifier Architecture

### Option A: Modular Sub-Functions (Recommended)

Separate detector for each category group:

```python
class DocumentClassifier:
    def classify(self, content: str, title: str, url: str) -> list[TypeScore]:
        scores = []
        scores.extend(self._detect_guide_signals(content, title))
        scores.extend(self._detect_academic_signals(content, title))
        scores.extend(self._detect_report_signals(content, title))
        scores.extend(self._detect_reference_signals(content, title))
        scores.extend(self._detect_planning_signals(content, title))
        # Normalize and return top 3
        return self._normalize_top_n(scores, n=3)

    def _detect_guide_signals(self, content, title) -> list[TypeScore]:
        """Detect: guide, manual, field_guide"""
        # Numbered steps, imperative verbs, materials lists...

    def _detect_academic_signals(self, content, title) -> list[TypeScore]:
        """Detect: academic, article, reference, training"""
        # Abstract, citations, methodology sections...
```

**Pros:**
- Test each detector independently
- Tune weights per category without affecting others
- Add new types without touching existing code
- Run detectors in parallel
- Easier debugging ("why was this classified as X?")

**Cons:**
- More files/functions to maintain
- Need to handle cross-category normalization

### Option B: Monolithic Function

One large function with all detection logic:

```python
def classify_document(content: str, title: str, url: str) -> list[TypeScore]:
    scores = {}
    # All pattern matching in one place
    if has_numbered_steps(content):
        scores["guide"] = scores.get("guide", 0) + 0.3
    if has_abstract(content):
        scores["academic"] = scores.get("academic", 0) + 0.4
    # ... hundreds of rules ...
    return normalize_top_n(scores, n=3)
```

**Pros:**
- Single file, simple to find everything
- No coordination between modules

**Cons:**
- Hard to test individual detection logic
- Changes risk breaking unrelated detections
- Becomes unwieldy as rules grow

### Option C: Hybrid - Rule-based + LLM

```python
class DocumentClassifier:
    def classify(self, content: str, title: str, url: str) -> list[TypeScore]:
        # Fast rule-based first pass
        rule_scores = self._rule_based_classify(content, title)

        # If confident enough, return early
        if rule_scores[0].confidence > 0.7:
            return rule_scores

        # Otherwise, use LLM for uncertain cases
        llm_scores = self._llm_classify(content[:2000], title)
        return self._merge_scores(rule_scores, llm_scores)
```

**Recommendation:** Start with Option A (modular), add LLM fallback (Option C) for low-confidence cases.

---

## LLM Selection for Classification

### Task Requirements

Document classification is a constrained task:
- Fixed set of output labels (17 types)
- Confidence scoring (0-1 scale)
- Input: ~1000-2000 chars of content + title
- No open-ended generation needed

This is well-suited for smaller models.

### Llama 3.2 3B Assessment

**Strengths:**
- Classification is a simpler task than generation - 3B can handle it
- Fast inference (important for batch processing thousands of docs)
- Runs locally, no API costs
- Constrained output format reduces hallucination risk

**Concerns:**
- May struggle with nuanced distinctions (guide vs manual vs field_guide)
- Confidence calibration might be poor (says 0.9 when it should be 0.6)
- May need careful prompting to get structured output

**Recommended Approach:**

```
Prompt structure:
---
Classify this document into types. Return JSON only.

Types: guide, manual, field_guide, academic, article, reference,
training, report, case_study, protocol, checklist, plan, policy,
assessment, news, faq, multimedia

Document title: {title}
Content preview:
{first_1500_chars}

Return format:
{"types": [{"type": "X", "confidence": 0.X}, ...], "reasoning": "brief"}
---
```

**Confidence Calibration:**
- Don't trust raw LLM confidence numbers directly
- Use temperature=0 for consistency
- Could calibrate by testing on labeled samples
- Or: ask for reasoning, derive confidence from reasoning quality

### Alternative Local Models

| Model | Size | Speed | Classification Quality |
|-------|------|-------|----------------------|
| Llama 3.2 3B | 3B | Fast | Good for clear cases |
| Llama 3.2 8B | 8B | Medium | Better nuance |
| Mistral 7B | 7B | Medium | Strong reasoning |
| Phi-3 Mini | 3.8B | Fast | Good instruction following |

### Testing Plan

1. Create labeled test set (~50-100 docs with known types)
2. Run Llama 3.2 3B classification
3. Measure accuracy and confidence calibration
4. Compare with rule-based baseline
5. Decide on hybrid threshold

---

## Code Locations

### Search and Filtering

| Location | Purpose |
|----------|---------|
| `app.py:880-910` | `search_articles()` - unified search interface |
| `app.py:1018-1031` | Chat endpoint, calls search with `n_results=15` |
| `app.py:2070-2114` | `detect_doc_type_preference()` - query intent detection |
| `app.py:2268-2303` | `prioritize_results_by_doc_type()` - score boosting |
| `app.py:2311-2374` | `ensure_source_diversity()` - limits results, diversifies sources |
| `admin/ai_service.py:147-284` | `AIService.search()` - core search dispatcher |

### Indexing Pipeline

| Location | Purpose |
|----------|---------|
| `offline_tools/source_manager.py` | SourceManager - unified pipeline dispatch |
| `offline_tools/indexer.py` | HTMLBackupIndexer, ZIMIndexer, PDFIndexer |
| `offline_tools/vectordb/store.py:229-286` | ChromaDB semantic search |

### Key Hardcoded Values

| Value | Location | Current |
|-------|----------|---------|
| Initial retrieval count | `app.py:1027` | `n_results=15` |
| Final result count | `app.py:1041` | `total_results=5` |
| Max per source | `app.py:1041` | `max_per_source=2` |
| Doc type priority boost | `app.py:2288` | `+0.15` |
| Doc type penalty | `app.py:2288` | `-0.05` |

---

## Test Results

### Test 1: builditsolar (DIY Solar Projects)

**Source:** `D:\disaster-backups-local\builditsolar` (~930 documents)

**Classifier v1 Results:**

| Type | Count | Percentage |
|------|-------|------------|
| misc | 747 | 80% |
| guide | 163 | 17.5% |
| academic | 23 | 2.5% |

**Problem:** 80% misc is too high for a DIY solar site. Documents like "ProMaster Camper Van Conversion -- Plumbing" and "Solar Wall or Trombe Wall" were classified as misc instead of guide.

**Root cause:** Classifier v1 was too strict, looking for explicit patterns (numbered steps, imperative verbs) while builditsolar content is often narrative/descriptive.

**Classifier v2 Results (after adding looser patterns):**

| Type | Count | Change |
|------|-------|--------|
| misc | 524 | -223 (56%) |
| guide | 262 | +99 |
| field_guide | 34 | new |
| case_study | 31 | new |
| reference | 29 | new |
| manual | 22 | new |
| report | 7 | new |
| plan | 6 | new |
| faq | 6 | new |
| academic | 7 | -16 |
| news | 2 | new |
| training | 1 | new |
| checklist | 1 | new |
| multimedia | 1 | new |

**Improvements made:**
- Added narrative guide patterns ("I built", "here's how", "total cost")
- Added project indicators (photos, measurements, test results)
- Misc dropped from 80% to 56%

**Remaining issues:**
- Some calculators/tools still getting misc (could add "tool" type?)
- "Academic" dropped too much - builditsolar uses "References" sections but isn't truly academic

---

### Test 2: wiki_top_100 (Wikipedia)

**Source:** `D:\disaster-backups-local\wiki_top_100` (~111 documents)

**Results:**

| Type | Count |
|------|-------|
| academic | 30 |
| field_guide | 25 |
| guide | 20 |
| misc | 20 |
| report | 12 |
| news | 3 |
| multimedia | 1 |

**Key issues identified:**

1. **numbered_steps false positives** - Bob Dylan, Catholic Church, Bat articles getting "guide" classification because Wikipedia's numbered reference lists match step patterns:
   ```
   References
   1. Smith (2020)...
   2. Jones (2021)...
   ```
   This matches `^\s*\d+\.\s+\w` pattern meant for procedural steps.

2. **Multi-type outliers** - African Americans, Alcoholism, Fungus articles hit 0.45 confidence in multiple categories simultaneously. Wikipedia articles blend information types.

3. **field_guide working well** - Biology articles (Bat, Fungus, Bivalvia, Amphibian) correctly detecting species/habitat/characteristics language.

4. **Missing "article" detection** - Encyclopedic content scatters across academic/field_guide/report instead of being recognized as "article" type.

---

### Test 3: LLM vs Rule-based Comparison (builditsolar)

**Sample:** 10 random documents from builditsolar

**Results:**
- Agreements: 2/10 (20%)
- Disagreements: 8/10 (80%)

**Pattern observed:**
- LLM consistently identifies "guide" content that rule-based marks as "misc"
- LLM better understands context of DIY content even without explicit step patterns
- LLM correctly identified a book review as "article" while rules said "guide" (numbered_steps false positive)

**Performance:** ~3-5 seconds per document on CPU (Llama 3.2 3B Q4)

**Conclusion:** LLM is more accurate but slower. Hybrid approach recommended: use rules first, fall back to LLM only for "misc" results.

---

### Test 4: Hybrid Classification (Rules + LLM fallback)

**builditsolar (50 samples):**

| Phase | misc | guide | Other types |
|-------|------|-------|-------------|
| Rule-based only | 27 (54%) | 18 (36%) | 5 (10%) |
| After LLM fallback | 5 (10%) | 35 (70%) | 10 (20%) |

- Rescued 22/27 misc documents (81.5%)
- LLM reclassified as: guide (17), article (3), reference (2)
- 5 documents remained misc (legitimate low-content pages)

**wiki_top_100 (50 samples):**

| Phase | misc | field_guide | academic | Other |
|-------|------|-------------|----------|-------|
| Rule-based only | 8 (16%) | 15 (30%) | 13 (26%) | 14 (28%) |
| After LLM fallback | 0 (0%) | 15 (30%) | 14 (28%) | 21 (42%) |

- Rescued 8/8 misc documents (100%)
- LLM reclassified as: article (4), academic (1), reference (1), policy (1), guide (1)

**Key finding:** Hybrid approach dramatically reduces misc classification while maintaining speed (LLM only runs on ~15-50% of documents).

---

### Learnings

**What's working:**
- Modular detector architecture allows independent tuning
- Narrative patterns improve DIY content detection
- field_guide signals work well for biological/identification content
- Confidence scoring helps identify uncertain classifications
- **Hybrid approach is highly effective** - LLM rescues 80-100% of misc documents
- LLM understands context better than pattern matching alone

**What needs improvement:**
- Numbered step detection needs to distinguish procedural vs reference numbering
- "Article" type needs stronger detection for encyclopedic content
- Source-level hints could improve classification (Wikipedia = likely article)
- Some document types (tool, calculator) may need their own category

**Hybrid approach benefits:**
- Speed: Rules process most documents instantly
- Accuracy: LLM handles edge cases that rules miss
- Cost: LLM only runs on 15-50% of documents (misc results)
- Result: misc drops from 50-80% to 0-10%

---

## Classifier Implementation

### Current Files

| File | Purpose |
|------|---------|
| `offline_tools/document_classifier.py` | Main classifier with 3 classes |
| `offline_tools/test_classifier_report.py` | Batch testing script (rule-based only) |
| `offline_tools/test_llm_classifier.py` | Rule vs LLM side-by-side comparison |
| `offline_tools/test_hybrid_classifier.py` | Hybrid test: rules first, LLM for misc |

### Classes Implemented

**DocumentClassifier** - Rule-based with 5 modular detectors:
- `_detect_guide_signals()` - guide, manual, field_guide
- `_detect_academic_signals()` - academic, article, reference, training
- `_detect_report_signals()` - report, case_study, protocol, checklist
- `_detect_planning_signals()` - plan, policy, assessment
- `_detect_other_signals()` - news, faq, multimedia
- `_detect_misc_signals()` - penalties for low-quality content

**LLMClassifier** - llama-cpp-python based classification (uses local GGUF models)

**HybridClassifier** - Rules first, LLM fallback for uncertain cases

---

## Available Local Models

**Location:** `D:\disaster-backups-local\models\`

| Type | Model | Format | Size |
|------|-------|--------|------|
| LLM | Llama-3.2-3B-Instruct-Q4_K_M | GGUF | 2.0 GB |
| Embeddings | all-mpnet-base-v2 | PyTorch | 438 MB |
| Translation | opus-mt-en-fr / opus-mt-fr-en | PyTorch | 301 MB each |

**Runtime:** Uses llama-cpp-python via `offline_tools/llama_runtime.py` which:
- Auto-detects model from `BACKUP_PATH/models/llm/` via model_registry
- Supports GPU acceleration when configured
- Handles Llama 3.x prompt formatting automatically

---

## Open Questions

1. **Classification job timing** - New job type after metadata, or integrated into metadata generation?

2. **LLM vs rule-based** - Cost/accuracy tradeoff for classification

3. **Frontend approach** - Smart detection vs separate frontends vs mode selector

4. **Manual overrides** - Should admins be able to correct classifications?

5. **Confidence thresholds** - Below what confidence should we flag for review?

6. **Weight tuning** - How to make boost/penalty values easily configurable for experiments?

7. **Numbered step pattern refinement** - How to distinguish procedural steps ("1. Cut the wood") from reference lists ("1. Smith (2020)")?

8. **Source-level hints** - Should known sources (Wikipedia, academic journals) provide classification hints?

9. **Tool/calculator type** - Should we add a dedicated type for interactive tools and calculators?

10. **LLM accuracy testing** - Need to test LLM classifier against rule-based on sample documents

---

## Decisions Made

1. **Per-document classification** - Types stored at document level, not source level (a source contains many document types)

2. **Multi-label with confidence** - Top 3 types with confidence scores, not single label

3. **Deep content analysis** - Not just URL/title heuristics, actual content scanning for reliability

4. **Before indexing** - Classification must complete before index creation so types can influence embeddings

5. **Configurable weights** - Boost/penalty values should be easily adjustable for experimentation

6. **Use llama-cpp-python** - LLM classifier uses existing llama_runtime.py infrastructure, not Ollama

7. **Hybrid classification** - Use rule-based first, LLM fallback only for misc results (balances speed and accuracy)

---

## Next Steps

- [x] Review existing `detect_doc_type_preference()` and `prioritize_results_by_doc_type()` code
- [x] Design classification job structure
- [x] Prototype rule-based classifier (DocumentClassifier)
- [x] Test on sample documents (builditsolar, wiki_top_100)
- [x] Integrate LLM classifier with llama_runtime.py
- [x] Test LLM classifier and compare with rule-based
- [x] Test hybrid approach (rules + LLM fallback for misc)
- [ ] Refine numbered_steps pattern to avoid reference list false positives
- [ ] Improve "article" type detection for encyclopedic content
- [ ] Define weight configuration system
- [ ] Integrate classification into indexing pipeline

---

*Last Updated: December 2025*
