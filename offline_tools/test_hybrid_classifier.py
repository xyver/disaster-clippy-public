"""
Test hybrid classification: Rule-based first, LLM fallback for misc.

Usage:
    python -m offline_tools.test_hybrid_classifier <source_path> [sample_size]
"""

import json
import random
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from offline_tools.document_classifier import DocumentClassifier, LLMClassifier


def extract_text_from_html(html_path: Path) -> str:
    """Extract text content from HTML file."""
    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        return f"ERROR: {e}"


def load_sample_documents(source_path: str, sample_size: int = 50):
    """Load a random sample of documents from a source."""
    source_path = Path(source_path)
    metadata_path = source_path / "_metadata.json"
    pages_path = source_path / "pages"

    if not metadata_path.exists():
        print(f"ERROR: No _metadata.json found at {metadata_path}")
        return []

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    documents = metadata.get('documents', {})
    doc_list = list(documents.items())

    if len(doc_list) > sample_size:
        doc_list = random.sample(doc_list, sample_size)

    loaded = []
    for doc_id, doc_info in doc_list:
        title = doc_info.get('title', '')
        url = doc_info.get('url', '')
        local_url = doc_info.get('local_url', '')

        if local_url:
            filename = local_url.split('/')[-1]
            html_path = pages_path / filename
        else:
            continue

        if not html_path.exists():
            continue

        content = extract_text_from_html(html_path)
        if content.startswith("ERROR:"):
            continue

        loaded.append({
            'doc_id': doc_id,
            'title': title,
            'url': url,
            'content': content
        })

    return loaded


def run_hybrid_test(source_path: str, sample_size: int = 50):
    """
    Run hybrid classification test:
    1. Rule-based classification on all documents
    2. LLM classification only for documents classified as "misc"
    3. Report statistics
    """
    print(f"Loading {sample_size} sample documents from {source_path}...")
    documents = load_sample_documents(source_path, sample_size)
    print(f"Loaded {len(documents)} documents\n")

    rule_classifier = DocumentClassifier()
    llm_classifier = LLMClassifier()

    # Check LLM availability
    llm_available = llm_classifier.is_available()
    print(f"LLM available: {llm_available}")
    if not llm_available:
        print("  -> LLM not available, will only show rule-based results\n")

    print("=" * 80)
    print("PHASE 1: Rule-based Classification")
    print("=" * 80)

    # Track results
    rule_results = {}  # doc_id -> (type, confidence, signals)
    misc_docs = []  # Documents classified as misc

    for i, doc in enumerate(documents):
        results = rule_classifier.classify(doc['content'], doc['title'], doc['url'])
        top_type = results[0].type if results else 'misc'
        top_conf = results[0].confidence if results else 0
        top_signals = results[0].signals if results else []

        rule_results[doc['doc_id']] = (top_type, top_conf, top_signals)

        if top_type == 'misc':
            misc_docs.append(doc)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(documents)}...")

    # Count by type
    rule_counts = {}
    for doc_id, (dtype, conf, signals) in rule_results.items():
        rule_counts[dtype] = rule_counts.get(dtype, 0) + 1

    print(f"\nRule-based results:")
    for dtype, count in sorted(rule_counts.items(), key=lambda x: -x[1]):
        pct = count / len(documents) * 100
        print(f"  {dtype}: {count} ({pct:.1f}%)")

    print(f"\nDocuments classified as misc: {len(misc_docs)}")

    # Phase 2: LLM for misc documents
    if llm_available and misc_docs:
        print("\n" + "=" * 80)
        print(f"PHASE 2: LLM Classification for {len(misc_docs)} misc documents")
        print("=" * 80)

        llm_results = {}  # doc_id -> (type, confidence, signals)
        rescued = 0  # Count of docs reclassified from misc

        for i, doc in enumerate(misc_docs):
            print(f"\n--- LLM {i + 1}/{len(misc_docs)} ---")
            print(f"Title: {doc['title'][:50]}...")

            results = llm_classifier.classify(doc['content'], doc['title'], doc['url'])
            top_type = results[0].type if results else 'misc'
            top_conf = results[0].confidence if results else 0
            top_signals = results[0].signals if results else []

            llm_results[doc['doc_id']] = (top_type, top_conf, top_signals)

            if top_type != 'misc':
                rescued += 1
                print(f"  RESCUED: misc -> {top_type} ({top_conf:.2f})")
            else:
                print(f"  Still misc ({top_conf:.2f})")

        # Count LLM results
        llm_counts = {}
        for doc_id, (dtype, conf, signals) in llm_results.items():
            llm_counts[dtype] = llm_counts.get(dtype, 0) + 1

        print(f"\nLLM results for misc documents:")
        for dtype, count in sorted(llm_counts.items(), key=lambda x: -x[1]):
            print(f"  {dtype}: {count}")

        print(f"\nRescued from misc: {rescued}/{len(misc_docs)} ({rescued/len(misc_docs)*100:.1f}%)")

        # Final combined results
        print("\n" + "=" * 80)
        print("FINAL COMBINED RESULTS (Rules + LLM fallback)")
        print("=" * 80)

        final_counts = {}
        for doc_id, (dtype, conf, signals) in rule_results.items():
            if dtype == 'misc' and doc_id in llm_results:
                # Use LLM result instead
                dtype = llm_results[doc_id][0]
            final_counts[dtype] = final_counts.get(dtype, 0) + 1

        for dtype, count in sorted(final_counts.items(), key=lambda x: -x[1]):
            pct = count / len(documents) * 100
            print(f"  {dtype}: {count} ({pct:.1f}%)")

        # Calculate improvement
        original_misc = rule_counts.get('misc', 0)
        final_misc = final_counts.get('misc', 0)
        improvement = original_misc - final_misc

        print(f"\nImprovement:")
        print(f"  Original misc: {original_misc} ({original_misc/len(documents)*100:.1f}%)")
        print(f"  Final misc: {final_misc} ({final_misc/len(documents)*100:.1f}%)")
        print(f"  Rescued: {improvement} documents")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source_path = sys.argv[1]
        sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        run_hybrid_test(source_path, sample_size)
    else:
        print("Usage: python -m offline_tools.test_hybrid_classifier <source_path> [sample_size]")
        print("\nExample:")
        print("  python -m offline_tools.test_hybrid_classifier D:\\disaster-backups-local\\builditsolar 50")
