"""
Test LLM classifier and compare with rule-based results.

Uses llama-cpp-python with local GGUF models (auto-detected from config).
"""

import json
import random
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from offline_tools.document_classifier import DocumentClassifier, LLMClassifier, HybridClassifier


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


def load_sample_documents(source_path: str, sample_size: int = 20):
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

    # Sample randomly
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


def compare_classifiers(source_path: str, sample_size: int = 10):
    """
    Compare rule-based and LLM classification on sample documents.
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
        print("  -> Make sure llama-cpp-python is installed: pip install llama-cpp-python")
        print("  -> And a GGUF model is configured in BACKUP_PATH/models/llm/")
        print("\nRunning rule-based only...\n")

    print("=" * 80)
    print("COMPARISON: Rule-based vs LLM Classification")
    print("=" * 80)

    agreements = 0
    total = 0

    for i, doc in enumerate(documents):
        print(f"\n--- Document {i+1}/{len(documents)} ---")
        print(f"Title: {doc['title'][:60]}")
        print(f"URL: {doc['url'][:60]}")
        print(f"Content length: {len(doc['content'])} chars")

        # Rule-based classification
        rule_results = rule_classifier.classify(
            doc['content'], doc['title'], doc['url']
        )
        print(f"\nRule-based:")
        for r in rule_results:
            print(f"  {r.type}: {r.confidence:.2f}  {r.signals[:2]}")

        # LLM classification (if available)
        if llm_available:
            print("\nLLM classifying...", end=" ", flush=True)
            llm_results = llm_classifier.classify(
                doc['content'], doc['title'], doc['url']
            )
            print("done")
            print(f"LLM-based:")
            for r in llm_results:
                print(f"  {r.type}: {r.confidence:.2f}  {r.signals[:1]}")

            # Check agreement
            rule_top = rule_results[0].type if rule_results else 'misc'
            llm_top = llm_results[0].type if llm_results else 'misc'

            if rule_top == llm_top:
                print(f"\n[AGREE] Both say: {rule_top}")
                agreements += 1
            else:
                print(f"\n[DISAGREE] Rules: {rule_top}, LLM: {llm_top}")

            total += 1

    # Summary
    if llm_available and total > 0:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total documents: {total}")
        print(f"Agreements: {agreements} ({agreements/total*100:.1f}%)")
        print(f"Disagreements: {total - agreements} ({(total-agreements)/total*100:.1f}%)")


def test_single_document(content: str, title: str = ""):
    """Test classification on a single document."""
    rule_classifier = DocumentClassifier()
    llm_classifier = LLMClassifier()

    print("=" * 60)
    print(f"Title: {title}")
    print(f"Content preview: {content[:200]}...")
    print("=" * 60)

    print("\nRule-based classification:")
    rule_results = rule_classifier.classify(content, title)
    for r in rule_results:
        print(f"  {r.type}: {r.confidence:.2f}")
        print(f"    Signals: {r.signals}")

    if llm_classifier.is_available():
        print("\nLLM classification:")
        llm_results = llm_classifier.classify(content, title)
        for r in llm_results:
            print(f"  {r.type}: {r.confidence:.2f}")
            print(f"    Signals: {r.signals}")
    else:
        print("\nLLM not available (llama-cpp-python not installed or no model configured)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source_path = sys.argv[1]
        sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        compare_classifiers(source_path, sample_size)
    else:
        # Default test
        print("Usage: python test_llm_classifier.py <source_path> [sample_size]")
        print("\nExample:")
        print("  python test_llm_classifier.py D:\\disaster-backups-local\\wiki_top_100 10")
        print("\nRunning quick test with sample content...")

        test_content = """
        How to Build a Rain Barrel

        A rain barrel is a container that collects and stores rainwater from your
        roof for later use in your garden.

        Materials needed:
        - 55 gallon food-grade barrel
        - Spigot
        - Overflow fitting
        - Screen mesh

        Step 1: Clean the barrel
        Make sure your barrel is food-safe and clean it thoroughly.

        Step 2: Install the spigot
        Drill a hole near the bottom and install the spigot with plumber's tape.

        Step 3: Add overflow
        Near the top, install an overflow fitting to redirect excess water.

        Step 4: Add screen
        Cover the top opening with screen mesh to keep out debris and mosquitoes.

        Warning: Never drink rainwater without proper filtration and treatment.
        """

        test_single_document(test_content, "How to Build a Rain Barrel")
