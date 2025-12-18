"""
Test classifier on a real source and generate detailed report.
"""

import json
import random
import sys
from pathlib import Path
from bs4 import BeautifulSoup
from document_classifier import DocumentClassifier, TypeScore


def extract_text_from_html(html_path: Path) -> str:
    """Extract text content from HTML file."""
    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()

        # Get text
        text = soup.get_text(separator='\n', strip=True)
        return text
    except Exception as e:
        return f"ERROR: {e}"


def run_classification_report(source_path: str):
    """Run classifier on all documents and generate report."""

    source_path = Path(source_path)
    metadata_path = source_path / "_metadata.json"
    pages_path = source_path / "pages"

    if not metadata_path.exists():
        print(f"ERROR: No _metadata.json found at {metadata_path}")
        return

    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    documents = metadata.get('documents', {})
    print(f"Found {len(documents)} documents in {source_path.name}")
    print("=" * 60)

    classifier = DocumentClassifier()

    # Results storage
    results = []
    errors = []
    by_type = {'guide': [], 'academic': [], 'misc': []}

    # Process each document
    for i, (doc_id, doc_info) in enumerate(documents.items()):
        if i % 100 == 0:
            print(f"Processing {i}/{len(documents)}...")

        title = doc_info.get('title', '')
        url = doc_info.get('url', '')
        local_url = doc_info.get('local_url', '')

        # Build file path from local_url
        # local_url format: /backup/builditsolar/Comments.htm.html
        # We need: pages/Comments.htm.html
        if local_url:
            filename = local_url.split('/')[-1]
            html_path = pages_path / filename
        else:
            errors.append({
                'doc_id': doc_id,
                'title': title,
                'error': 'No local_url'
            })
            continue

        if not html_path.exists():
            errors.append({
                'doc_id': doc_id,
                'title': title,
                'error': f'File not found: {html_path}'
            })
            continue

        # Extract content
        content = extract_text_from_html(html_path)
        if content.startswith("ERROR:"):
            errors.append({
                'doc_id': doc_id,
                'title': title,
                'error': content
            })
            continue

        # Classify
        try:
            classification = classifier.classify(content, title, url)

            result = {
                'doc_id': doc_id,
                'title': title,
                'url': url,
                'char_count': len(content),
                'types': classification
            }
            results.append(result)

            # Index by primary type
            primary_type = classification[0].type if classification else 'misc'
            primary_conf = classification[0].confidence if classification else 0

            if primary_type in by_type:
                by_type[primary_type].append((result, primary_conf))
            else:
                # Track other types too
                if primary_type not in by_type:
                    by_type[primary_type] = []
                by_type[primary_type].append((result, primary_conf))

        except Exception as e:
            errors.append({
                'doc_id': doc_id,
                'title': title,
                'error': f'Classification error: {e}'
            })

    print(f"\nProcessed {len(results)} documents successfully")
    print(f"Errors: {len(errors)}")
    print("=" * 60)

    # Generate report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT: " + source_path.name)
    print("=" * 60)

    # Summary stats
    print("\n--- SUMMARY ---")
    for doc_type, items in sorted(by_type.items()):
        print(f"  {doc_type}: {len(items)} documents")

    # Top 3 highest confidence for each tracked type
    for doc_type in ['guide', 'academic', 'misc']:
        print(f"\n--- TOP 3 HIGHEST CONFIDENCE: {doc_type.upper()} ---")
        if doc_type in by_type and by_type[doc_type]:
            sorted_items = sorted(by_type[doc_type], key=lambda x: x[1], reverse=True)
            for result, conf in sorted_items[:3]:
                print(f"\n  [{conf:.2f}] {result['title'][:60]}")
                print(f"         URL: {result['url'][:70]}")
                print(f"         Signals: {[t.signals for t in result['types'][:1]]}")
        else:
            print("  (none found)")

    # Bottom 3 lowest confidence for each tracked type
    for doc_type in ['guide', 'academic', 'misc']:
        print(f"\n--- BOTTOM 3 LOWEST CONFIDENCE: {doc_type.upper()} ---")
        if doc_type in by_type and by_type[doc_type]:
            sorted_items = sorted(by_type[doc_type], key=lambda x: x[1])
            for result, conf in sorted_items[:3]:
                print(f"\n  [{conf:.2f}] {result['title'][:60]}")
                print(f"         URL: {result['url'][:70]}")
                print(f"         Signals: {[t.signals for t in result['types'][:1]]}")
        else:
            print("  (none found)")

    # 10 random articles with all confidence levels
    print("\n--- 10 RANDOM ARTICLES ---")
    if len(results) >= 10:
        random_sample = random.sample(results, 10)
    else:
        random_sample = results

    for result in random_sample:
        print(f"\n  Title: {result['title'][:60]}")
        print(f"  URL: {result['url'][:70]}")
        print(f"  Chars: {result['char_count']}")
        print(f"  Classifications:")
        for t in result['types']:
            print(f"    - {t.type}: {t.confidence:.2f} {t.signals}")

    # Outliers: high confidence in multiple categories
    print("\n--- OUTLIERS: HIGH CONFIDENCE IN MULTIPLE CATEGORIES ---")
    outliers = []
    for result in results:
        types = result['types']
        if len(types) >= 2:
            # Check if top 2 types both have confidence > 0.4
            if types[0].confidence > 0.4 and types[1].confidence > 0.4:
                outliers.append(result)

    if outliers:
        for result in outliers[:10]:
            print(f"\n  Title: {result['title'][:60]}")
            print(f"  URL: {result['url'][:70]}")
            for t in result['types']:
                print(f"    - {t.type}: {t.confidence:.2f}")
    else:
        print("  (none found - no documents with multiple high-confidence types)")

    # Error report
    if errors:
        print(f"\n--- ERRORS ({len(errors)}) ---")
        for err in errors[:10]:
            print(f"\n  Title: {err['title'][:50]}")
            print(f"  Error: {err['error']}")
        if len(errors) > 10:
            print(f"\n  ... and {len(errors) - 10} more errors")

    # Documents that couldn't be categorized (misc with low confidence)
    print("\n--- UNCATEGORIZABLE (misc with signals suggesting otherwise) ---")
    uncertain = []
    for result in results:
        types = result['types']
        if types and types[0].type == 'misc':
            # Check if any other type got a score
            if len(types) > 1 or types[0].confidence < 0.7:
                uncertain.append(result)

    if uncertain:
        for result in uncertain[:5]:
            print(f"\n  Title: {result['title'][:60]}")
            print(f"  URL: {result['url'][:70]}")
            for t in result['types']:
                print(f"    - {t.type}: {t.confidence:.2f} {t.signals}")
    else:
        print("  (none found)")

    print("\n" + "=" * 60)
    print("END OF REPORT")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        source_path = sys.argv[1]
    else:
        source_path = r"D:\disaster-backups-local\builditsolar"

    run_classification_report(source_path)
