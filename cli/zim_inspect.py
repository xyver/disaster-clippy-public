"""
ZIM File Inspector CLI
Diagnostic tool to analyze ZIM file contents before indexing

This is a CLI wrapper around offline_tools.zim_utils.inspect_zim_file()
"""

import sys
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from offline_tools.zim_utils import inspect_zim_file, ZIMInspectionResult


def print_inspection_report(result: ZIMInspectionResult, verbose: bool = False) -> None:
    """Print a formatted inspection report to console"""

    print(f"\n{'='*60}")
    print(f"ZIM INSPECTION: {Path(result.file_path).name}")
    print(f"{'='*60}\n")

    if result.error:
        print(f"ERROR: {result.error}")
        return

    # 1. Header metadata
    print("HEADER METADATA:")
    print("-" * 40)
    print(f"  File size: {result.file_size_mb} MB")
    print(f"  Total articles: {result.article_count}")
    if result.title:
        print(f"  Title: {result.title}")
    if result.description:
        desc = result.description[:100] + "..." if len(result.description) > 100 else result.description
        print(f"  Description: {desc}")
    if result.creator:
        print(f"  Creator: {result.creator}")
    if result.publisher:
        print(f"  Publisher: {result.publisher}")
    if result.language:
        print(f"  Language: {result.language}")
    if result.license:
        print(f"  License: {result.license}")
    if result.source_url:
        print(f"  Source URL: {result.source_url}")
    if result.date:
        print(f"  Date: {result.date}")
    if result.tags:
        print(f"  Tags: {', '.join(result.tags)}")

    # 2. Content type analysis
    print(f"\n\nCONTENT TYPE: {result.content_type.upper()}")
    print("-" * 40)

    if result.mimetype_distribution:
        print("\nMimetype distribution:")
        for mime, count in sorted(result.mimetype_distribution.items(), key=lambda x: -x[1])[:10]:
            print(f"  {mime}: {count}")

    if result.namespace_distribution:
        print("\nNamespace distribution:")
        for ns, count in sorted(result.namespace_distribution.items(), key=lambda x: -x[1])[:5]:
            print(f"  {ns}: {count}")

    # 3. Text length analysis
    if result.html_article_count > 0:
        print("\n\nTEXT LENGTH ANALYSIS:")
        print("-" * 40)
        skip_rate = result.skipped_count / result.html_article_count * 100

        print(f"  HTML articles scanned: {result.html_article_count}")
        print(f"  Indexable (>={result.min_text_length_used} chars): {result.indexable_count}")
        print(f"  Would be skipped (<{result.min_text_length_used} chars): {result.skipped_count} ({skip_rate:.1f}%)")
        print(f"  Average text length: {result.avg_text_length:.0f} chars")

        if result.has_video_content:
            print(f"\n  Video-related articles: {result.video_article_count}")

    # 4. Sample articles
    if result.sample_indexable:
        print("\n\nSAMPLE INDEXABLE ARTICLES:")
        print("-" * 40)
        for a in result.sample_indexable[:5]:
            print(f"\n  [OK] {a['title'][:60]}")
            print(f"       URL: {a['url'][:60]}")
            print(f"       Text length: {a['text_length']} chars")
            if verbose and a.get('preview'):
                print(f"       Preview: {a['preview'][:80]}...")

    if result.sample_skipped:
        print("\n\nSAMPLE ARTICLES THAT WOULD BE SKIPPED:")
        print("-" * 40)
        for a in result.sample_skipped[:5]:
            print(f"\n  [SKIP] {a['title'][:60]}")
            print(f"         URL: {a['url'][:60]}")
            print(f"         Text length: {a['text_length']} chars")

    if result.sample_videos:
        print("\n\nSAMPLE VIDEO ARTICLES:")
        print("-" * 40)
        for a in result.sample_videos[:5]:
            status = "[OK]" if a['text_length'] >= result.min_text_length_used else "[SKIP]"
            print(f"  {status} {a['title'][:50]} ({a['text_length']} chars)")

    # 5. Recommendations
    if result.recommendations:
        print("\n\nRECOMMENDATIONS:")
        print("-" * 40)
        for rec in result.recommendations:
            print(f"  {rec}")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect ZIM file contents before indexing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python zim_inspect.py /path/to/file.zim
  python zim_inspect.py /path/to/file.zim --scan-limit 10000
  python zim_inspect.py /path/to/file.zim --min-text 30 -v
        """
    )
    parser.add_argument("zim_path", help="Path to ZIM file")
    parser.add_argument(
        "-s", "--scan-limit",
        type=int,
        default=5000,
        help="Maximum articles to scan (default: 5000)"
    )
    parser.add_argument(
        "-m", "--min-text",
        type=int,
        default=50,
        help="Minimum text length to consider indexable (default: 50)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show more details including content previews"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of formatted text"
    )

    args = parser.parse_args()

    # Progress callback for CLI
    def progress(current, total, message):
        if current % 500 == 0 or current == total:
            print(f"\r  {message} ({current}/{total})", end="", flush=True)
        if current == total:
            print()

    # Run inspection
    result = inspect_zim_file(
        zim_path=args.zim_path,
        scan_limit=args.scan_limit,
        min_text_length=args.min_text,
        progress_callback=progress if not args.json else None
    )

    if args.json:
        import json
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print_inspection_report(result, verbose=args.verbose)

    # Return error code if inspection failed
    if result.error:
        sys.exit(1)


if __name__ == "__main__":
    main()
