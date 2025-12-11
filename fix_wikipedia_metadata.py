#!/usr/bin/env python3
"""
Fix wikipedia-medical metadata URLs.

Converts:
  url: "/zim/wikipedia-medical/Article_Name"
  local_url: ""

To:
  url: "https://en.wikipedia.org/wiki/Article_Name"
  local_url: "/zim/wikipedia-medical/Article_Name"
"""

import json
from pathlib import Path

def fix_wikipedia_metadata(backup_folder: str):
    """Fix Wikipedia metadata file by swapping URLs."""

    metadata_path = Path(backup_folder) / "wikipedia-medical" / "_metadata.json"

    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        return False

    print(f"Loading metadata from {metadata_path}...")

    # Load the metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = data.get("documents", {})
    print(f"Found {len(documents)} documents")

    # Fix each document
    fixed_count = 0
    for doc_id, doc_data in documents.items():
        url = doc_data.get("url", "")
        local_url = doc_data.get("local_url", "")

        # If url starts with /zim/ and local_url is empty, fix it
        if url.startswith("/zim/wikipedia-medical/"):
            # Extract article name: /zim/wikipedia-medical/Article_Name -> Article_Name
            article_name = url.replace("/zim/wikipedia-medical/", "")

            # Build online Wikipedia URL
            online_url = f"https://en.wikipedia.org/wiki/{article_name}"

            # Swap: old url becomes local_url, new online_url becomes url
            doc_data["local_url"] = url  # Store old /zim/ path as local_url
            doc_data["url"] = online_url  # Store Wikipedia URL as url

            fixed_count += 1

    print(f"Fixed {fixed_count} document URLs")

    # Create backup of original file
    backup_path = metadata_path.with_suffix('.json.backup')
    print(f"Creating backup at {backup_path}...")
    with open(backup_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    # Write fixed metadata
    print(f"Writing fixed metadata to {metadata_path}...")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(data, f)

    print("Done!")
    print(f"\nSummary:")
    print(f"  - Total documents: {len(documents)}")
    print(f"  - Fixed URLs: {fixed_count}")
    print(f"  - Backup saved: {backup_path}")

    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fix_wikipedia_metadata.py <BACKUP_FOLDER>")
        print("\nExample:")
        print("  python fix_wikipedia_metadata.py D:\\disaster-backups")
        sys.exit(1)

    backup_folder = sys.argv[1]
    fix_wikipedia_metadata(backup_folder)
