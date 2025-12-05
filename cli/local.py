#!/usr/bin/env python3
"""
Local CLI for Disaster Clippy

This is the PUBLIC CLI wrapper for local admins. It provides access to:
- Generating metadata from local backups
- Indexing to local ChromaDB
- Exporting local indexes
- Submitting packs for review

Usage:
    python -m cli.local metadata --path ./backups/mysite
    python -m cli.local index-html --path ./backups/mysite --source-id mysite
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def cmd_generate_metadata(args):
    """Generate metadata from a local backup folder."""
    from offline_tools.packager import generate_metadata_from_html, save_metadata

    print(f"Generating metadata from: {args.path}")

    if not os.path.exists(args.path):
        print(f"Error: Path does not exist: {args.path}")
        return 1

    metadata = generate_metadata_from_html(args.path)

    if args.output:
        save_metadata(metadata, args.output)
        print(f"Saved metadata to: {args.output}")
    else:
        import json
        print(json.dumps(metadata, indent=2))

    return 0


def cmd_index_html(args):
    """Index HTML backup to local ChromaDB."""
    from offline_tools.packager import index_html_to_chromadb

    print(f"Indexing HTML backup: {args.path}")
    print(f"Source ID: {args.source_id}")

    if not os.path.exists(args.path):
        print(f"Error: Path does not exist: {args.path}")
        return 1

    result = index_html_to_chromadb(args.path, args.source_id)
    print(f"Result: {result}")
    return 0


def cmd_index_zim(args):
    """Index ZIM file to local ChromaDB."""
    from offline_tools.packager import index_zim_to_chromadb

    print(f"Indexing ZIM file: {args.path}")
    print(f"Source ID: {args.source_id}")

    if not os.path.exists(args.path):
        print(f"Error: Path does not exist: {args.path}")
        return 1

    result = index_zim_to_chromadb(args.path, args.source_id)
    print(f"Result: {result}")
    return 0


def cmd_export_index(args):
    """Export local ChromaDB index to file."""
    from offline_tools.packager import export_chromadb_index

    print(f"Exporting index for source: {args.source_id}")

    result = export_chromadb_index(args.source_id, args.output)
    print(f"Exported to: {result}")
    return 0


def cmd_check_completeness(args):
    """Check if a source pack is complete and ready for submission."""
    from offline_tools.packager import get_source_completeness

    result = get_source_completeness(args.source_id)

    print(f"\nSource: {args.source_id}")
    print("-" * 40)
    for key, value in result.items():
        status = "[OK]" if value else "[MISSING]"
        print(f"  {status} {key}")

    if all(result.values()):
        print("\nPack is complete and ready for submission!")
    else:
        print("\nPack is incomplete. Address missing items before submitting.")

    return 0


def cmd_create_manifest(args):
    """Create a pack manifest for submission."""
    from offline_tools.packager import create_pack_manifest

    print(f"Creating manifest for: {args.source_id}")

    manifest = create_pack_manifest(
        source_id=args.source_id,
        backup_path=args.backup_path,
        metadata_path=args.metadata_path
    )

    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest to: {args.output}")
    else:
        import json
        print(json.dumps(manifest, indent=2))

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Disaster Clippy Local CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate metadata from HTML backup
  python -m cli.local metadata --path ./backups/mysite --output metadata.json

  # Index HTML backup to ChromaDB
  python -m cli.local index-html --path ./backups/mysite --source-id mysite

  # Index ZIM file to ChromaDB
  python -m cli.local index-zim --path ./backups/wikipedia.zim --source-id wikipedia

  # Check if pack is ready for submission
  python -m cli.local check --source-id mysite

  # Create manifest for submission
  python -m cli.local manifest --source-id mysite --backup-path ./backups/mysite
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # metadata command
    p_meta = subparsers.add_parser('metadata', help='Generate metadata from backup')
    p_meta.add_argument('--path', required=True, help='Path to backup folder')
    p_meta.add_argument('--output', help='Output file (default: print to stdout)')
    p_meta.set_defaults(func=cmd_generate_metadata)

    # index-html command
    p_html = subparsers.add_parser('index-html', help='Index HTML backup to ChromaDB')
    p_html.add_argument('--path', required=True, help='Path to HTML backup folder')
    p_html.add_argument('--source-id', required=True, help='Source identifier')
    p_html.set_defaults(func=cmd_index_html)

    # index-zim command
    p_zim = subparsers.add_parser('index-zim', help='Index ZIM file to ChromaDB')
    p_zim.add_argument('--path', required=True, help='Path to ZIM file')
    p_zim.add_argument('--source-id', required=True, help='Source identifier')
    p_zim.set_defaults(func=cmd_index_zim)

    # export command
    p_export = subparsers.add_parser('export', help='Export ChromaDB index')
    p_export.add_argument('--source-id', required=True, help='Source identifier')
    p_export.add_argument('--output', required=True, help='Output file path')
    p_export.set_defaults(func=cmd_export_index)

    # check command
    p_check = subparsers.add_parser('check', help='Check pack completeness')
    p_check.add_argument('--source-id', required=True, help='Source identifier')
    p_check.set_defaults(func=cmd_check_completeness)

    # manifest command
    p_manifest = subparsers.add_parser('manifest', help='Create pack manifest')
    p_manifest.add_argument('--source-id', required=True, help='Source identifier')
    p_manifest.add_argument('--backup-path', required=True, help='Path to backup')
    p_manifest.add_argument('--metadata-path', help='Path to metadata.json')
    p_manifest.add_argument('--output', help='Output file (default: print to stdout)')
    p_manifest.set_defaults(func=cmd_create_manifest)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
