#!/usr/bin/env python3
"""
Database sync CLI for Disaster Clippy.

Sync vector databases using metadata comparison for fast, efficient updates.

Usage:
    python -m cli.sync compare                      # Compare local vs remote
    python -m cli.sync compare --verbose            # Show detailed diff
    python -m cli.sync push --dry-run               # See what would be pushed
    python -m cli.sync push --commit                # Actually push changes
    python -m cli.sync pull --commit                # Pull remote changes to local

Examples:
    # Test sync between two local databases
    python -m cli.sync compare --local data/chroma --remote data/chroma_backup

    # Sync to production (when Pinecone configured)
    python -m cli.sync compare --remote pinecone
    python -m cli.sync push --commit --remote pinecone
"""

import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

from offline_tools.vectordb import VectorStore, PineconeStore, get_vector_store as factory_get_store
from offline_tools.vectordb.sync import SyncManager

# Load environment
load_dotenv()


def get_vector_store(mode: str, **kwargs):
    """
    Factory function to get vector store based on mode.

    Args:
        mode: 'local', 'pinecone', 'qdrant', or path to directory
        **kwargs: Additional arguments for store initialization

    Returns:
        VectorStore or PineconeStore instance
    """
    # If mode is a path, treat as local ChromaDB
    if os.path.exists(mode) or mode.startswith("data/") or mode.startswith("/"):
        return VectorStore(persist_dir=mode)

    # Otherwise use factory
    if mode in ["local", "railway"]:
        return factory_get_store(mode=mode, **kwargs)

    elif mode == "pinecone":
        print("Connecting to Pinecone...")
        return PineconeStore(**kwargs)

    elif mode == "qdrant":
        raise NotImplementedError("Qdrant sync not yet implemented.")

    else:
        raise ValueError(f"Unknown vector store mode: {mode}")


def cmd_compare(args):
    """Compare local and remote databases"""
    print("Loading databases...")

    try:
        local = get_vector_store(args.local)
        remote = get_vector_store(args.remote) if args.remote else None

        if not remote:
            print("Error: No remote database specified.")
            print("Use --remote <path> or --remote pinecone")
            return 1

        sync = SyncManager(local, remote)
        diff = sync.compare(source=args.source)
        sync.print_diff(diff, verbose=args.verbose)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_push(args):
    """Push local changes to remote"""
    print("Loading databases...")

    try:
        local = get_vector_store(args.local)
        remote = get_vector_store(args.remote) if args.remote else None

        if not remote:
            print("Error: No remote database specified.")
            return 1

        sync = SyncManager(local, remote)

        # Compare first
        print("Comparing databases...")
        diff = sync.compare(source=args.source)
        sync.print_diff(diff, verbose=False)

        # Confirm if not dry run
        if not args.dry_run and not args.yes:
            print("\n[WARNING] This will modify the remote database.")
            response = input("Continue? (yes/no): ")
            if response.lower() != "yes":
                print("Aborted.")
                return 0

        # Push
        stats = sync.push(
            diff,
            dry_run=args.dry_run,
            update=not args.no_update,
            force=args.force
        )

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_pull(args):
    """Pull remote changes to local"""
    print("Loading databases...")

    try:
        local = get_vector_store(args.local)
        remote = get_vector_store(args.remote) if args.remote else None

        if not remote:
            print("Error: No remote database specified.")
            return 1

        sync = SyncManager(local, remote)

        # Compare first
        print("Comparing databases...")
        diff = sync.compare(source=args.source)
        sync.print_diff(diff, verbose=False)

        # Confirm if not dry run
        if not args.dry_run and not args.yes:
            print("\n[WARNING] This will modify the local database.")
            response = input("Continue? (yes/no): ")
            if response.lower() != "yes":
                print("Aborted.")
                return 0

        # Pull
        stats = sync.pull(diff, dry_run=args.dry_run)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_export_diff(args):
    """Export diff report as JSON"""
    try:
        local = get_vector_store(args.local)
        remote = get_vector_store(args.remote) if args.remote else None

        if not remote:
            print("Error: No remote database specified.")
            return 1

        sync = SyncManager(local, remote)
        diff = sync.compare(source=args.source)
        sync.export_diff_report(diff, args.output)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Sync vector databases for Disaster Clippy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare local database with backup
  python -m cli.sync compare --local data/chroma --remote data/chroma_backup

  # Compare with verbose output
  python -m cli.sync compare --remote data/chroma_backup --verbose

  # Dry run push (see what would happen)
  python -m cli.sync push --remote data/chroma_backup --dry-run

  # Actually push changes
  python -m cli.sync push --remote data/chroma_backup --commit

  # Pull remote changes to local
  python -m cli.sync pull --remote data/chroma_backup --commit

  # Export diff report
  python -m cli.sync export-diff --remote data/chroma_backup --output diff.json

  # Future: Sync to Pinecone
  # python -m cli.sync compare --remote pinecone
  # python -m cli.sync push --commit --remote pinecone
        """
    )

    # Global arguments
    from offline_tools.vectordb.store import get_default_chroma_path
    default_chroma = get_default_chroma_path()
    parser.add_argument(
        "--local",
        default=default_chroma,
        help=f"Local database path (default: {default_chroma})"
    )

    parser.add_argument(
        "--remote",
        help="Remote database path or mode (path, pinecone, qdrant)"
    )

    parser.add_argument(
        "--source",
        help="Filter to specific source (e.g., appropedia)"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Compare command
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare local and remote databases"
    )
    compare_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed diff with document titles"
    )

    # Push command
    push_parser = subparsers.add_parser(
        "push",
        help="Push local changes to remote"
    )
    push_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be pushed without actually doing it"
    )
    push_parser.add_argument(
        "--commit",
        dest="dry_run",
        action="store_false",
        help="Actually push changes (opposite of --dry-run)"
    )
    push_parser.add_argument(
        "--no-update",
        action="store_true",
        help="Only push new documents, don't update existing ones"
    )
    push_parser.add_argument(
        "--force",
        action="store_true",
        help="Force push even if there are conflicts"
    )
    push_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    push_parser.set_defaults(dry_run=True)  # Default to dry run for safety

    # Pull command
    pull_parser = subparsers.add_parser(
        "pull",
        help="Pull remote changes to local"
    )
    pull_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be pulled without actually doing it"
    )
    pull_parser.add_argument(
        "--commit",
        dest="dry_run",
        action="store_false",
        help="Actually pull changes"
    )
    pull_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt"
    )
    pull_parser.set_defaults(dry_run=True)

    # Export diff command
    export_parser = subparsers.add_parser(
        "export-diff",
        help="Export diff report as JSON"
    )
    export_parser.add_argument(
        "output",
        help="Output JSON file path"
    )

    args = parser.parse_args()

    # Dispatch to command
    if args.command == "compare":
        return cmd_compare(args)
    elif args.command == "push":
        return cmd_push(args)
    elif args.command == "pull":
        return cmd_pull(args)
    elif args.command == "export-diff":
        return cmd_export_diff(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
