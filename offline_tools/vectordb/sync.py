"""
Sync manager for comparing and syncing vector databases.
Uses metadata index for fast comparison without downloading embeddings.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path
import json
from datetime import datetime

from .metadata import MetadataIndex
from .store import VectorStore


class SyncDiff:
    """Represents differences between two databases"""

    def __init__(self):
        self.to_push: Set[str] = set()      # New docs in local, not in remote
        self.to_pull: Set[str] = set()      # New docs in remote, not in local
        self.to_update: Set[str] = set()    # Docs that changed (different content_hash)
        self.in_sync: Set[str] = set()      # Docs that match perfectly
        self.conflicts: List[Dict] = []     # Docs changed on both sides

    def __repr__(self):
        return (
            f"SyncDiff(to_push={len(self.to_push)}, "
            f"to_pull={len(self.to_pull)}, "
            f"to_update={len(self.to_update)}, "
            f"in_sync={len(self.in_sync)}, "
            f"conflicts={len(self.conflicts)})"
        )


class SyncManager:
    """
    Compare and sync between vector stores using metadata indices.

    Supports:
    - Local ChromaDB to Local ChromaDB (testing)
    - Local ChromaDB to Pinecone (production sync)
    - Metadata-based comparison (fast, no embedding download)
    - Conflict detection
    - Dry-run mode
    """

    def __init__(self, local_store: VectorStore, remote_store: Optional[VectorStore] = None):
        """
        Args:
            local_store: Local vector store (usually ChromaDB)
            remote_store: Remote vector store (ChromaDB, Pinecone, etc.)
        """
        self.local_store = local_store
        self.remote_store = remote_store
        self.local_metadata = local_store.metadata_index
        self.remote_metadata = remote_store.metadata_index if remote_store else None

    def compare(self, source: Optional[str] = None) -> SyncDiff:
        """
        Compare local and remote databases using metadata.

        Args:
            source: Optional filter to specific source (e.g., "appropedia")

        Returns:
            SyncDiff object with differences
        """
        if not self.remote_metadata:
            raise ValueError("No remote store configured for comparison")

        diff = SyncDiff()

        # Get document IDs from both sides
        local_ids = self.local_metadata.get_ids(source)
        remote_ids = self.remote_metadata.get_ids(source)

        # Find new documents
        diff.to_push = local_ids - remote_ids
        diff.to_pull = remote_ids - local_ids

        # Check documents that exist in both
        in_both = local_ids & remote_ids

        for doc_id in in_both:
            # Get metadata from both sides
            local_doc = self._get_doc_metadata(self.local_metadata, doc_id, source)
            remote_doc = self._get_doc_metadata(self.remote_metadata, doc_id, source)

            if not local_doc or not remote_doc:
                continue

            # Compare content hashes
            local_hash = local_doc.get("content_hash", "")
            remote_hash = remote_doc.get("content_hash", "")

            if local_hash == remote_hash:
                # Perfect match
                diff.in_sync.add(doc_id)
            else:
                # Content changed
                local_scraped = local_doc.get("scraped_at", "")
                remote_scraped = remote_doc.get("scraped_at", "")

                # Check if both sides changed (conflict)
                if local_scraped > remote_scraped:
                    # Local is newer
                    diff.to_update.add(doc_id)
                elif remote_scraped > local_scraped:
                    # Remote is newer - potential conflict
                    diff.conflicts.append({
                        "doc_id": doc_id,
                        "title": local_doc.get("title", "Unknown"),
                        "local_scraped": local_scraped,
                        "remote_scraped": remote_scraped,
                        "reason": "Remote version is newer than local"
                    })
                else:
                    # Same timestamp, different content - real conflict
                    diff.to_update.add(doc_id)

        return diff

    def _get_doc_metadata(self, metadata_index: MetadataIndex,
                         doc_id: str, source: Optional[str] = None) -> Optional[Dict]:
        """Get document metadata from an index"""
        # If source specified, look there
        if source:
            source_data = metadata_index._load_source(source)
            return source_data["documents"].get(doc_id)

        # Otherwise search all sources
        for src in metadata_index.list_sources():
            source_data = metadata_index._load_source(src)
            if doc_id in source_data["documents"]:
                return source_data["documents"][doc_id]

        return None

    def print_diff(self, diff: SyncDiff, verbose: bool = False):
        """Print a human-readable diff summary"""
        print("\n" + "=" * 60)
        print("SYNC COMPARISON")
        print("=" * 60)

        total_local = len(self.local_metadata.get_ids())
        total_remote = len(self.remote_metadata.get_ids()) if self.remote_metadata else 0

        print(f"\nDatabase Status:")
        print(f"  Local:  {total_local} documents")
        print(f"  Remote: {total_remote} documents")

        print(f"\nSync Status:")
        print(f"  [OK] In sync:    {len(diff.in_sync)} documents")
        print(f"  [->] To push:    {len(diff.to_push)} new documents")
        print(f"  [UP] To update:  {len(diff.to_update)} changed documents")
        print(f"  [<-] To pull:    {len(diff.to_pull)} documents (remote only)")

        if diff.conflicts:
            print(f"  [!!] Conflicts:  {len(diff.conflicts)} documents")

        # Verbose output
        if verbose:
            if diff.to_push:
                print(f"\nDocuments to push ({len(diff.to_push)}):")
                for doc_id in list(diff.to_push)[:10]:
                    doc = self._get_doc_metadata(self.local_metadata, doc_id)
                    if doc:
                        print(f"  + {doc.get('title', 'Unknown')[:60]}")
                if len(diff.to_push) > 10:
                    print(f"  ... and {len(diff.to_push) - 10} more")

            if diff.to_update:
                print(f"\nDocuments to update ({len(diff.to_update)}):")
                for doc_id in list(diff.to_update)[:10]:
                    doc = self._get_doc_metadata(self.local_metadata, doc_id)
                    if doc:
                        print(f"  ^ {doc.get('title', 'Unknown')[:60]}")
                if len(diff.to_update) > 10:
                    print(f"  ... and {len(diff.to_update) - 10} more")

            if diff.conflicts:
                print(f"\nConflicts ({len(diff.conflicts)}):")
                for conflict in diff.conflicts[:5]:
                    print(f"  !! {conflict['title'][:60]}")
                    print(f"     Local:  {conflict['local_scraped']}")
                    print(f"     Remote: {conflict['remote_scraped']}")
                if len(diff.conflicts) > 5:
                    print(f"  ... and {len(diff.conflicts) - 5} more")

        print("\n" + "=" * 60)

    def push(self, diff: SyncDiff, dry_run: bool = True,
             update: bool = True, force: bool = False) -> Dict[str, int]:
        """
        Push local changes to remote.

        Args:
            diff: SyncDiff from compare()
            dry_run: If True, only show what would be done
            update: If True, also push updates (not just new docs)
            force: If True, push even if there are conflicts

        Returns:
            Stats dict with counts of pushed/updated documents
        """
        if not self.remote_store:
            raise ValueError("No remote store configured")

        stats = {
            "pushed": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0
        }

        # Check for conflicts
        if diff.conflicts and not force:
            print(f"\n[!!] Warning: {len(diff.conflicts)} conflicts detected!")
            print("Remote has newer versions of some documents.")
            print("Use --force to override, or resolve conflicts manually.")
            return stats

        if dry_run:
            print("\n" + "=" * 60)
            print("DRY RUN MODE - No changes will be made")
            print("=" * 60)

        # Push new documents
        if diff.to_push:
            print(f"\nPushing {len(diff.to_push)} new documents...")

            docs_to_push = []
            for doc_id in diff.to_push:
                # Get full document from local store
                doc = self._get_full_document(self.local_store, doc_id)
                if doc:
                    docs_to_push.append(doc)

            if not dry_run and docs_to_push:
                try:
                    count = self.remote_store.add_documents(docs_to_push)
                    stats["pushed"] = count
                    print(f"  [OK] Pushed {count} documents")
                except Exception as e:
                    print(f"  [ERR] Error pushing documents: {e}")
                    stats["errors"] += len(docs_to_push)
            else:
                stats["pushed"] = len(docs_to_push)
                print(f"  Would push {len(docs_to_push)} documents")

        # Update changed documents
        if update and diff.to_update:
            print(f"\nUpdating {len(diff.to_update)} changed documents...")

            docs_to_update = []
            for doc_id in diff.to_update:
                doc = self._get_full_document(self.local_store, doc_id)
                if doc:
                    docs_to_update.append(doc)

            if not dry_run and docs_to_update:
                try:
                    # Remove old versions
                    for doc in docs_to_update:
                        doc_id = doc.get("id") or doc.get("content_hash")
                        # Note: This assumes remote store has a delete method
                        # Will need to implement for each store type

                    # Add new versions (Pinecone upsert handles updates)
                    count = self.remote_store.add_documents(docs_to_update)
                    stats["updated"] = count
                    print(f"  [OK] Updated {count} documents")
                except Exception as e:
                    print(f"  [ERR] Error updating documents: {e}")
                    stats["errors"] += len(docs_to_update)
            else:
                stats["updated"] = len(docs_to_update)
                print(f"  Would update {len(docs_to_update)} documents")

        if dry_run:
            print("\n" + "=" * 60)
            print("DRY RUN COMPLETE - Run with --commit to apply changes")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("SYNC COMPLETE")
            print("=" * 60)
            print(f"  Pushed:  {stats['pushed']} documents")
            print(f"  Updated: {stats['updated']} documents")
            if stats['errors']:
                print(f"  Errors:  {stats['errors']} documents")

        return stats

    def _get_full_document(self, store: VectorStore, doc_id: str) -> Optional[Dict]:
        """
        Get full document with embeddings from a vector store.

        Note: This requires querying the actual vector DB, not just metadata.
        """
        try:
            # Get from ChromaDB
            result = store.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"]
            )

            if not result["ids"]:
                return None

            # Reconstruct document
            metadata = result["metadatas"][0]

            # Parse categories back from JSON
            if "categories" in metadata:
                try:
                    metadata["categories"] = json.loads(metadata["categories"])
                except:
                    metadata["categories"] = []

            doc = {
                "id": doc_id,
                "content": result["documents"][0],
                "embedding": result["embeddings"][0],
                **metadata
            }

            return doc

        except Exception as e:
            print(f"Error getting document {doc_id}: {e}")
            return None

    def pull(self, diff: SyncDiff, dry_run: bool = True) -> Dict[str, int]:
        """
        Pull remote changes to local.
        Useful for setting up a new dev environment.

        Args:
            diff: SyncDiff from compare()
            dry_run: If True, only show what would be done

        Returns:
            Stats dict with counts
        """
        if not self.remote_store:
            raise ValueError("No remote store configured")

        stats = {"pulled": 0, "errors": 0}

        if dry_run:
            print("\n" + "=" * 60)
            print("DRY RUN MODE - No changes will be made")
            print("=" * 60)

        if diff.to_pull:
            print(f"\nPulling {len(diff.to_pull)} documents from remote...")

            docs_to_pull = []
            for doc_id in diff.to_pull:
                doc = self._get_full_document(self.remote_store, doc_id)
                if doc:
                    docs_to_pull.append(doc)

            if not dry_run and docs_to_pull:
                try:
                    count = self.local_store.add_documents(docs_to_pull)
                    stats["pulled"] = count
                    print(f"  [OK] Pulled {count} documents")
                except Exception as e:
                    print(f"  [ERR] Error pulling documents: {e}")
                    stats["errors"] += len(docs_to_pull)
            else:
                stats["pulled"] = len(docs_to_pull)
                print(f"  Would pull {len(docs_to_pull)} documents")

        if dry_run:
            print("\n" + "=" * 60)
            print("DRY RUN COMPLETE - Run with --commit to apply changes")
            print("=" * 60)

        return stats

    def export_diff_report(self, diff: SyncDiff, output_file: str):
        """Export diff as JSON report"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "local_total": len(self.local_metadata.get_ids()),
            "remote_total": len(self.remote_metadata.get_ids()) if self.remote_metadata else 0,
            "to_push": list(diff.to_push),
            "to_pull": list(diff.to_pull),
            "to_update": list(diff.to_update),
            "in_sync": len(diff.in_sync),
            "conflicts": diff.conflicts
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nDiff report saved to: {output_file}")


# Quick test
if __name__ == "__main__":
    # Example: Compare two local ChromaDB instances
    local = VectorStore(persist_dir="data/chroma")
    # remote = VectorStore(persist_dir="data/chroma_backup")

    # sync = SyncManager(local, remote)
    # diff = sync.compare()
    # sync.print_diff(diff, verbose=True)

    print("Sync manager loaded. Use sync.py CLI for operations.")
