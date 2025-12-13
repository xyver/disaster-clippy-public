"""
Pinecone vector store for cloud-based semantic search.
Drop-in replacement for ChromaDB VectorStore when using Pinecone.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass, field, asdict

try:
    from pinecone import Pinecone, ServerlessSpec
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

from offline_tools.embeddings import EmbeddingService
from .metadata import MetadataIndex


# =============================================================================
# PINECONE SYNC CHECKPOINT SYSTEM
# =============================================================================

@dataclass
class PineconeSyncCheckpoint:
    """
    Checkpoint for resumable Pinecone sync operations.

    Tracks which batches have been uploaded AND verified to exist in Pinecone.
    This ensures no data is lost even if interrupted mid-sync.
    """
    source_id: str
    total_docs: int = 0
    total_batches: int = 0
    batch_size: int = 100

    # Track completed batches (batch_num -> list of doc IDs in that batch)
    completed_batches: Dict[int, List[str]] = field(default_factory=dict)

    # All doc IDs that have been verified in Pinecone
    verified_ids: Set[str] = field(default_factory=set)

    # Progress tracking
    created_at: str = ""
    last_saved: str = ""
    last_verified_at: str = ""

    # Error tracking
    failed_batches: List[int] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "total_docs": self.total_docs,
            "total_batches": self.total_batches,
            "batch_size": self.batch_size,
            "completed_batches": self.completed_batches,
            "verified_ids": list(self.verified_ids),
            "created_at": self.created_at,
            "last_saved": self.last_saved,
            "last_verified_at": self.last_verified_at,
            "failed_batches": self.failed_batches,
            "errors": self.errors
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PineconeSyncCheckpoint":
        checkpoint = cls(
            source_id=data.get("source_id", ""),
            total_docs=data.get("total_docs", 0),
            total_batches=data.get("total_batches", 0),
            batch_size=data.get("batch_size", 100),
            created_at=data.get("created_at", ""),
            last_saved=data.get("last_saved", ""),
            last_verified_at=data.get("last_verified_at", ""),
            failed_batches=data.get("failed_batches", []),
            errors=data.get("errors", [])
        )
        checkpoint.completed_batches = data.get("completed_batches", {})
        # Convert string keys back to int (JSON serialization issue)
        checkpoint.completed_batches = {int(k): v for k, v in checkpoint.completed_batches.items()}
        checkpoint.verified_ids = set(data.get("verified_ids", []))
        return checkpoint


def _get_pinecone_checkpoint_path(source_id: str) -> Optional[Path]:
    """Get the path for a Pinecone sync checkpoint file."""
    try:
        from admin.job_manager import get_jobs_folder
        jobs_folder = get_jobs_folder()
        if jobs_folder:
            return jobs_folder / f"{source_id}_pinecone_sync.checkpoint.json"
    except ImportError:
        pass
    return None


def save_pinecone_checkpoint(checkpoint: PineconeSyncCheckpoint) -> bool:
    """Save Pinecone sync checkpoint to disk (atomic write)."""
    checkpoint_path = _get_pinecone_checkpoint_path(checkpoint.source_id)
    if not checkpoint_path:
        print("[pinecone-checkpoint] Cannot save - no jobs folder")
        return False

    checkpoint.last_saved = datetime.now().isoformat()
    if not checkpoint.created_at:
        checkpoint.created_at = checkpoint.last_saved

    try:
        temp_path = checkpoint_path.with_suffix('.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        temp_path.replace(checkpoint_path)
        return True
    except Exception as e:
        print(f"[pinecone-checkpoint] Error saving: {e}")
        return False


def load_pinecone_checkpoint(source_id: str) -> Optional[PineconeSyncCheckpoint]:
    """Load Pinecone sync checkpoint from disk."""
    checkpoint_path = _get_pinecone_checkpoint_path(source_id)
    if not checkpoint_path or not checkpoint_path.exists():
        return None

    try:
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        checkpoint = PineconeSyncCheckpoint.from_dict(data)
        print(f"[pinecone-checkpoint] Loaded: {len(checkpoint.completed_batches)} batches completed, {len(checkpoint.verified_ids)} verified IDs")
        return checkpoint
    except Exception as e:
        print(f"[pinecone-checkpoint] Error loading: {e}")
        return None


def delete_pinecone_checkpoint(source_id: str) -> bool:
    """Delete Pinecone sync checkpoint after successful completion."""
    checkpoint_path = _get_pinecone_checkpoint_path(source_id)
    if checkpoint_path and checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            print(f"[pinecone-checkpoint] Deleted checkpoint for {source_id}")
            return True
        except Exception as e:
            print(f"[pinecone-checkpoint] Error deleting: {e}")
    return False


class PineconeMetadataWrapper:
    """
    Wrapper that provides MetadataIndex-like interface but queries Pinecone directly.
    Used for sync operations to get actual cloud state, not local file state.
    """

    def __init__(self, pinecone_store):
        self.store = pinecone_store
        self._id_cache = None
        self._source_cache = {}

    def get_ids(self, source: str = None) -> set:
        """Get document IDs from Pinecone (expensive - use sparingly)"""
        # Query Pinecone for IDs with the given source filter
        try:
            # Use a dummy vector to fetch with metadata filter
            if source:
                # Fetch IDs by querying with source filter
                # Note: Pinecone list() is more efficient but may not support filtering
                results = self.store.index.query(
                    vector=[0.0] * 1536,
                    top_k=10000,  # Max allowed
                    filter={"source": source},
                    include_metadata=False
                )
                return {m.id for m in results.matches}
            else:
                # No filter - get stats to see namespaces, then query each
                stats = self.store.index.describe_index_stats()
                if stats.total_vector_count == 0:
                    return set()
                # Query without filter to get IDs
                results = self.store.index.query(
                    vector=[0.0] * 1536,
                    top_k=10000,
                    include_metadata=False
                )
                return {m.id for m in results.matches}
        except Exception as e:
            print(f"[PineconeMetadataWrapper] Error querying IDs: {e}")
            return set()

    def get_titles(self, source: str = None) -> set:
        """Get titles from Pinecone"""
        try:
            filter_dict = {"source": source} if source else None
            results = self.store.index.query(
                vector=[0.0] * 1536,
                top_k=10000,
                filter=filter_dict,
                include_metadata=True
            )
            return {m.metadata.get("title", "") for m in results.matches if m.metadata}
        except Exception:
            return set()

    def list_sources(self) -> list:
        """List unique sources in Pinecone"""
        # This is expensive - would need to query and extract unique sources
        # For now, return empty - sync will handle this differently
        return []

    def add_documents(self, documents: list):
        """No-op - Pinecone metadata is managed by the store itself"""
        # The actual vectors are added via PineconeStore.add_documents()
        # This wrapper just needs to not throw an error
        pass

    def get_stats(self):
        """Get stats from Pinecone"""
        stats = self.store.index.describe_index_stats()
        return {
            "total_documents": stats.total_vector_count,
            "sources": {}
        }


class PineconeStore:
    """Pinecone-based vector store for article embeddings"""

    def __init__(self, index_name: Optional[str] = None, namespace: str = "default"):
        """
        Args:
            index_name: Pinecone index name (defaults to env var PINECONE_INDEX_NAME)
            namespace: Namespace within the index (useful for multi-tenant)
        """
        if not HAS_PINECONE:
            raise ImportError(
                "Pinecone not installed. Install with:\n"
                "  pip install pinecone"
            )

        # Get configuration from environment
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "disaster-clippy")
        self.namespace = namespace

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)

        # Get or create index
        self._ensure_index()

        # Connect to index
        self.index = self.pc.Index(self.index_name)

        # Embedding service for queries
        self.embedding_service = EmbeddingService()

        # Metadata index wrapper that queries Pinecone directly
        # (not local files - those reflect local state, not cloud state)
        self.metadata_index = PineconeMetadataWrapper(self)

        # Cache for R2 source data (avoid repeated downloads)
        # Cache is long-lived (1 hour) since data rarely changes
        # Call invalidate_sources_cache() after syncing new data to R2
        self._r2_sources_cache = None
        self._r2_sources_cache_time = 0

    def invalidate_sources_cache(self):
        """Clear the R2 sources cache. Call after uploading new data to R2."""
        self._r2_sources_cache = None
        self._r2_sources_cache_time = 0

    def _ensure_index(self):
        """Create index if it doesn't exist"""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            # Check if there's a similar index we can use
            print(f"Index '{self.index_name}' not found.")
            print(f"Available indexes: {existing_indexes}")

            # Try to find disaster-clippy variant
            for idx in existing_indexes:
                if "disaster-clippy" in idx:
                    print(f"Using existing index: {idx}")
                    self.index_name = idx
                    return

            # Create new index if none found
            print(f"Creating Pinecone index: {self.index_name}")
            # OpenAI embeddings are 1536 dimensions
            # Extract just the region (e.g., "us-east-1" from "us-east-1-aws")
            env = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
            region = env.replace("-aws", "").replace("-gcp", "").replace("-azure", "")

            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=region
                )
            )
            print(f"Index {self.index_name} created successfully")

    def add_documents(self, documents: List[Dict[str, Any]],
                      embeddings: Optional[List[List[float]]] = None,
                      progress_callback=None,
                      resume: bool = True) -> int:
        """
        Add documents to the vector store with checkpoint support.

        Args:
            documents: List of dicts with keys: id, content, metadata
            embeddings: Pre-computed embeddings (optional, will compute if not provided)
            progress_callback: Optional function(batch_num, total_batches, message) for progress
            resume: If True, attempt to resume from checkpoint (default True)

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

        # Extract source_id for checkpointing
        source_id = documents[0].get("source", "unknown") if documents else "unknown"

        ids = []
        contents = []
        metadatas = []

        for doc in documents:
            # Generate unique ID from URL hash
            doc_id = doc.get("id") or doc.get("content_hash") or str(hash(doc["url"]))
            ids.append(doc_id)
            contents.append(doc["content"])

            # Store metadata (everything except content - Pinecone has metadata limits)
            metadata = {
                "title": doc.get("title", "Unknown")[:500],  # Truncate for Pinecone limits
                "url": doc.get("url", ""),
                "source": doc.get("source", "unknown"),
                "categories": json.dumps(doc.get("categories", []))[:500],
                "content_hash": doc.get("content_hash", ""),
                "scraped_at": doc.get("scraped_at", ""),
                "doc_type": doc.get("doc_type", "article"),
                # Store first 1000 chars of content for retrieval
                "content_preview": doc.get("content", "")[:1000]
            }
            metadatas.append(metadata)

        # Use pre-computed embeddings if provided (from sync), otherwise compute
        if embeddings is None:
            # Check if documents have embeddings already
            if documents and "embedding" in documents[0]:
                embeddings = [doc["embedding"] for doc in documents]
            else:
                print(f"Computing embeddings for {len(contents)} documents...")
                embeddings = self.embedding_service.embed_batch(contents)

        # Prepare vectors for upsert, filtering out zero vectors
        vectors = []
        skipped_zero_vectors = 0
        for i, (doc_id, embedding, metadata) in enumerate(zip(ids, embeddings, metadatas)):
            # Check for zero vectors (Pinecone rejects these)
            if embedding is None or all(v == 0 for v in embedding):
                skipped_zero_vectors += 1
                print(f"  [SKIP] Zero vector for doc: {metadata.get('title', doc_id)[:50]}")
                continue

            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            })

        if skipped_zero_vectors > 0:
            print(f"[pinecone] Skipped {skipped_zero_vectors} documents with zero/null embeddings")

        # Checkpoint setup
        batch_size = 100
        total_batches = (len(vectors) - 1) // batch_size + 1 if vectors else 0
        checkpoint = None
        skipped_batches = 0
        verified_ids_count = 0

        # Try to load checkpoint for resume
        if resume:
            checkpoint = load_pinecone_checkpoint(source_id)
            if checkpoint:
                # Verify checkpoint matches current upload
                if checkpoint.total_docs != len(documents):
                    print(f"[pinecone] Checkpoint mismatch: checkpoint has {checkpoint.total_docs} docs, current has {len(documents)}")
                    print(f"[pinecone] Starting fresh (document count changed)")
                    delete_pinecone_checkpoint(source_id)
                    checkpoint = None
                else:
                    # Verify a sample of completed batches actually exist in Pinecone
                    verified_ids_count = len(checkpoint.verified_ids)
                    if checkpoint.completed_batches:
                        print(f"[pinecone] Resuming: {len(checkpoint.completed_batches)} batches already completed")
                        print(f"[pinecone] Verifying {min(3, len(checkpoint.completed_batches))} sample batches exist in Pinecone...")

                        # Verify a few random batches
                        sample_batches = list(checkpoint.completed_batches.keys())[:3]
                        all_verified = True
                        for batch_num in sample_batches:
                            batch_ids = checkpoint.completed_batches[batch_num]
                            # Check if these IDs exist in Pinecone
                            try:
                                fetch_result = self.index.fetch(ids=batch_ids[:10], namespace=self.namespace)
                                found_ids = set(fetch_result.vectors.keys()) if fetch_result.vectors else set()
                                expected_sample = set(batch_ids[:10])
                                if not expected_sample.issubset(found_ids):
                                    print(f"[pinecone] WARNING: Batch {batch_num} verification failed - some IDs missing")
                                    all_verified = False
                                    break
                            except Exception as e:
                                print(f"[pinecone] Verification error for batch {batch_num}: {e}")
                                all_verified = False
                                break

                        if not all_verified:
                            print(f"[pinecone] Checkpoint verification failed - starting fresh")
                            delete_pinecone_checkpoint(source_id)
                            checkpoint = None
                        else:
                            print(f"[pinecone] Checkpoint verified - skipping {len(checkpoint.completed_batches)} completed batches")

        # Create new checkpoint if needed
        if checkpoint is None:
            checkpoint = PineconeSyncCheckpoint(
                source_id=source_id,
                total_docs=len(documents),
                total_batches=total_batches,
                batch_size=batch_size
            )

        # Upsert in batches with checkpoint support
        uploaded_count = 0
        for i in range(0, len(vectors), batch_size):
            batch_num = i // batch_size + 1

            # Skip already-completed batches
            if batch_num in checkpoint.completed_batches:
                skipped_batches += 1
                if progress_callback:
                    progress_callback(batch_num, total_batches, f"Skipping batch {batch_num}/{total_batches} (already uploaded)")
                continue

            batch = vectors[i:i + batch_size]
            batch_ids = [v["id"] for v in batch]

            try:
                # Upload batch to Pinecone
                self.index.upsert(vectors=batch, namespace=self.namespace)
                uploaded_count += len(batch)

                # Mark batch as completed
                checkpoint.completed_batches[batch_num] = batch_ids
                checkpoint.verified_ids.update(batch_ids)

                # Save checkpoint every 10 batches or on last batch
                if batch_num % 10 == 0 or batch_num == total_batches:
                    save_pinecone_checkpoint(checkpoint)

                print(f"  Uploaded batch {batch_num}/{total_batches} ({len(batch)} vectors)")

                # Report batch progress if callback provided
                if progress_callback:
                    progress_callback(batch_num, total_batches, f"Uploading batch {batch_num}/{total_batches}")

            except Exception as e:
                print(f"  ERROR uploading batch {batch_num}: {e}")
                checkpoint.failed_batches.append(batch_num)
                checkpoint.errors.append({"batch": batch_num, "error": str(e)})
                save_pinecone_checkpoint(checkpoint)
                raise  # Re-raise to signal failure

        # Update local metadata index
        self.metadata_index.add_documents(documents)

        # Delete checkpoint on successful completion
        if len(checkpoint.completed_batches) == total_batches:
            delete_pinecone_checkpoint(source_id)
            print(f"[pinecone] Sync complete: {uploaded_count} uploaded, {skipped_batches} skipped (from checkpoint)")
        else:
            # Save final state if not all batches completed
            save_pinecone_checkpoint(checkpoint)

        return len(ids)

    def search(self, query: str, n_results: int = 5,
               source_filter: Optional[str] = None,
               filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Semantic search for relevant documents.

        Args:
            query: Natural language query
            n_results: Number of results to return
            source_filter: Optional filter by single source (e.g., "appropedia")
            filter: Optional Pinecone filter dict (e.g., {"source": {"$in": ["a", "b"]}})

        Returns:
            List of matching documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Build filter if needed
        filter_dict = None
        if filter:
            # Use provided filter dict directly
            filter_dict = filter
        elif source_filter:
            filter_dict = {"source": {"$eq": source_filter}}

        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True,
            namespace=self.namespace,
            filter=filter_dict
        )

        # Format results
        formatted = []
        for match in results.matches:
            metadata = dict(match.metadata) if match.metadata else {}

            # Parse categories back from JSON
            categories = []
            if "categories" in metadata:
                try:
                    categories = json.loads(metadata["categories"])
                except:
                    pass
            metadata["categories"] = categories

            formatted.append({
                "id": match.id,
                "content": metadata.get("content_preview", ""),
                "metadata": metadata,
                "score": match.score  # Pinecone returns similarity score directly
            })

        return formatted

    def search_similar(self, doc_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find documents similar to a given document.

        Args:
            doc_id: ID of the reference document
            n_results: Number of similar documents to return

        Returns:
            List of similar documents
        """
        # Fetch the document to get its embedding
        result = self.index.fetch(ids=[doc_id], namespace=self.namespace)

        if not result.vectors or doc_id not in result.vectors:
            return []

        embedding = result.vectors[doc_id].values

        # Query with that embedding
        results = self.index.query(
            vector=embedding,
            top_k=n_results + 1,  # +1 to exclude self
            include_metadata=True,
            namespace=self.namespace
        )

        # Format and filter out the original document
        formatted = []
        for match in results.matches:
            if match.id == doc_id:
                continue

            metadata = dict(match.metadata) if match.metadata else {}
            categories = []
            if "categories" in metadata:
                try:
                    categories = json.loads(metadata["categories"])
                except:
                    pass
            metadata["categories"] = categories

            formatted.append({
                "id": match.id,
                "content": metadata.get("content_preview", ""),
                "metadata": metadata,
                "score": match.score
            })

            if len(formatted) >= n_results:
                break

        return formatted

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics including source breakdown"""
        stats = self.index.describe_index_stats()

        # Get metadata stats which includes source breakdown
        metadata_stats = self.metadata_index.get_stats()

        # Build sources dict with counts
        sources = {}
        for source_name, source_info in metadata_stats.get("sources", {}).items():
            sources[source_name] = source_info.get("count", 0)

        # If no sources from local metadata (e.g., Railway without local metadata),
        # try to get sources from R2 cloud storage
        if not sources and stats.total_vector_count > 0:
            sources = self._get_sources_from_r2()

        # If still no sources but Pinecone has vectors, query Pinecone directly
        if not sources and stats.total_vector_count > 0:
            sources = self._get_sources_from_pinecone()

        return {
            "total_documents": stats.total_vector_count,
            "index_name": self.index_name,
            "namespace": self.namespace,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {},
            "sources": sources,
            "last_updated": metadata_stats.get("last_updated")
        }

    def _get_sources_from_pinecone(self) -> Dict[str, int]:
        """
        Query Pinecone directly to discover unique sources.
        This is expensive but works without R2 or local metadata.
        """
        try:
            # Sample vectors to discover sources
            results = self.index.query(
                vector=[0.0] * 1536,
                top_k=10000,  # Max allowed
                include_metadata=True,
                namespace=self.namespace
            )

            # Count sources from results
            source_counts = {}
            for match in results.matches:
                if match.metadata and "source" in match.metadata:
                    source = match.metadata["source"]
                    source_counts[source] = source_counts.get(source, 0) + 1

            print(f"[PineconeStore] Discovered {len(source_counts)} sources from Pinecone query")
            return source_counts
        except Exception as e:
            print(f"[PineconeStore] Error querying sources from Pinecone: {e}")
            return {}

    def _get_sources_from_r2(self) -> Dict[str, int]:
        """
        Get source breakdown from R2 cloud storage metadata.
        Downloads metadata/_master.json from R2 and parses source counts.
        """
        full_info = self._get_full_sources_from_r2()
        # Return just counts for backward compatibility
        return {source_id: info.get("count", 0) for source_id, info in full_info.items()}

    def _get_full_sources_from_r2(self) -> Dict[str, Dict[str, Any]]:
        """
        Get full source info from R2 cloud storage metadata.
        Returns dict with source_id -> {name, count, topics, etc.}
        Uses caching to avoid repeated R2 downloads (1 hour cache).
        Call invalidate_sources_cache() after uploading new data.
        """
        import time

        # Check cache first (1 hour TTL - data rarely changes)
        cache_ttl = 3600  # 1 hour
        if self._r2_sources_cache is not None:
            if time.time() - self._r2_sources_cache_time < cache_ttl:
                return self._r2_sources_cache

        try:
            from offline_tools.cloud.r2 import get_backups_storage

            r2 = get_backups_storage()
            if not r2.is_configured():
                print("R2 storage not configured, cannot fetch metadata")
                return self._r2_sources_cache or {}

            # Use download_file_content for small JSON files (no temp file needed)
            content = r2.download_file_content("backups/_master.json")
            if not content:
                # Silently return cached or empty - this is expected if no master.json exists yet
                return self._r2_sources_cache or {}

            # Parse the metadata
            master_data = json.loads(content)

            # Cache the result
            self._r2_sources_cache = master_data.get("sources", {})
            self._r2_sources_cache_time = time.time()

            return self._r2_sources_cache

        except Exception as e:
            print(f"Warning: Could not get sources from R2: {e}")
            return self._r2_sources_cache or {}

    def get_existing_ids(self, limit: int = 10000) -> set:
        """Get document IDs from metadata index"""
        return self.metadata_index.get_ids()

    def get_existing_titles(self) -> set:
        """Get all document titles from local metadata index"""
        return self.metadata_index.get_titles()

    def delete_all(self):
        """Delete all documents in namespace (use with caution!)"""
        self.index.delete(delete_all=True, namespace=self.namespace)
        self.metadata_index.clear()

    def delete_by_ids(self, ids: List[str]):
        """Delete specific documents by ID"""
        if ids:
            self.index.delete(ids=ids, namespace=self.namespace)

    def delete_by_source(self, source_id: str) -> Dict[str, Any]:
        """
        Delete all vectors for a specific source.

        Queries Pinecone to find all vectors with source metadata matching source_id,
        then deletes them in batches.

        Returns:
            Dict with deletion stats: {deleted_count, batches}
        """
        print(f"[PineconeStore] Finding vectors for source: {source_id}")

        # Query to find all vectors with this source
        # Use a zero vector query with filter - Pinecone requires a vector for queries
        all_ids = []

        try:
            # Query with metadata filter for source
            # We need to paginate since top_k max is 10000
            results = self.index.query(
                vector=[0.0] * 1536,  # Dummy vector for metadata-only query
                top_k=10000,
                include_metadata=True,
                filter={"source": {"$eq": source_id}},
                namespace=self.namespace
            )

            for match in results.matches:
                all_ids.append(match.id)

            print(f"[PineconeStore] Found {len(all_ids)} vectors to delete for source {source_id}")

            if not all_ids:
                return {"deleted_count": 0, "batches": 0}

            # Delete in batches of 1000 (Pinecone limit)
            batch_size = 1000
            batches = 0
            for i in range(0, len(all_ids), batch_size):
                batch_ids = all_ids[i:i + batch_size]
                self.index.delete(ids=batch_ids, namespace=self.namespace)
                batches += 1
                print(f"[PineconeStore] Deleted batch {batches} ({len(batch_ids)} vectors)")

            return {"deleted_count": len(all_ids), "batches": batches}

        except Exception as e:
            print(f"[PineconeStore] Error deleting source {source_id}: {e}")
            raise

    def get_source_vector_count(self, source_id: str) -> int:
        """
        Get count of vectors for a specific source in Pinecone.

        Returns:
            Number of vectors for this source
        """
        try:
            results = self.index.query(
                vector=[0.0] * 1536,
                top_k=10000,
                include_metadata=False,
                filter={"source": {"$eq": source_id}},
                namespace=self.namespace
            )
            return len(results.matches)
        except Exception as e:
            print(f"[PineconeStore] Error counting source {source_id}: {e}")
            return 0

    def get_metadata_stats(self) -> Dict[str, Any]:
        """Get fast stats from local metadata index"""
        return self.metadata_index.get_stats()

    def fetch_vectors(self, ids: List[str]) -> Dict[str, Any]:
        """
        Fetch vectors by ID (for sync operations).

        Returns:
            Dict mapping id -> {values, metadata}
        """
        if not ids:
            return {}

        # Pinecone fetch limit is 1000
        all_vectors = {}
        batch_size = 1000
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            result = self.index.fetch(ids=batch_ids, namespace=self.namespace)
            if result.vectors:
                all_vectors.update(result.vectors)

        return all_vectors

    def upsert_vectors(self, vectors: List[Dict[str, Any]]):
        """
        Upsert pre-computed vectors (for sync operations).

        Args:
            vectors: List of {id, values, metadata} dicts
        """
        if not vectors:
            return

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)


def create_pinecone_store(**kwargs) -> PineconeStore:
    """Convenience function to create a Pinecone store"""
    return PineconeStore(**kwargs)


# Quick test
if __name__ == "__main__":
    if not HAS_PINECONE:
        print("Pinecone not installed. Install with:")
        print("  pip install pinecone")
    else:
        print("Testing Pinecone connection...")
        try:
            store = PineconeStore()
            stats = store.get_stats()
            print(f"Connected to index: {stats['index_name']}")
            print(f"Total vectors: {stats['total_documents']}")
            print("Pinecone store ready!")
        except Exception as e:
            print(f"Error: {e}")
