"""
Pinecone vector store for cloud-based semantic search.
Drop-in replacement for ChromaDB VectorStore when using Pinecone.
"""

import os
import json
from typing import List, Dict, Optional, Any
from datetime import datetime

try:
    from pinecone import Pinecone, ServerlessSpec
    HAS_PINECONE = True
except ImportError:
    HAS_PINECONE = False

from offline_tools.embeddings import EmbeddingService
from .metadata import MetadataIndex


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
                      progress_callback=None) -> int:
        """
        Add documents to the vector store.

        Args:
            documents: List of dicts with keys: id, content, metadata
            embeddings: Pre-computed embeddings (optional, will compute if not provided)
            progress_callback: Optional function(current, total) for progress

        Returns:
            Number of documents added
        """
        if not documents:
            return 0

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

        # Prepare vectors for upsert
        vectors = []
        for i, (doc_id, embedding, metadata) in enumerate(zip(ids, embeddings, metadatas)):
            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            })

            if progress_callback:
                progress_callback(i + 1, len(documents))

        # Upsert in batches (Pinecone limit is 100 vectors per request)
        batch_size = 100
        total_batches = (len(vectors) - 1) // batch_size + 1 if vectors else 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
            batch_num = i // batch_size + 1
            print(f"  Uploaded batch {batch_num}/{total_batches}")
            # Report batch progress if callback provided
            if progress_callback:
                progress_callback(batch_num, total_batches, f"Uploading batch {batch_num}/{total_batches}")

        # Update local metadata index
        self.metadata_index.add_documents(documents)

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
        """
        try:
            from offline_tools.cloud.r2 import get_r2_storage
            import tempfile

            r2 = get_r2_storage()
            if not r2.is_configured():
                print("R2 storage not configured, cannot fetch metadata")
                return {}

            # Download _master.json to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                tmp_path = tmp.name

            # Try backups/_master.json first (current structure), then metadata/_master.json (legacy)
            if not r2.download_file("backups/_master.json", tmp_path):
                if not r2.download_file("metadata/_master.json", tmp_path):
                    # Silently return empty - this is expected if no master.json exists yet
                    return {}

            # Parse the metadata
            with open(tmp_path, 'r', encoding='utf-8') as f:
                master_data = json.load(f)

            # Clean up temp file
            import os as os_module
            os_module.unlink(tmp_path)

            # Return full sources data
            return master_data.get("sources", {})

        except Exception as e:
            print(f"Warning: Could not get sources from R2: {e}")
            return {}

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
