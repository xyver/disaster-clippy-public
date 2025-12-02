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

from .embeddings import EmbeddingService
from .metadata import MetadataIndex


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

        # Metadata index for fast lookups (local cache)
        self.metadata_index = MetadataIndex()

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
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=self.namespace)
            print(f"  Uploaded batch {i // batch_size + 1}/{(len(vectors) - 1) // batch_size + 1}")

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

        return {
            "total_documents": stats.total_vector_count,
            "index_name": self.index_name,
            "namespace": self.namespace,
            "namespaces": dict(stats.namespaces) if stats.namespaces else {},
            "sources": sources,
            "last_updated": metadata_stats.get("last_updated")
        }

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
