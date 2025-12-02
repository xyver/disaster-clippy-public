"""
Vector store using ChromaDB for semantic search.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
from pathlib import Path
import json

from .embeddings import EmbeddingService
from .metadata import MetadataIndex


class VectorStore:
    """ChromaDB-based vector store for article embeddings"""

    def __init__(self, persist_dir: str = "data/chroma", collection_name: str = "articles"):
        """
        Args:
            persist_dir: Directory to persist ChromaDB data
            collection_name: Name of the collection
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "DIY/humanitarian article embeddings"}
        )

        # Embedding service for queries
        self.embedding_service = EmbeddingService()

        # Metadata index for fast lookups
        self.metadata_index = MetadataIndex()

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

            # Store metadata (everything except content)
            metadata = {k: v for k, v in doc.items() if k != "content"}
            # ChromaDB requires string/int/float/bool values
            metadata["categories"] = json.dumps(metadata.get("categories", []))
            metadatas.append(metadata)

        # Compute embeddings if not provided
        if embeddings is None:
            print(f"Computing embeddings for {len(contents)} documents...")
            embeddings = self.embedding_service.embed_batch(contents)

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )

        # Update metadata index
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
            filter: Optional ChromaDB filter dict (e.g., {"source": {"$in": ["a", "b"]}})

        Returns:
            List of matching documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)

        # Build where filter if needed
        where_filter = None
        if filter:
            # Use provided filter dict directly
            where_filter = filter
        elif source_filter:
            where_filter = {"source": source_filter}

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            metadata = results["metadatas"][0][i]
            # Parse categories back from JSON
            if "categories" in metadata:
                try:
                    metadata["categories"] = json.loads(metadata["categories"])
                except:
                    metadata["categories"] = []

            # Convert L2 distance to similarity score (0-1 range)
            # Using 1/(1+d) formula which always produces 0-1 values
            distance = results["distances"][0][i]
            similarity = 1 / (1 + distance)

            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": metadata,
                "score": similarity
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
        # Get the document's embedding
        result = self.collection.get(ids=[doc_id], include=["embeddings"])

        if not result["embeddings"]:
            return []

        embedding = result["embeddings"][0]

        # Query with that embedding (exclude the original)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results + 1,  # +1 because it might include itself
            include=["documents", "metadatas", "distances"]
        )

        # Format and filter out the original document
        formatted = []
        for i in range(len(results["ids"][0])):
            if results["ids"][0][i] == doc_id:
                continue

            metadata = results["metadatas"][0][i]
            if "categories" in metadata:
                try:
                    metadata["categories"] = json.loads(metadata["categories"])
                except:
                    metadata["categories"] = []

            # Convert L2 distance to similarity score (0-1 range)
            distance = results["distances"][0][i]
            similarity = 1 / (1 + distance)

            formatted.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": metadata,
                "score": similarity
            })

            if len(formatted) >= n_results:
                break

        return formatted

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics including source breakdown"""
        # Get metadata stats which includes source breakdown
        metadata_stats = self.metadata_index.get_stats()

        # Build sources dict with counts
        sources = {}
        for source_name, source_info in metadata_stats.get("sources", {}).items():
            sources[source_name] = source_info.get("count", 0)

        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name,
            "sources": sources,
            "last_updated": metadata_stats.get("last_updated")
        }

    def get_existing_ids(self) -> set:
        """Get all document IDs currently in the database"""
        try:
            result = self.collection.get()
            return set(result.get("ids", []))
        except:
            return set()

    def get_existing_titles(self) -> set:
        """Get all document titles currently in the database"""
        try:
            result = self.collection.get(include=["metadatas"])
            titles = set()
            for metadata in result.get("metadatas", []):
                if metadata and "title" in metadata:
                    titles.add(metadata["title"])
            return titles
        except:
            return set()

    def delete_all(self):
        """Delete all documents (use with caution!)"""
        # Get all IDs
        all_ids = self.collection.get()["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
        # Clear metadata index too
        self.metadata_index.clear()

    def get_metadata_stats(self) -> Dict[str, Any]:
        """Get fast stats from metadata index (no DB query needed)"""
        return self.metadata_index.get_stats()

    def rebuild_metadata_index(self):
        """Rebuild metadata index from vector DB (if out of sync)"""
        print("Rebuilding metadata index from vector database...")
        result = self.collection.get(include=["metadatas", "documents"])

        docs = []
        for i, doc_id in enumerate(result.get("ids", [])):
            metadata = result["metadatas"][i] if result.get("metadatas") else {}
            content = result["documents"][i] if result.get("documents") else ""

            # Parse categories back from JSON
            categories = []
            if "categories" in metadata:
                try:
                    categories = json.loads(metadata["categories"])
                except:
                    pass

            docs.append({
                "id": doc_id,
                "title": metadata.get("title", "Unknown"),
                "url": metadata.get("url", ""),
                "source": metadata.get("source", "unknown"),
                "categories": categories,
                "content_hash": metadata.get("content_hash", ""),
                "scraped_at": metadata.get("scraped_at", ""),
                "content": content
            })

        self.metadata_index.sync_from_vectordb(docs)
        print(f"Rebuilt index with {len(docs)} documents")


# Quick test
if __name__ == "__main__":
    store = VectorStore()
    print(f"Stats: {store.get_stats()}")

    # Test search
    results = store.search("how to filter water", n_results=3)
    print(f"Found {len(results)} results")
    for r in results:
        print(f"  - {r['metadata'].get('title', 'Unknown')} (score: {r['score']:.3f})")
