"""
Vector store using ChromaDB for semantic search.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import os

from offline_tools.embeddings import EmbeddingService


def get_default_chroma_path() -> str:
    """Get default ChromaDB path from local_config or BACKUP_PATH env var"""
    # Try local_config first (user's GUI setting)
    try:
        from admin.local_config import get_local_config
        config = get_local_config()
        backup_folder = config.get_backup_folder()
        if backup_folder:
            return os.path.join(backup_folder, "chroma")
    except ImportError:
        pass

    # Fallback to env var
    backup_path = os.getenv("BACKUP_PATH", "")
    if backup_path:
        return os.path.join(backup_path, "chroma")
    return "data/chroma"


class VectorStore:
    """ChromaDB-based vector store for article embeddings"""

    def __init__(self, persist_dir: str = None, collection_name: str = "articles"):
        """
        Args:
            persist_dir: Directory to persist ChromaDB data (defaults to BACKUP_PATH/chroma)
            collection_name: Name of the collection
        """
        if persist_dir is None:
            persist_dir = get_default_chroma_path()
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

    def add_documents(self, documents: List[Dict[str, Any]],
                      embeddings: Optional[List[List[float]]] = None,
                      progress_callback=None,
                      return_index_data: bool = False) -> Any:
        """
        Add documents to the vector store.

        Args:
            documents: List of dicts with keys: id, content, metadata
            embeddings: Pre-computed embeddings (optional, will compute if not provided)
            progress_callback: Optional function(current, total) for progress
            return_index_data: If True, returns dict with count and index data for saving

        Returns:
            Number of documents added (int), or dict with index data if return_index_data=True
        """
        if not documents:
            return {"count": 0, "index_data": None} if return_index_data else 0

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

        # Note: Metadata is now saved by the indexer directly to {source_id}_metadata.json
        # The old MetadataIndex source files are no longer needed

        if return_index_data:
            # Build index data structure for saving to file
            index_data = {
                "ids": ids,
                "embeddings": [list(e) for e in embeddings],  # Ensure lists not numpy arrays
                "contents": contents,
                "metadatas": metadatas
            }
            return {"count": len(ids), "index_data": index_data}

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
        # Get source breakdown directly from ChromaDB
        sources = {}
        try:
            result = self.collection.get(include=["metadatas"])
            for metadata in result.get("metadatas", []):
                if metadata and "source" in metadata:
                    source = metadata["source"]
                    sources[source] = sources.get(source, 0) + 1
        except:
            pass

        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name,
            "sources": sources
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

    def get_metadata_stats(self) -> Dict[str, Any]:
        """Get stats - now just calls get_stats()"""
        return self.get_stats()

    def search_offline(self, query: str, n_results: int = 5,
                       filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Keyword-based search for offline mode (no API calls).

        Uses simple text matching against document content and titles.
        Not as good as semantic search, but works completely offline.

        Args:
            query: Search query string
            n_results: Number of results to return
            filter: Optional ChromaDB filter dict

        Returns:
            List of matching documents with scores
        """
        # Get all documents (with filter if specified)
        try:
            if filter:
                result = self.collection.get(
                    where=filter,
                    include=["documents", "metadatas"]
                )
            else:
                result = self.collection.get(include=["documents", "metadatas"])
        except Exception as e:
            print(f"Error getting documents for offline search: {e}")
            return []

        if not result.get("ids"):
            return []

        query_terms = query.lower().split()

        # Score each document
        scored = []
        for i, doc_id in enumerate(result["ids"]):
            content = result["documents"][i] if result["documents"] else ""
            metadata = result["metadatas"][i] if result["metadatas"] else {}

            title = metadata.get("title", "").lower()
            content_lower = content.lower() if content else ""

            # Simple scoring: title matches worth more than content
            score = 0
            for term in query_terms:
                if term in title:
                    score += 5  # Title match
                if term in content_lower:
                    score += content_lower.count(term)  # Content frequency

            if score > 0:
                # Parse categories if present
                if "categories" in metadata:
                    try:
                        metadata["categories"] = json.loads(metadata["categories"])
                    except:
                        metadata["categories"] = []

                scored.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "raw_score": score
                })

        # Sort by score descending
        scored.sort(key=lambda x: x["raw_score"], reverse=True)

        # Normalize scores to 0-1 range and return top results
        max_score = scored[0]["raw_score"] if scored else 1
        results = []
        for item in scored[:n_results]:
            results.append({
                "id": item["id"],
                "content": item["content"],
                "metadata": item["metadata"],
                "score": min(item["raw_score"] / max_score, 1.0)
            })

        return results


# Quick test
if __name__ == "__main__":
    store = VectorStore()
    print(f"Stats: {store.get_stats()}")

    # Test search
    results = store.search("how to filter water", n_results=3)
    print(f"Found {len(results)} results")
    for r in results:
        print(f"  - {r['metadata'].get('title', 'Unknown')} (score: {r['score']:.3f})")
