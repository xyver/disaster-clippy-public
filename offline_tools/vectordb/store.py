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

    def __init__(self, persist_dir: str = None, collection_name: str = "articles", read_only: bool = False):
        """
        Args:
            persist_dir: Directory to persist ChromaDB data (defaults to BACKUP_PATH/chroma)
            collection_name: Name of the collection
            read_only: If True, skip embedding model initialization (faster for read-only ops)
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

        # Embedding service for queries - skip in read_only mode for faster init
        self._read_only = read_only
        self._embedding_service = None if read_only else EmbeddingService()

        # Metadata index for sync operations - reads from backup folder JSON files
        from .metadata import MetadataIndex
        self.metadata_index = MetadataIndex()

    @property
    def embedding_service(self):
        """Lazy-load embedding service on first access if in read_only mode"""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

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
            # ChromaDB requires string/int/float/bool values - serialize lists
            metadata["categories"] = json.dumps(metadata.get("categories", []))
            # Remove internal_links from ChromaDB metadata (stored in _metadata.json instead)
            # This avoids serialization overhead and keeps ChromaDB metadata lean
            metadata.pop("internal_links", None)
            metadatas.append(metadata)

        # Compute embeddings if not provided
        if embeddings is None:
            print(f"Computing embeddings for {len(contents)} documents...")
            embeddings = self.embedding_service.embed_batch(
                contents,
                progress_callback=progress_callback
            )

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

    def get_source_document_ids(self, source_id: str) -> set:
        """
        Get all document IDs for a specific source.

        Used by incremental indexing to skip already-indexed documents.

        Args:
            source_id: The source identifier to query

        Returns:
            Set of document IDs for this source
        """
        try:
            result = self.collection.get(
                where={"source": source_id},
                include=[]  # Only need IDs, not content/metadata
            )
            return set(result.get("ids", []))
        except Exception as e:
            print(f"[VectorStore] Error getting source IDs for {source_id}: {e}")
            return set()

    def has_valid_embedding(self, doc_id: str) -> bool:
        """
        Check if a document has a valid (non-zero) embedding.

        Used to detect documents that were indexed but have bad embeddings
        (e.g., due to API errors). These should be re-indexed.

        Args:
            doc_id: The document ID to check

        Returns:
            True if embedding exists and is non-zero, False otherwise
        """
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["embeddings"]
            )
            if not result["ids"]:
                return False

            embedding = result["embeddings"][0] if result["embeddings"] else None
            if embedding is None:
                return False

            # Check if all values are zero (invalid embedding)
            if all(v == 0 for v in embedding):
                return False

            return True
        except Exception as e:
            print(f"[VectorStore] Error checking embedding for {doc_id}: {e}")
            return False

    def add_documents_incremental(self, documents: List[Dict[str, Any]],
                                  source_id: str,
                                  batch_size: int = 100,
                                  progress_callback=None,
                                  return_index_data: bool = False) -> Any:
        """
        Add documents incrementally with resume support.

        Processes documents in batches, persisting after each batch.
        Skips documents that are already indexed (by ID).

        This allows resuming if interrupted - only the current batch is lost.

        Args:
            documents: List of document dicts
            source_id: Source identifier (used to check existing docs)
            batch_size: Documents per batch (default 100)
            progress_callback: Function(current, total, message) for progress
            return_index_data: If True, return index data for file saving

        Returns:
            Dict with count, skipped, index_data (if requested)
        """
        if not documents:
            return {"count": 0, "skipped": 0, "index_data": None}

        # Get existing doc IDs for this source
        existing_ids = self.get_source_document_ids(source_id)
        print(f"[VectorStore] Found {len(existing_ids)} existing documents for {source_id}")

        # Filter documents: keep new ones AND ones with invalid embeddings
        new_documents = []
        skipped_count = 0
        reindex_count = 0
        reindex_ids = []  # Track IDs that need re-indexing (have bad embeddings)

        for doc in documents:
            doc_id = doc.get("id")
            if doc_id not in existing_ids:
                # New document - add it
                new_documents.append(doc)
            else:
                # Existing document - check if it has a valid embedding
                if not self.has_valid_embedding(doc_id):
                    # Bad embedding - needs re-indexing
                    new_documents.append(doc)
                    reindex_ids.append(doc_id)
                    reindex_count += 1
                else:
                    # Valid embedding - skip
                    skipped_count += 1

        if skipped_count > 0:
            print(f"[VectorStore] Skipping {skipped_count} already-indexed documents")

        if reindex_count > 0:
            print(f"[VectorStore] Re-indexing {reindex_count} documents with invalid embeddings")
            # Delete the old entries so we can re-add them with fresh embeddings
            for doc_id in reindex_ids:
                try:
                    self.collection.delete(ids=[doc_id])
                except Exception as e:
                    print(f"[VectorStore] Warning: could not delete {doc_id}: {e}")

        if not new_documents:
            print(f"[VectorStore] All documents already indexed")
            return {"count": 0, "skipped": skipped_count, "reindexed": 0, "index_data": None, "resumed": True}

        print(f"[VectorStore] Processing {len(new_documents)} new documents in batches of {batch_size}")

        total_added = 0
        all_index_data = {"ids": [], "embeddings": [], "contents": [], "metadatas": []}

        # Process in batches
        for batch_start in range(0, len(new_documents), batch_size):
            batch_end = min(batch_start + batch_size, len(new_documents))
            batch = new_documents[batch_start:batch_end]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(new_documents) + batch_size - 1) // batch_size

            if progress_callback:
                progress_callback(batch_start, len(new_documents),
                                f"Batch {batch_num}/{total_batches}: Processing {len(batch)} documents...")

            # Prepare batch data
            ids = []
            contents = []
            metadatas = []

            for doc in batch:
                doc_id = doc.get("id") or doc.get("content_hash") or str(hash(doc["url"]))
                ids.append(doc_id)
                contents.append(doc["content"])

                # Exclude content and internal_links (lists not allowed in ChromaDB metadata)
                metadata = {k: v for k, v in doc.items() if k not in ("content", "internal_links")}
                metadata["categories"] = json.dumps(metadata.get("categories", []))
                metadatas.append(metadata)

            # Compute embeddings for this batch
            embeddings = self.embedding_service.embed_batch(contents)

            # Add to ChromaDB (persists immediately)
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )

            total_added += len(ids)
            print(f"[VectorStore] Batch {batch_num}/{total_batches}: Added {len(ids)} documents (total: {total_added})")

            # Accumulate index data if needed
            if return_index_data:
                all_index_data["ids"].extend(ids)
                all_index_data["embeddings"].extend([list(e) for e in embeddings])
                all_index_data["contents"].extend(contents)
                all_index_data["metadatas"].extend(metadatas)

        if progress_callback:
            progress_callback(len(new_documents), len(new_documents), "Indexing complete")

        result = {
            "count": total_added,
            "skipped": skipped_count,
            "reindexed": reindex_count,
            "resumed": skipped_count > 0 or reindex_count > 0
        }

        if return_index_data:
            result["index_data"] = all_index_data if total_added > 0 else None

        return result

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

    def delete_source(self, source_id: str) -> int:
        """
        Delete all documents from a specific source.

        Args:
            source_id: The source identifier to delete

        Returns:
            Number of documents deleted
        """
        result = self.delete_by_source(source_id)
        return result.get("deleted_count", 0)

    def delete_by_source(self, source_id: str, progress_callback=None) -> Dict[str, Any]:
        """
        Delete all vectors for a specific source.

        This method matches the PineconeStore interface for unified handling.

        Args:
            source_id: The source identifier to delete
            progress_callback: Optional function(current, total, message) for progress

        Returns:
            Dict with deletion stats: {deleted_count, batches}
        """
        try:
            print(f"[VectorStore] Deleting documents for source: '{source_id}'")

            # Get all document IDs for this source
            result = self.collection.get(
                where={"source": source_id},
                include=[]
            )
            ids_to_delete = result.get("ids", [])
            total_to_delete = len(ids_to_delete)
            print(f"[VectorStore] Found {total_to_delete} documents to delete")

            if ids_to_delete:
                # Delete in batches to prevent long blocking and show progress
                BATCH_SIZE = 500
                deleted = 0
                batches = 0

                for i in range(0, total_to_delete, BATCH_SIZE):
                    batch_ids = ids_to_delete[i:i + BATCH_SIZE]
                    self.collection.delete(ids=batch_ids)
                    deleted += len(batch_ids)
                    batches += 1

                    if progress_callback:
                        progress_callback(deleted, total_to_delete, f"Deleting old documents ({deleted}/{total_to_delete})...")
                    else:
                        print(f"[VectorStore] Deleted {deleted}/{total_to_delete} documents...")

                print(f"[VectorStore] Deleted {total_to_delete} documents from source '{source_id}'")
                return {"deleted_count": total_to_delete, "batches": batches}
            else:
                print(f"[VectorStore] No documents found for source '{source_id}'")

            return {"deleted_count": 0, "batches": 0}
        except Exception as e:
            print(f"[VectorStore] Error deleting source {source_id}: {e}")
            import traceback
            traceback.print_exc()
            return {"deleted_count": 0, "batches": 0, "error": str(e)}

    def get_source_vector_count(self, source_id: str) -> int:
        """
        Get count of vectors for a specific source.

        This method matches the PineconeStore interface for unified handling.

        Returns:
            Number of vectors for this source
        """
        try:
            result = self.collection.get(
                where={"source": source_id},
                include=[]
            )
            return len(result.get("ids", []))
        except Exception as e:
            print(f"[VectorStore] Error counting source {source_id}: {e}")
            return 0

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
