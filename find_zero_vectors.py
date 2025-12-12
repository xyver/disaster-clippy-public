"""
Find documents with zero embeddings in ChromaDB.
Run with: python find_zero_vectors.py
"""

import sys
sys.path.insert(0, '.')

from offline_tools.vectordb import VectorStore

def find_zero_vectors():
    print("Loading ChromaDB...")
    store = VectorStore()

    # Get all documents with embeddings
    print("Fetching all documents (this may take a moment)...")

    # ChromaDB collection
    collection = store.collection

    # Get count first
    count = collection.count()
    print(f"Total documents in ChromaDB: {count}")

    # Fetch in batches to avoid memory issues
    batch_size = 5000
    zero_vectors = []
    null_vectors = []

    for offset in range(0, count, batch_size):
        print(f"Checking documents {offset} to {offset + batch_size}...")

        result = collection.get(
            limit=batch_size,
            offset=offset,
            include=["embeddings", "metadatas"]
        )

        # Handle case where embeddings might be None or numpy array
        embeddings = result["embeddings"]
        if embeddings is None:
            embeddings = [None] * len(result["ids"])
        metadatas = result["metadatas"]
        if metadatas is None:
            metadatas = [{}] * len(result["ids"])

        for i, (doc_id, embedding, metadata) in enumerate(zip(
            result["ids"],
            embeddings,
            metadatas
        )):
            if embedding is None:
                null_vectors.append({
                    "id": doc_id,
                    "title": metadata.get("title", "Unknown")[:60],
                    "source": metadata.get("source", "unknown"),
                    "reason": "null embedding"
                })
            elif all(v == 0 for v in embedding):
                zero_vectors.append({
                    "id": doc_id,
                    "title": metadata.get("title", "Unknown")[:60],
                    "source": metadata.get("source", "unknown"),
                    "reason": "all zeros"
                })

    # Report findings
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(f"\nNull embeddings: {len(null_vectors)}")
    print(f"Zero embeddings: {len(zero_vectors)}")
    print(f"Total problematic: {len(null_vectors) + len(zero_vectors)}")

    # Group by source
    source_counts = {}
    for doc in null_vectors + zero_vectors:
        source = doc["source"]
        source_counts[source] = source_counts.get(source, 0) + 1

    print("\nBy source:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")

    # Show sample of affected documents
    all_bad = null_vectors + zero_vectors
    if all_bad:
        print("\nSample of affected documents:")
        for doc in all_bad[:20]:
            print(f"  [{doc['source']}] {doc['title']} ({doc['reason']})")
        if len(all_bad) > 20:
            print(f"  ... and {len(all_bad) - 20} more")

    # Save full list to file
    if all_bad:
        import json
        with open("zero_vectors_report.json", "w") as f:
            json.dump({
                "total_null": len(null_vectors),
                "total_zero": len(zero_vectors),
                "by_source": source_counts,
                "documents": all_bad
            }, f, indent=2)
        print(f"\nFull report saved to: zero_vectors_report.json")

    return all_bad

if __name__ == "__main__":
    find_zero_vectors()