"""
Fix documents with zero embeddings by deleting and re-embedding them.
Run with: python fix_zero_vectors.py
"""

import sys
sys.path.insert(0, '.')

from offline_tools.vectordb import VectorStore
from offline_tools.embeddings import EmbeddingService

def fix_zero_vectors():
    print("Loading ChromaDB...")
    store = VectorStore()
    embedder = EmbeddingService()

    # The 7 documents with zero embeddings
    bad_ids = [
        '88958d3beaf44199c9da565b030987fc',  # Blogs relevant to Appropedia
        '197a643117e81ebb60843cee3b186ce4',  # Earthing (electrical)
        '127ab47557f89e08414ddc46301db003',  # Energy and Environmental Security Initiative
        '6b03783a68ec5613e302ada993748070',  # Humboldt SoE mechanical arm demo
        'ef82428ea93c25dae8a2a6f14785f19c',  # Lisa Blair
        '1da33099a100f3367337287149264f72',  # Low bandwidth browsing
        '7988c211949418960edd4fca59f4141f',  # News by UK location 2021
    ]

    print(f"\nFetching {len(bad_ids)} documents with zero embeddings...")

    # Get the documents with their content and metadata
    result = store.collection.get(
        ids=bad_ids,
        include=["documents", "metadatas"]
    )

    if not result["ids"]:
        print("No documents found!")
        return

    print(f"Found {len(result['ids'])} documents")

    # Re-embed each document
    fixed = 0
    for doc_id, content, metadata in zip(result["ids"], result["documents"], result["metadatas"]):
        title = metadata.get("title", "Unknown")[:50]
        print(f"\n  Re-embedding: {title}")

        try:
            # Generate new embedding
            new_embedding = embedder.embed_text(content)

            # Check if valid
            if new_embedding is None or all(v == 0 for v in new_embedding):
                print(f"    FAILED - still got zero embedding")
                continue

            # Delete old document
            store.collection.delete(ids=[doc_id])

            # Re-add with new embedding
            store.collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata],
                embeddings=[new_embedding]
            )

            print(f"    OK - new embedding has {sum(1 for v in new_embedding if v != 0)} non-zero values")
            fixed += 1

        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\n{'=' * 60}")
    print(f"DONE: Fixed {fixed}/{len(bad_ids)} documents")
    print(f"{'=' * 60}")

    if fixed < len(bad_ids):
        print("\nSome documents could not be fixed. Check the errors above.")
    else:
        print("\nAll documents fixed! You can now re-sync to Pinecone.")

if __name__ == "__main__":
    fix_zero_vectors()