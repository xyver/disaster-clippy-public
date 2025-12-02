"""
Embedding service for converting text to vectors.
Supports both OpenAI API and local models (sentence-transformers).
"""

from typing import List, Optional
import os
from dotenv import load_dotenv

load_dotenv()


class EmbeddingService:
    """
    Handles text-to-vector embedding.

    Supports:
    - OpenAI API (default): text-embedding-3-small, text-embedding-3-large
    - Local models (free): all-MiniLM-L6-v2, all-mpnet-base-v2, etc.

    Set EMBEDDING_MODE=local in .env to use local models.
    """

    def __init__(self, model: Optional[str] = None):
        """
        Args:
            model: Model to use. If None, auto-detects based on EMBEDDING_MODE env var.
                   OpenAI: "text-embedding-3-small", "text-embedding-3-large"
                   Local: "all-MiniLM-L6-v2", "all-mpnet-base-v2", etc.
        """
        self.mode = os.getenv("EMBEDDING_MODE", "openai").lower()

        # Check for model override from env
        env_model = os.getenv("EMBEDDING_MODEL", "")

        if self.mode == "local":
            # Use sentence-transformers for local embeddings
            # all-mpnet-base-v2 is recommended for best quality
            # all-MiniLM-L6-v2 is faster but lower quality
            default_local = "all-mpnet-base-v2"
            self._init_local(model or env_model or default_local)
        else:
            # Use OpenAI API
            self._init_openai(model or env_model or "text-embedding-3-small")

    def _init_openai(self, model: str):
        """Initialize OpenAI embedding client"""
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self._local_model = None
        print(f"Using OpenAI embeddings: {model}")

    def _init_local(self, model: str):
        """Initialize local sentence-transformers model"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading local embedding model: {model}...")
            self._local_model = SentenceTransformer(model)
            self.model = model
            self.client = None
            print(f"Local model loaded: {model} (dimension: {self._local_model.get_sentence_embedding_dimension()})")
        except ImportError:
            print("sentence-transformers not installed. Run: pip install sentence-transformers")
            print("Falling back to OpenAI embeddings...")
            self._init_openai("text-embedding-3-small")
            self.mode = "openai"

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats (embedding vector)
        """
        if self.mode == "local" and self._local_model:
            # Local model - no token limit issues, but truncate for memory
            text = text[:50000]
            embedding = self._local_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            # OpenAI API - truncate to safe limit
            text = text[:20000]
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding

    def embed_batch(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call

        Returns:
            List of embedding vectors
        """
        if self.mode == "local" and self._local_model:
            # Local model - can process larger batches, no API costs
            MAX_CHARS = 50000
            truncated = [t[:MAX_CHARS] for t in texts]
            embeddings = self._local_model.encode(truncated, convert_to_numpy=True, show_progress_bar=True)
            return [e.tolist() for e in embeddings]

        # OpenAI API path
        all_embeddings = []
        MAX_CHARS = 20000

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [t[:MAX_CHARS] for t in batch]

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                # If batch fails, try one at a time
                print(f"Batch embedding failed, trying individually: {e}")
                for text in batch:
                    try:
                        response = self.client.embeddings.create(
                            model=self.model,
                            input=text[:MAX_CHARS]
                        )
                        all_embeddings.append(response.data[0].embedding)
                    except Exception as e2:
                        print(f"Failed to embed text: {e2}")
                        dim = 1536 if "small" in self.model or "ada" in self.model else 3072
                        all_embeddings.append([0.0] * dim)

        return all_embeddings

    def get_dimension(self) -> int:
        """Get the embedding dimension for the current model"""
        if self.mode == "local" and self._local_model:
            return self._local_model.get_sentence_embedding_dimension()
        elif "large" in self.model:
            return 3072
        else:
            return 1536


# Quick test
if __name__ == "__main__":
    service = EmbeddingService()

    # Test single embedding
    embedding = service.embed_text("How to build a water filter")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
