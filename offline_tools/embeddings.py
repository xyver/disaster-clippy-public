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
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Either:\n"
                "  1. Set OPENAI_API_KEY in your .env file, or\n"
                "  2. Set EMBEDDING_MODE=local and install sentence-transformers:\n"
                "     pip install sentence-transformers"
            )
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
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
            # Check if we can fall back to OpenAI
            if os.getenv("OPENAI_API_KEY"):
                print("sentence-transformers not installed. Run: pip install sentence-transformers")
                print("Falling back to OpenAI embeddings...")
                self._init_openai("text-embedding-3-small")
                self.mode = "openai"
            else:
                raise ValueError(
                    "EMBEDDING_MODE=local but sentence-transformers is not installed.\n"
                    "Install it with: pip install sentence-transformers\n"
                    "Or set OPENAI_API_KEY to use OpenAI embeddings instead."
                )

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
            # OpenAI API - use chunking for long texts
            return self._embed_with_chunking(text)

    def _embed_with_chunking(self, text: str, max_depth: int = 3) -> List[float]:
        """
        Embed text, chunking if it exceeds token limit.

        If the text is too long for the model's context window, splits it in half
        and averages the embeddings. Recursively splits up to max_depth times.

        Args:
            text: Text to embed
            max_depth: Maximum recursion depth for splitting (default 3 = up to 8 chunks)

        Returns:
            Embedding vector (averaged if chunked)
        """
        # Truncate to reasonable max (prevents extremely long texts)
        MAX_CHARS = 32000  # ~8k tokens max, gives room for dense text
        text = text[:MAX_CHARS]

        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            error_str = str(e).lower()

            # Check if it's a token limit error
            if "token" in error_str or "maximum context length" in error_str:
                if max_depth <= 0:
                    # Give up and use very aggressive truncation
                    print(f"Max chunk depth reached, truncating aggressively")
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=text[:4000]  # ~1k tokens, should always work
                    )
                    return response.data[0].embedding

                # Split text in half and embed each part
                mid = len(text) // 2

                # Try to split at a sentence boundary near the middle
                # Look for period, newline, or space within 500 chars of middle
                best_split = mid
                for offset in range(0, min(500, mid)):
                    # Check both directions from middle
                    for pos in [mid + offset, mid - offset]:
                        if 0 < pos < len(text):
                            if text[pos] in '.!?\n':
                                best_split = pos + 1
                                break
                    else:
                        continue
                    break

                first_half = text[:best_split].strip()
                second_half = text[best_split:].strip()

                print(f"Text too long ({len(text)} chars), splitting into chunks...")

                # Recursively embed each half
                emb1 = self._embed_with_chunking(first_half, max_depth - 1)
                emb2 = self._embed_with_chunking(second_half, max_depth - 1)

                # Average the embeddings
                averaged = [(a + b) / 2 for a, b in zip(emb1, emb2)]
                return averaged
            else:
                # Not a token limit error, re-raise
                raise

    def embed_batch(self, texts: List[str], batch_size: int = 50,
                    progress_callback=None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per API call
            progress_callback: Optional function(current, total, message) for progress

        Returns:
            List of embedding vectors
        """
        total = len(texts)

        if self.mode == "local" and self._local_model:
            # Local model - can process larger batches, no API costs
            MAX_CHARS = 50000
            truncated = [t[:MAX_CHARS] for t in texts]
            if progress_callback:
                progress_callback(0, total, "Computing embeddings (local model)...")
            embeddings = self._local_model.encode(truncated, convert_to_numpy=True, show_progress_bar=True)
            if progress_callback:
                progress_callback(total, total, "Embeddings complete")
            return [e.tolist() for e in embeddings]

        # OpenAI API path
        all_embeddings = []
        # Allow longer texts - chunking handles overflow
        MAX_CHARS = 32000

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_truncated = [t[:MAX_CHARS] for t in batch]

            if progress_callback:
                progress_callback(i, total, f"Computing embeddings ({i}/{total})...")

            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch_truncated
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                # If batch fails, try one at a time with chunking support
                print(f"Batch embedding failed, trying individually with chunking: {e}")
                for text in batch_truncated:
                    try:
                        embedding = self._embed_with_chunking(text)
                        all_embeddings.append(embedding)
                    except Exception as e2:
                        print(f"Failed to embed text after chunking: {e2}")
                        dim = 1536 if "small" in self.model or "ada" in self.model else 3072
                        all_embeddings.append([0.0] * dim)

        if progress_callback:
            progress_callback(total, total, "Embeddings complete")

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
