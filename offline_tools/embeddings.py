"""
Embedding service for converting text to vectors.
Supports both OpenAI API and local models (sentence-transformers).

Fallback chain based on offline_mode setting:
- online_only: Try OpenAI first, warn before falling back to local
- hybrid: Try OpenAI, auto-fallback to local with notice
- offline_only: Skip API entirely, use local chain

Local fallback order:
1. Portable model (BACKUP_PATH/models/embeddings/)
2. HuggingFace cache (~/.cache/huggingface/)
3. Auto-download from HuggingFace
"""

from typing import List, Optional, Callable
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class EmbeddingFallbackNotice:
    """Notices shown during embedding fallback"""
    API_UNAVAILABLE = "Cloud embedding API unavailable. Using local model."
    USING_PORTABLE = "Using portable embedding model from backup folder."
    USING_CACHE = "Using cached embedding model from HuggingFace cache."
    DOWNLOADING = "Downloading embedding model (this only happens once)..."
    NO_MODEL = "No embedding model available. Semantic search disabled."


class EmbeddingService:
    """
    Handles text-to-vector embedding.

    Supports:
    - OpenAI API (default): text-embedding-3-small, text-embedding-3-large
    - Local models (free): all-MiniLM-L6-v2, all-mpnet-base-v2, etc.

    Set EMBEDDING_MODE=local in .env to use local models.

    Fallback behavior depends on offline_mode setting in local_settings.json.
    """

    # Known local model names (sentence-transformers)
    # These map to model IDs in the model_registry
    LOCAL_MODEL_NAMES = [
        "all-MiniLM-L6-v2",          # 384-dim, fast/lightweight
        "all-mpnet-base-v2",          # 768-dim, recommended balance
        "intfloat-e5-large-v2",       # 1024-dim, high quality
        "intfloat/e5-large-v2",       # HuggingFace format
        "multi-qa-mpnet-base-dot-v1",
        "paraphrase-multilingual-MiniLM-L12-v2",
    ]

    def __init__(self, model: Optional[str] = None, fallback_callback: Optional[Callable[[str], None]] = None):
        """
        Args:
            model: Model to use. If None, auto-detects based on EMBEDDING_MODE env var.
                   OpenAI: "text-embedding-3-small", "text-embedding-3-large"
                   Local: "all-MiniLM-L6-v2", "all-mpnet-base-v2", etc.
            fallback_callback: Optional callback to notify about fallback events
        """
        self.fallback_callback = fallback_callback
        self._local_model = None
        self.client = None
        self.fallback_notices = []  # Track notices for the caller

        # Get offline mode setting
        self.offline_mode = self._get_offline_mode()

        # Check for model override from env
        env_model = os.getenv("EMBEDDING_MODEL", "")
        self.mode = os.getenv("EMBEDDING_MODE", "openai").lower()

        # Check if explicitly requesting a local model
        requested_model = model or env_model
        is_local_model = requested_model and any(
            requested_model.lower() == m.lower() for m in self.LOCAL_MODEL_NAMES
        )

        # Decide initialization based on offline_mode and model type
        if is_local_model:
            # Explicitly requested local model - use it regardless of offline_mode
            print(f"[EmbeddingService] Using local model (explicitly requested): {requested_model}")
            self._init_local_with_fallback(requested_model)
        elif self.offline_mode == "offline_only":
            # Skip API entirely, go straight to local
            # Only use requested model if it's a valid local model, otherwise use default
            default_local = "all-mpnet-base-v2"
            local_model = default_local
            if requested_model and requested_model in self.LOCAL_MODEL_NAMES:
                local_model = requested_model
            self._init_local_with_fallback(local_model)
        elif self.mode == "local":
            # User explicitly wants local mode
            # Only use requested model if it's a valid local model, otherwise use default
            default_local = "all-mpnet-base-v2"
            local_model = default_local
            if requested_model and requested_model in self.LOCAL_MODEL_NAMES:
                local_model = requested_model
            self._init_local_with_fallback(local_model)
        else:
            # Try OpenAI first (online_only or hybrid mode)
            try:
                self._init_openai(model or env_model or "text-embedding-3-small")
            except ValueError as e:
                # No API key - decide based on mode
                if self.offline_mode == "online_only":
                    # Warn but still try local
                    self._notify(EmbeddingFallbackNotice.API_UNAVAILABLE)
                    self._init_local_with_fallback("all-mpnet-base-v2")
                else:
                    # Hybrid mode - silently fall back to local
                    self._init_local_with_fallback("all-mpnet-base-v2")

    def _get_offline_mode(self) -> str:
        """Get the offline_mode setting from local_config"""
        try:
            from admin.local_config import get_local_config
            return get_local_config().get_offline_mode()
        except Exception:
            return "hybrid"  # Default to hybrid if config not available

    def _get_portable_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to portable model in BACKUP_PATH/models/embeddings/"""
        try:
            from admin.local_config import get_local_config
            backup_folder = get_local_config().get_backup_folder()
            if backup_folder:
                model_path = Path(backup_folder) / "models" / "embeddings" / model_name
                if model_path.exists():
                    # Check for required files
                    if (model_path / "pytorch_model.bin").exists() or \
                       (model_path / "model.safetensors").exists():
                        return model_path
        except Exception:
            pass
        return None

    def _notify(self, message: str):
        """Send a fallback notification"""
        self.fallback_notices.append(message)
        print(f"[Embedding] {message}")
        if self.fallback_callback:
            self.fallback_callback(message)

    def _init_local_with_fallback(self, model: str):
        """
        Initialize local model with fallback chain:
        1. Portable model (BACKUP_PATH/models/embeddings/)
        2. HuggingFace cache
        3. Auto-download from HuggingFace
        """
        # Try portable model first
        portable_path = self._get_portable_model_path(model)
        if portable_path:
            try:
                self._init_local(str(portable_path))
                self._notify(EmbeddingFallbackNotice.USING_PORTABLE)
                return
            except Exception as e:
                print(f"Failed to load portable model: {e}")

        # Try loading from default location (HuggingFace cache or download)
        try:
            self._init_local(model)
            # Check if it was cached or downloaded
            cache_path = Path.home() / ".cache" / "huggingface"
            if cache_path.exists():
                self._notify(EmbeddingFallbackNotice.USING_CACHE)
        except ImportError:
            # sentence-transformers not installed
            # Try OpenAI as final fallback if available
            if os.getenv("OPENAI_API_KEY"):
                print("sentence-transformers not installed, falling back to OpenAI")
                self._init_openai("text-embedding-3-small")
                self.mode = "openai"
            else:
                self._notify(EmbeddingFallbackNotice.NO_MODEL)
                self.mode = "disabled"
                self.model = None
        except Exception as e:
            print(f"Failed to load local model: {e}")
            self._notify(EmbeddingFallbackNotice.NO_MODEL)
            self.mode = "disabled"
            self.model = None

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
        import httpx

        # Custom retry configuration with longer delays to avoid 429 rate limit spam
        # Default OpenAI SDK retries too fast (0.025s), we want minimum 0.25s between retries
        custom_timeout = httpx.Timeout(60.0, connect=10.0)

        self.client = OpenAI(
            api_key=api_key,
            max_retries=3,
            timeout=custom_timeout
        )
        # Store min retry delay for manual backoff in rate limit handling
        self._min_retry_delay = 0.1

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
            self.mode = "local"  # Important: set mode to local so embed_text uses local model
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

    def embed_text(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats (embedding vector), or None if embeddings disabled
        """
        if self.mode == "disabled":
            return None

        if self.mode == "local" and self._local_model:
            # Local model - no token limit issues, but truncate for memory
            text = text[:50000]
            embedding = self._local_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            # OpenAI API - use chunking for long texts
            return self._embed_with_chunking(text)

    def _embed_with_chunking(self, text: str, max_depth: int = 3, _retry_count: int = 0) -> List[float]:
        """
        Embed text, chunking if it exceeds token limit.

        If the text is too long for the model's context window, splits it in half
        and averages the embeddings. Recursively splits up to max_depth times.

        Args:
            text: Text to embed
            max_depth: Maximum recursion depth for splitting (default 3 = up to 8 chunks)
            _retry_count: Internal retry counter for rate limit handling

        Returns:
            Embedding vector (averaged if chunked)
        """
        import time

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

            # Handle rate limit errors with longer delay (429 Too Many Requests)
            if "429" in str(e) or "rate" in error_str:
                if _retry_count < 5:
                    delay = self._min_retry_delay * (2 ** _retry_count)  # Exponential backoff starting at 0.25s
                    print(f"Rate limited, waiting {delay:.2f}s before retry {_retry_count + 1}/5...")
                    time.sleep(delay)
                    return self._embed_with_chunking(text, max_depth, _retry_count + 1)
                else:
                    raise Exception(f"Rate limit exceeded after 5 retries: {e}")

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

    def is_available(self) -> bool:
        """Check if embedding service is available"""
        return self.mode != "disabled" and (self._local_model is not None or self.client is not None)

    def get_status(self) -> dict:
        """Get current embedding service status"""
        gpu_count = self._get_gpu_count()
        gpu_info = self._get_gpu_info()

        return {
            "available": self.is_available(),
            "mode": self.mode,
            "model": self.model if hasattr(self, 'model') and self.model else None,
            "dimension": self.get_dimension() if self.is_available() else None,
            "offline_mode": self.offline_mode,
            "notices": self.fallback_notices,
            "gpu_count": gpu_count,
            "gpu_info": gpu_info,
            "multi_gpu_available": gpu_count > 1
        }

    def _get_gpu_info(self) -> list:
        """Get detailed info about available GPUs"""
        gpu_info = []
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info.append({
                        "index": i,
                        "name": props.name,
                        "total_memory_gb": round(props.total_memory / (1024**3), 1),
                        "compute_capability": f"{props.major}.{props.minor}"
                    })
        except Exception as e:
            print(f"[EmbeddingService] Could not get GPU info: {e}")
        return gpu_info

    def embed_batch(self, texts: List[str], batch_size: int = 64,
                    progress_callback=None, multi_gpu: bool = True) -> Optional[List[List[float]]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts per batch (controls VRAM usage, 64 is safe for 6-8GB GPUs)
            progress_callback: Optional function(current, total, message) for progress
            multi_gpu: If True and multiple GPUs available, use multi-process pool

        Returns:
            List of embedding vectors, or None if embeddings disabled
        """
        if self.mode == "disabled":
            return None

        total = len(texts)

        if self.mode == "local" and self._local_model:
            # Local model path
            MAX_CHARS = 50000
            truncated = [t[:MAX_CHARS] for t in texts]

            # Check for multi-GPU
            gpu_count = self._get_gpu_count()

            if multi_gpu and gpu_count > 1 and total > 100:
                # Use multi-process pool for multiple GPUs
                return self._embed_multi_gpu(truncated, batch_size, progress_callback, gpu_count)
            else:
                # Single GPU path with explicit batch_size for VRAM control
                if progress_callback:
                    gpu_info = f"GPU x{gpu_count}" if gpu_count > 0 else "CPU"
                    progress_callback(0, total, f"Computing embeddings ({gpu_info}, batch={batch_size})...")

                embeddings = self._local_model.encode(
                    truncated,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True
                )

                if progress_callback:
                    progress_callback(total, total, "Embeddings complete")
                return [e.tolist() for e in embeddings]

        # OpenAI API path
        import time
        all_embeddings = []
        MAX_CHARS = 32000

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_truncated = [t[:MAX_CHARS] for t in batch]

            if progress_callback:
                progress_callback(i, total, f"Computing embeddings ({i}/{total})...")

            # Retry loop for rate limiting
            retry_count = 0
            max_retries = 5
            while retry_count < max_retries:
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch_truncated
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break  # Success, exit retry loop
                except Exception as e:
                    error_str = str(e).lower()

                    # Handle rate limit errors with exponential backoff starting at 0.25s
                    if "429" in str(e) or "rate" in error_str:
                        retry_count += 1
                        if retry_count < max_retries:
                            delay = self._min_retry_delay * (2 ** (retry_count - 1))  # 0.25, 0.5, 1.0, 2.0s
                            print(f"Rate limited on batch {i//batch_size}, waiting {delay:.2f}s (retry {retry_count}/{max_retries})...")
                            time.sleep(delay)
                            continue
                        else:
                            print(f"Rate limit persisted after {max_retries} retries, falling back to individual...")

                    # If batch fails (rate limit exhausted or other error), try one at a time
                    print(f"Batch embedding failed, trying individually with chunking: {e}")
                    for text in batch_truncated:
                        try:
                            embedding = self._embed_with_chunking(text)
                            all_embeddings.append(embedding)
                        except Exception as e2:
                            print(f"Failed to embed text after chunking: {e2}")
                            dim = 1536 if "small" in self.model or "ada" in self.model else 3072
                            all_embeddings.append([0.0] * dim)
                    break  # Exit retry loop after individual processing

        if progress_callback:
            progress_callback(total, total, "Embeddings complete")

        return all_embeddings

    def _get_gpu_count(self) -> int:
        """Detect number of available CUDA GPUs"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.device_count()
        except ImportError:
            pass
        return 0

    def _embed_multi_gpu(self, texts: List[str], batch_size: int,
                         progress_callback, gpu_count: int) -> List[List[float]]:
        """
        Embed texts using multiple GPUs via sentence-transformers multi-process pool.

        Each GPU runs a separate process with a copy of the model.
        Texts are automatically distributed across GPUs.
        """
        total = len(texts)

        if progress_callback:
            progress_callback(0, total, f"Starting multi-GPU embedding ({gpu_count} GPUs)...")

        try:
            # Start multi-process pool (one process per GPU)
            pool = self._local_model.start_multi_process_pool()

            if progress_callback:
                progress_callback(0, total, f"Encoding on {gpu_count} GPUs (batch={batch_size})...")

            # Encode using all GPUs
            embeddings = self._local_model.encode_multi_process(
                texts,
                pool,
                batch_size=batch_size
            )

            # Stop the pool
            self._local_model.stop_multi_process_pool(pool)

            if progress_callback:
                progress_callback(total, total, f"Multi-GPU embedding complete ({gpu_count} GPUs)")

            return [e.tolist() for e in embeddings]

        except Exception as e:
            print(f"[EmbeddingService] Multi-GPU failed, falling back to single GPU: {e}")
            # Fallback to single GPU
            if progress_callback:
                progress_callback(0, total, "Falling back to single GPU...")

            embeddings = self._local_model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=True
            )

            if progress_callback:
                progress_callback(total, total, "Embeddings complete (single GPU fallback)")

            return [e.tolist() for e in embeddings]

    def get_dimension(self) -> int:
        """Get the embedding dimension for the current model"""
        if self.mode == "disabled":
            return 0
        if self.mode == "local" and self._local_model:
            return self._local_model.get_sentence_embedding_dimension()
        elif hasattr(self, 'model') and self.model and "large" in self.model:
            return 3072
        else:
            return 1536


# Quick test
if __name__ == "__main__":
    import json

    print("=" * 60)
    print("EMBEDDING SERVICE TEST")
    print("=" * 60)

    service = EmbeddingService()

    # Show status with GPU info
    status = service.get_status()
    print(f"\nStatus:")
    print(f"  Mode: {status['mode']}")
    print(f"  Model: {status['model']}")
    print(f"  Dimension: {status['dimension']}")
    print(f"  GPU Count: {status['gpu_count']}")
    print(f"  Multi-GPU Available: {status['multi_gpu_available']}")

    if status['gpu_info']:
        print(f"\nGPU Details:")
        for gpu in status['gpu_info']:
            print(f"  [{gpu['index']}] {gpu['name']} - {gpu['total_memory_gb']}GB VRAM")

    # Test single embedding
    print(f"\nSingle embedding test:")
    embedding = service.embed_text("How to build a water filter")
    print(f"  Dimension: {len(embedding)}")
    print(f"  First 5 values: {embedding[:5]}")

    # Test batch embedding (small batch to verify multi-GPU path)
    if status['gpu_count'] > 1:
        print(f"\nMulti-GPU batch test (200 texts):")
        test_texts = ["Test document " + str(i) for i in range(200)]
        embeddings = service.embed_batch(test_texts, batch_size=64, multi_gpu=True)
        print(f"  Embedded {len(embeddings)} texts")
        print(f"  Each embedding dimension: {len(embeddings[0])}")
    else:
        print(f"\nSingle GPU batch test (50 texts):")
        test_texts = ["Test document " + str(i) for i in range(50)]
        embeddings = service.embed_batch(test_texts, batch_size=32, multi_gpu=False)
        print(f"  Embedded {len(embeddings)} texts")

    print("\n" + "=" * 60)
