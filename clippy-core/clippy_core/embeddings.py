"""
Embedding providers for clippy_core.

Every embedder has a ``name`` (stored in the index so queries use the same
model), a ``dimension``, and ``embed(texts) -> np.ndarray`` returning
L2-normalised float32 vectors of shape (len(texts), dimension).

Providers
---------
openai  OpenAI embeddings API (default model text-embedding-3-small). Needs OPENAI_API_KEY.
local   sentence-transformers model (default all-MiniLM-L6-v2). Fully offline.
hash    Feature hashing of words and word pairs. No dependencies, deterministic.
        Captures word overlap, not meaning. For tests, demos and air-gapped
        fallback; keyword search does the heavy lifting in that case.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Sequence

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class Embedder:
    name: str = "base"
    dimension: int = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashEmbedder(Embedder):
    def __init__(self, dimension: int = 512):
        self.dimension = dimension
        self.name = f"hash-{dimension}"

    def _vector(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype=np.float32)
        words = _WORD.findall(text.lower())
        features = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
        for feat in features:
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            v[h % self.dimension] += 1.0 if (h >> 63) == 0 else -1.0
        return v

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return _normalise(np.stack([self._vector(t) for t in texts]))


class OpenAIEmbedder(Embedder):
    DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072,
                  "text-embedding-ada-002": 1536}

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None,
                 batch_size: int = 128):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.dimension = self.DIMENSIONS.get(model, 1536)
        self.name = f"openai:{model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[:30000] or " " for t in texts[i:i + self.batch_size]]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in resp.data)
        if not out:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return _normalise(np.array(out))


class LocalEmbedder(Embedder):
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        import os
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        cache_folder = os.getenv("CLIPPY_MODEL_CACHE")
        self.model = SentenceTransformer(model, cache_folder=cache_folder)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.name = f"local:{model}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return _normalise(self.model.encode(list(texts), batch_size=32, show_progress_bar=False))


def get_embedder(provider: str = "auto", model: str = "", api_key: Optional[str] = None) -> Embedder:
    """Create an embedder. See module docstring for provider values."""
    import os

    provider = (provider or "auto").lower()
    if provider == "auto":
        if api_key or os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            try:
                import sentence_transformers  # noqa: F401
                provider = "local"
            except ImportError:
                provider = "hash"

    if provider == "openai":
        return OpenAIEmbedder(model or "text-embedding-3-small", api_key=api_key)
    if provider == "local":
        return LocalEmbedder(model or "all-MiniLM-L6-v2")
    if provider == "hash":
        return HashEmbedder(int(model) if model and model.isdigit() else 512)
    raise ValueError(f"Unknown embedding provider: {provider!r}")


def embedder_from_name(name: str, api_key: Optional[str] = None) -> Embedder:
    """Recreate the embedder that built an index, from its stored name."""
    if name.startswith("hash-"):
        return HashEmbedder(int(name.split("-", 1)[1]))
    if name.startswith("openai:"):
        return OpenAIEmbedder(name.split(":", 1)[1], api_key=api_key)
    if name.startswith("local:"):
        return LocalEmbedder(name.split(":", 1)[1])
    raise ValueError(f"Unrecognised embedder name stored in index: {name!r}")
