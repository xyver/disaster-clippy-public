"""
ClippyConfig - runtime configuration for clippy_core.

Create it directly, from environment variables, or from a dict (e.g. the
``config:`` block of a sources.yaml manifest).

    config = ClippyConfig(llm_provider="anthropic", index_path="rules.sqlite")
    config = ClippyConfig.from_env()
    config = ClippyConfig.from_dict({"search_mode": "keyword"})

Provider values
---------------
llm_provider:        "auto" | "anthropic" | "openai" | "none"
    "auto" picks Anthropic if ANTHROPIC_API_KEY is set, then OpenAI, then "none".
    "none" returns an extractive answer (the cited passages) with no LLM call.

embedding_provider:  "auto" | "openai" | "local" | "hash"
    "auto" picks OpenAI if OPENAI_API_KEY is set, then a local
    sentence-transformers model if installed, then "hash".
    "hash" is dependency-free and deterministic; fine for tests and demos,
    weak for real semantic search (keyword search still works fully).

An index remembers which embedder built it. Queries always use the same one.
"""

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional
import os


@dataclass
class ClippyConfig:
    # ---- storage ---------------------------------------------------------
    vector_db_mode: str = "sqlite"          # "sqlite" | "pgvector"
    index_path: str = "./index.sqlite"      # used by the sqlite store

    # ---- LLM -------------------------------------------------------------
    llm_provider: str = "auto"
    llm_model: str = ""                      # empty = provider default
    llm_temperature: float = 0.2             # low: rules and references reward precision
    llm_max_tokens: int = 1200

    # ---- embeddings ------------------------------------------------------
    embedding_provider: str = "auto"
    embedding_model: str = ""                # empty = provider default

    # ---- search and context ---------------------------------------------
    search_mode: str = "hybrid"              # "hybrid" | "keyword" | "semantic"
    n_results: int = 8
    max_context_chars: int = 12000           # total passage text sent to the LLM
    per_result_chars: int = 2500             # max chars per passage
    prompt_path: Optional[str] = None        # system prompt file; None = built-in default

    # ---- API keys (optional; env vars are used when these are empty) ----
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # ---- pgvector (optional backend) -------------------------------------
    pgvector_connection_string: Optional[str] = None
    pgvector_table_name: str = "source_vectors"

    # ---------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "ClippyConfig":
        env = os.getenv
        return cls(
            vector_db_mode=env("VECTOR_DB_MODE", "sqlite"),
            index_path=env("CLIPPY_INDEX", "./index.sqlite"),
            llm_provider=env("LLM_PROVIDER", "auto"),
            llm_model=env("LLM_MODEL", ""),
            llm_temperature=float(env("LLM_TEMPERATURE", "0.2")),
            llm_max_tokens=int(env("LLM_MAX_TOKENS", "1200")),
            embedding_provider=env("EMBEDDING_PROVIDER", env("EMBEDDING_MODE", "auto")),
            embedding_model=env("EMBEDDING_MODEL", ""),
            search_mode=env("SEARCH_MODE", "hybrid"),
            n_results=int(env("N_RESULTS", "8")),
            prompt_path=env("CLIPPY_PROMPT") or None,
            openai_api_key=env("OPENAI_API_KEY"),
            anthropic_api_key=env("ANTHROPIC_API_KEY"),
            pgvector_connection_string=env("PGVECTOR_CONNECTION_STRING", env("SUPABASE_DB_URL")),
            pgvector_table_name=env("PGVECTOR_TABLE_NAME", "source_vectors"),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], base: Optional["ClippyConfig"] = None) -> "ClippyConfig":
        """Overlay ``data`` onto ``base`` (or defaults). Unknown keys raise, to catch typos."""
        config = base or cls()
        valid = {f.name for f in fields(cls)}
        for key, value in (data or {}).items():
            if key not in valid:
                raise ValueError(f"Unknown config key: {key!r}. Valid keys: {sorted(valid)}")
            setattr(config, key, value)
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Config as a dict, API keys excluded."""
        d = asdict(self)
        d.pop("openai_api_key", None)
        d.pop("anthropic_api_key", None)
        d.pop("pgvector_connection_string", None)
        return d

    # ---- key helpers -----------------------------------------------------

    def get_openai_api_key(self) -> Optional[str]:
        return self.openai_api_key or os.getenv("OPENAI_API_KEY")

    def get_anthropic_api_key(self) -> Optional[str]:
        return self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

    def resolved_llm_provider(self) -> str:
        p = (self.llm_provider or "auto").lower()
        if p != "auto":
            return p
        if self.get_anthropic_api_key():
            return "anthropic"
        if self.get_openai_api_key():
            return "openai"
        return "none"
