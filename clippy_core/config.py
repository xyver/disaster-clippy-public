"""
ClippyConfig - Runtime configuration for clippy_core.

This is the main configuration interface for external projects.
Provides a clean dataclass interface instead of reading from local_settings.json.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os


@dataclass
class ClippyConfig:
    """
    Runtime configuration for clippy_core.

    This can be:
    1. Instantiated directly with values
    2. Loaded from environment variables
    3. Created from a dict (e.g., from JSON config)

    Example:
        # Direct instantiation
        config = ClippyConfig(
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            embedding_dimension=768
        )

        # From environment
        config = ClippyConfig.from_env()

        # From dict
        config = ClippyConfig.from_dict({"llm_provider": "anthropic"})
    """

    # Vector DB settings
    vector_db_mode: str = "local"  # "local", "pinecone", "pgvector"
    backup_path: str = "./backups"
    embedding_dimension: int = 768  # 384, 768, 1024, or 1536

    # LLM settings
    llm_provider: str = "openai"  # "openai", "anthropic", "local"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    # Embedding settings
    embedding_provider: str = "openai"  # "openai", "local"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "all-mpnet-base-v2"

    # Connection settings
    offline_mode: str = "hybrid"  # "online_only", "hybrid", "offline_only"

    # API keys (optional - can also come from env)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None

    # pgvector settings (for Supabase integration)
    pgvector_connection_string: Optional[str] = None
    pgvector_table_name: str = "source_vectors"

    @classmethod
    def from_env(cls) -> "ClippyConfig":
        """
        Create config from environment variables.

        Environment variables (all optional):
            VECTOR_DB_MODE: local, pinecone, pgvector
            BACKUP_PATH: Path to backup folder
            EMBEDDING_DIMENSION: 384, 768, 1024, or 1536
            LLM_PROVIDER: openai, anthropic, local
            LLM_MODEL: Model name (e.g., gpt-4o-mini)
            EMBEDDING_MODE: openai or local
            EMBEDDING_MODEL: Model name for embeddings
            OFFLINE_MODE: online_only, hybrid, offline_only
            OPENAI_API_KEY: OpenAI API key
            ANTHROPIC_API_KEY: Anthropic API key
            PINECONE_API_KEY: Pinecone API key
            PGVECTOR_CONNECTION_STRING: PostgreSQL connection string
            PGVECTOR_TABLE_NAME: Table name for vectors
        """
        return cls(
            vector_db_mode=os.getenv("VECTOR_DB_MODE", "local"),
            backup_path=os.getenv("BACKUP_PATH", "./backups"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_model=os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
            embedding_provider=os.getenv("EMBEDDING_MODE", "openai"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            local_embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", "all-mpnet-base-v2"),
            offline_mode=os.getenv("OFFLINE_MODE", "hybrid"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            pgvector_connection_string=os.getenv("PGVECTOR_CONNECTION_STRING", os.getenv("SUPABASE_DB_URL")),
            pgvector_table_name=os.getenv("PGVECTOR_TABLE_NAME", "source_vectors"),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClippyConfig":
        """Create config from a dictionary."""
        # Start with defaults
        config = cls()

        # Override with provided values
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (excludes API keys)."""
        return {
            "vector_db_mode": self.vector_db_mode,
            "backup_path": self.backup_path,
            "embedding_dimension": self.embedding_dimension,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "local_embedding_model": self.local_embedding_model,
            "offline_mode": self.offline_mode,
            "pgvector_table_name": self.pgvector_table_name,
            # Note: API keys intentionally excluded
        }

    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from config or environment."""
        return self.openai_api_key or os.getenv("OPENAI_API_KEY")

    def get_anthropic_api_key(self) -> Optional[str]:
        """Get Anthropic API key from config or environment."""
        return self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

    def get_pinecone_api_key(self) -> Optional[str]:
        """Get Pinecone API key from config or environment."""
        return self.pinecone_api_key or os.getenv("PINECONE_API_KEY")

    def is_offline_only(self) -> bool:
        """Check if running in offline-only mode."""
        return self.offline_mode == "offline_only"

    def is_online_only(self) -> bool:
        """Check if running in online-only mode."""
        return self.offline_mode == "online_only"

    def should_try_online(self) -> bool:
        """Check if online APIs should be attempted."""
        return self.offline_mode != "offline_only"
