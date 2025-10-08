"""
Centralized configuration for RAG evaluation pipeline.

This module provides shared configuration constants used across:
- langchain_eval_foundations_e2e.py
- langchain_eval_golden_testset.py
- upload_golden_testset_to_phoenix.py
- langchain_eval_experiments.py

Environment variables can override defaults via .env file.
"""

import os
from dataclasses import dataclass

# =============================================================================
# Phoenix Observability Settings
# =============================================================================

PHOENIX_ENDPOINT = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
"""Phoenix UI and API endpoint"""

PHOENIX_OTLP_ENDPOINT = os.getenv("PHOENIX_OTLP_ENDPOINT", "http://localhost:4317")
"""Phoenix OpenTelemetry collector endpoint"""

PHOENIX_COLLECTOR_ENDPOINT = os.getenv(
    "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
)
"""Phoenix collector endpoint (legacy compatibility)"""

PHOENIX_API_KEY = os.getenv("PHOENIX_API_KEY")
"""Phoenix API key (optional, for cloud Phoenix)"""


# =============================================================================
# Database Settings
# =============================================================================

POSTGRES_USER = os.getenv("POSTGRES_USER", "langchain")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "langchain")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "6024")
POSTGRES_DB = os.getenv("POSTGRES_DB", "langchain")

VECTOR_SIZE = 1536
"""Dimension size for OpenAI text-embedding-3-small"""

# Table names for vector stores
BASELINE_TABLE = "mixed_baseline_documents"
"""Table for baseline vector search with standard chunking"""

SEMANTIC_TABLE = "mixed_semantic_documents"
"""Table for semantic chunking vector search"""


# =============================================================================
# Dataset Settings
# =============================================================================

GOLDEN_TESTSET_NAME = "mixed_golden_testset_phoenix"
"""
Canonical name for the golden testset dataset in Phoenix.

IMPORTANT: This name must be consistent across:
- upload_golden_testset_to_phoenix.py (upload)
- langchain_eval_golden_testset.py (generation + upload)
- langchain_eval_experiments.py (retrieval for experiments)

Note: The actual Phoenix dataset name may have a version suffix
(e.g., "golden_testset_vexternal_20251008_042303") due to
PhoenixIntegration's versioning system.
"""

GOLDEN_TESTSET_SIZE = int(os.getenv("GOLDEN_TESTSET_SIZE", "10"))
"""Number of examples to generate in RAGAS golden test set"""


# =============================================================================
# Model Settings (ENFORCED - see CLAUDE.md)
# =============================================================================

LLM_MODEL = "gpt-4.1-mini"
"""
LLM model for RAG and evaluation.
CRITICAL: Only gpt-4.1-mini is permitted per CLAUDE.md requirements.
"""

EMBEDDING_MODEL = "text-embedding-3-small"
"""
Embedding model for vector search.
CRITICAL: Only text-embedding-3-small is permitted per CLAUDE.md requirements.
"""

COHERE_RERANK_MODEL = "rerank-english-v3.0"
"""Cohere reranking model for contextual compression retrieval"""


# =============================================================================
# Helper Functions
# =============================================================================


def get_postgres_async_url() -> str:
    """Get PostgreSQL async connection string for asyncpg."""
    return (
        f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


def get_postgres_sync_url() -> str:
    """Get PostgreSQL sync connection string for psycopg2."""
    return (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


@dataclass
class PhoenixSettings:
    """Phoenix configuration settings."""

    endpoint: str = PHOENIX_ENDPOINT
    otlp_endpoint: str = PHOENIX_OTLP_ENDPOINT
    api_key: str | None = PHOENIX_API_KEY

    def __post_init__(self):
        """Validate Phoenix settings."""
        if not self.endpoint:
            raise ValueError("PHOENIX_ENDPOINT must be set")
        if not self.otlp_endpoint:
            raise ValueError("PHOENIX_OTLP_ENDPOINT must be set")


@dataclass
class DatabaseSettings:
    """Database configuration settings."""

    user: str = POSTGRES_USER
    password: str = POSTGRES_PASSWORD
    host: str = POSTGRES_HOST
    port: str = POSTGRES_PORT
    database: str = POSTGRES_DB
    baseline_table: str = BASELINE_TABLE
    semantic_table: str = SEMANTIC_TABLE
    vector_size: int = VECTOR_SIZE

    @property
    def async_url(self) -> str:
        """Get async connection URL."""
        return get_postgres_async_url()

    @property
    def sync_url(self) -> str:
        """Get sync connection URL."""
        return get_postgres_sync_url()


@dataclass
class ModelSettings:
    """Model configuration settings."""

    llm_model: str = LLM_MODEL
    embedding_model: str = EMBEDDING_MODEL
    cohere_rerank_model: str = COHERE_RERANK_MODEL

    def __post_init__(self):
        """Validate model settings against CLAUDE.md requirements."""
        if self.llm_model != "gpt-4.1-mini":
            raise ValueError(
                f"LLM model must be 'gpt-4.1-mini' per CLAUDE.md requirements. "
                f"Got: {self.llm_model}"
            )
        if self.embedding_model != "text-embedding-3-small":
            raise ValueError(
                f"Embedding model must be 'text-embedding-3-small' "
                f"per CLAUDE.md requirements. Got: {self.embedding_model}"
            )


# =============================================================================
# Convenience Functions
# =============================================================================


def get_all_settings() -> dict:
    """Get all configuration settings as a dictionary."""
    return {
        "phoenix": PhoenixSettings().__dict__,
        "database": DatabaseSettings().__dict__,
        "models": ModelSettings().__dict__,
        "golden_testset": {
            "name": GOLDEN_TESTSET_NAME,
            "size": GOLDEN_TESTSET_SIZE,
        },
    }


def validate_config() -> bool:
    """Validate all configuration settings."""
    try:
        PhoenixSettings()
        DatabaseSettings()
        ModelSettings()
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


if __name__ == "__main__":
    """Print configuration when run directly."""
    import json

    print("=" * 60)
    print("RAG Evaluation Pipeline Configuration")
    print("=" * 60)
    print(json.dumps(get_all_settings(), indent=2))
    print("=" * 60)
    print(f"✅ Configuration valid: {validate_config()}")
