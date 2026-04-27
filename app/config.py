from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Customer Support Agent"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    xai_api_key: str | None = Field(default=None, alias="XAI_API_KEY")
    xai_base_url: str = Field(default="https://api.x.ai/v1", alias="XAI_BASE_URL")
    xai_model: str = Field(default="grok-4.20-reasoning", alias="XAI_MODEL")
    xai_temperature: float = Field(default=0.1, alias="XAI_TEMPERATURE")
    xai_max_output_tokens: int = Field(default=900, alias="XAI_MAX_OUTPUT_TOKENS")
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_NAME",
    )
    hf_token: str | None = Field(default=None, alias="HF_TOKEN")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    chunk_size: int = Field(default=1200, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    retrieval_top_k: int = Field(default=4, alias="RETRIEVAL_TOP_K")
    min_retrieval_confidence: float = Field(
        default=0.45,
        alias="MIN_RETRIEVAL_CONFIDENCE",
    )
    min_llm_confidence: float = Field(default=0.60, alias="MIN_LLM_CONFIDENCE")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    uploads_dir: Path = Field(default=Path("data/uploads"), alias="UPLOADS_DIR")
    chroma_dir: Path = Field(default=Path("data/chroma"), alias="CHROMA_DIR")
    escalation_db_path: Path = Field(
        default=Path("data/escalations.sqlite3"),
        alias="ESCALATION_DB_PATH",
    )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.escalation_db_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
