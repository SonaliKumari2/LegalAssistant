from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_api_key: str = ""
    secret_key: str = "dev-secret-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    database_url: str = "sqlite+aiosqlite:///./kanooni_sahayak.db"
    faiss_index_dir: str = "./vector_db"
    upload_dir: str = "./uploads"
    document_ttl_hours: int = 24
    classification_confidence_gap: float = 0.05
    gemini_model: str = "gemini-1.5-flash"
    embedding_model: str = "models/text-embedding-004"
    reranker_model: str = "BAAI/bge-reranker-large"
    retrieval_small_k: int = 5
    retrieval_large_k: int = 5
    rerank_top_k: int = 8
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
