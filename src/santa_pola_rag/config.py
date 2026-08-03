from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    deepseek_api_key: str
    openrouter_api_key: str

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "santapola"
    postgres_password: str = "santapola"
    postgres_db: str = "santapola"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_api_key: str | None = None

    minio_endpoint_url: str = "http://localhost:9000"
    minio_access_key: str = "santapola"
    minio_secret_key: str = "santapola123"
    minio_bucket: str = "santa-pola-pdfs"

    otel_exporter_otlp_endpoint: str = "http://localhost:4318"
    otel_service_name: str = "santa-pola-rag"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
