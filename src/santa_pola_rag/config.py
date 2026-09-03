from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Main chat/RAG model, any OpenAI-Chat-Completions-compatible provider
    # (Z.ai's coding plan by default; DeepSeek or OpenRouter, including a
    # free OpenRouter model, are drop-in via these three vars alone, no code
    # change needed).
    llm_base_url: str = "https://api.z.ai/api/coding/paas/v4"
    llm_model: str = "glm-4.6"
    llm_api_key: str

    openrouter_api_key: str

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "santapola"
    postgres_password: str = "santapola"
    postgres_db: str = "santapola"
    # "require" for a managed instance (e.g. Neon), unset for local Postgres.
    postgres_sslmode: str | None = None

    # BM25 text search. Credentials only needed for a managed deployment
    # (e.g. Aiven for OpenSearch, which uses basic auth user+password);
    # local docker-compose is unauthenticated.
    opensearch_url: str = "http://localhost:9200"
    opensearch_api_key: str | None = None
    opensearch_user: str | None = None

    minio_endpoint_url: str = "http://localhost:9000"
    minio_access_key: str = "santapola"
    minio_secret_key: str = "santapola123"
    minio_bucket: str = "santa-pola-pdfs"
    # "auto" works for both: MinIO ignores it, Cloudflare R2 requires it.
    minio_region: str = "auto"

    otel_exporter_otlp_endpoint: str = "http://localhost:4318"
    otel_exporter_otlp_headers: str | None = None
    otel_service_name: str = "santa-pola-rag"

    # Visitor cost guards for the public deployment: no proxy-level rate
    # limiting is possible on Streamlit Community Cloud, so the budget is
    # enforced in app code. The session cap bounds one browser tab's
    # session_state lifetime; the daily budget is the hard spend ceiling
    # nobody can reset by clearing cookies.
    max_questions_per_session: int = 20
    daily_query_budget: int = 200

    @property
    def postgres_dsn(self) -> str:
        dsn = (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
        if self.postgres_sslmode:
            dsn += f"?sslmode={self.postgres_sslmode}"
        # Without this, an unreachable host that swallows packets instead of
        # refusing (firewall, wrong DNS) hangs each connection attempt for
        # the OS's ~75s TCP timeout; measured end to end at ~4.5 minutes
        # before the friendly error with the 3 retries in hybrid_search.
        dsn += f"{'&' if '?' in dsn else '?'}connect_timeout=5"
        return dsn


settings = Settings()
