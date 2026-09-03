from santa_pola_rag.config import Settings


def make_settings(**overrides) -> Settings:
    return Settings(llm_api_key="test-key", **overrides)


def test_dsn_appends_connect_timeout_to_a_plain_dsn():
    dsn = make_settings().postgres_dsn
    assert dsn == "postgresql://santapola:santapola@localhost:5432/santapola?connect_timeout=5"


def test_dsn_joins_connect_timeout_after_sslmode():
    dsn = make_settings(postgres_sslmode="require").postgres_dsn
    assert "sslmode=require&connect_timeout=5" in dsn
