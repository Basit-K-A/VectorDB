"""Tests for application configuration."""

import pytest

from vectordb.config import Environment, LogFormat, Settings, get_settings


def test_default_settings() -> None:
    settings = Settings()
    assert settings.app_name == "vectordb"
    assert settings.environment == Environment.DEVELOPMENT
    assert settings.port == 8000
    assert settings.log_format == LogFormat.TEXT


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VECTORDB_APP_NAME", "test-db")
    monkeypatch.setenv("VECTORDB_ENVIRONMENT", "test")
    monkeypatch.setenv("VECTORDB_LOG_FORMAT", "json")

    settings = Settings()
    assert settings.app_name == "test-db"
    assert settings.environment == Environment.TEST
    assert settings.log_format == LogFormat.JSON


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()


def test_environment_properties() -> None:
    dev = Settings(environment=Environment.DEVELOPMENT)
    prod = Settings(environment=Environment.PRODUCTION)

    assert dev.is_development is True
    assert dev.is_production is False
    assert prod.is_production is True
