"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from vectordb.api.app import create_app
from vectordb.config import Environment, Settings, get_settings


@pytest.fixture
def test_settings() -> Settings:
    """Return settings configured for the test environment."""
    return Settings(
        environment=Environment.TEST,
        debug=True,
        log_level="DEBUG",
    )


@pytest.fixture
def app(test_settings: Settings):
    """Return a configured FastAPI application."""
    get_settings.cache_clear()
    application = create_app(test_settings)
    yield application
    get_settings.cache_clear()


@pytest.fixture
def client(app) -> TestClient:
    """Return a synchronous test client."""
    return TestClient(app)
