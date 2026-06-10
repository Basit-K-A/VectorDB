"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from vectordb.api.routes import health, v1
from vectordb.config import Settings, get_settings
from vectordb.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown."""
    settings: Settings = app.state.settings
    logger = get_logger("api")
    logger.info("Application starting (environment=%s)", settings.environment.value)
    yield
    logger.info("Application shutting down")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    resolved = settings or get_settings()
    configure_logging(resolved)

    app = FastAPI(
        title=resolved.app_name,
        version="0.1.0",
        debug=resolved.debug,
        lifespan=lifespan,
    )
    app.state.settings = resolved

    app.include_router(health.router)
    app.include_router(v1.router, prefix="/api/v1")

    return app
