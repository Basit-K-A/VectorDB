"""Command-line entry point for the vector database server."""

import uvicorn

from vectordb.config import get_settings
from vectordb.logging import configure_logging, get_logger


def main() -> None:
    """Start the FastAPI application with uvicorn."""
    settings = get_settings()
    configure_logging(settings)
    logger = get_logger("cli")
    logger.info("Starting %s on %s:%s", settings.app_name, settings.host, settings.port)

    uvicorn.run(
        "vectordb.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
