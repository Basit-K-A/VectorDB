"""Tests for logging configuration."""

import json
import logging

from vectordb.config import LogFormat, Settings
from vectordb.logging import JsonFormatter, configure_logging, get_logger


def test_configure_logging_text_format() -> None:
    settings = Settings(log_level="DEBUG", log_format=LogFormat.TEXT)
    configure_logging(settings)

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1


def test_configure_logging_json_format() -> None:
    settings = Settings(log_level="INFO", log_format=LogFormat.JSON)
    configure_logging(settings)

    handler = logging.getLogger().handlers[0]
    assert isinstance(handler.formatter, JsonFormatter)


def test_json_formatter_output() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="vectordb.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello",
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))
    assert payload["level"] == "INFO"
    assert payload["message"] == "hello"
    assert payload["logger"] == "vectordb.test"


def test_get_logger_namespaces() -> None:
    logger = get_logger("api")
    assert logger.name == "vectordb.api"
