"""FastAPI dependency injection helpers."""

from typing import Annotated

from fastapi import Depends, Request

from vectordb.config import Settings, get_settings


def get_app_settings(request: Request) -> Settings:
    """Return settings from application state, falling back to cached defaults."""
    return getattr(request.app.state, "settings", get_settings())


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
