"""Health check endpoints."""

from fastapi import APIRouter

from vectordb.api.dependencies import SettingsDep
from vectordb.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: SettingsDep) -> HealthResponse:
    """Return application health status."""
    return HealthResponse(
        app_name=settings.app_name,
        environment=settings.environment.value,
    )
