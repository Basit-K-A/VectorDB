"""Pydantic request and response schemas for the HTTP API."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    app_name: str
    environment: str


class ErrorResponse(BaseModel):
    """Standard API error envelope."""

    detail: str
    code: str | None = None


class CollectionCreateRequest(BaseModel):
    """Request body for creating a collection."""

    name: str = Field(..., min_length=1, max_length=128)
    dimension: int = Field(..., gt=0)


class CollectionResponse(BaseModel):
    """Collection metadata response."""

    name: str
    dimension: int
    vector_count: int = 0


class QueryRequestSchema(BaseModel):
    """Request body for a similarity query."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=1000)
    metadata_filter: dict[str, object] | None = None


class QueryResultSchema(BaseModel):
    """A single query result item."""

    id: str
    score: float
    metadata: dict[str, object] = Field(default_factory=dict)


class QueryResponseSchema(BaseModel):
    """Response body for a similarity query."""

    query: str
    results: list[QueryResultSchema]
    total: int
