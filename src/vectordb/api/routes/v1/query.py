"""Query endpoints (scaffolding only)."""

from fastapi import APIRouter

from vectordb.api.schemas import ErrorResponse, QueryRequestSchema, QueryResponseSchema

router = APIRouter()


@router.post(
    "/{collection}",
    response_model=QueryResponseSchema,
    responses={501: {"model": ErrorResponse}},
)
async def query_collection(
    collection: str,
    body: QueryRequestSchema,
) -> QueryResponseSchema:
    """Execute a similarity query against a collection. Not yet implemented."""
    raise NotImplementedError("Query execution is not yet implemented")
