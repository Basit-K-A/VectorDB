"""Collection management endpoints (scaffolding only)."""

from fastapi import APIRouter, status

from vectordb.api.schemas import CollectionCreateRequest, CollectionResponse, ErrorResponse

router = APIRouter()


@router.post(
    "",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    responses={501: {"model": ErrorResponse}},
)
async def create_collection(body: CollectionCreateRequest) -> CollectionResponse:
    """Create a new vector collection. Not yet implemented."""
    raise NotImplementedError("Collection creation is not yet implemented")


@router.get(
    "/{name}",
    response_model=CollectionResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_collection(name: str) -> CollectionResponse:
    """Retrieve collection metadata. Not yet implemented."""
    raise NotImplementedError("Collection retrieval is not yet implemented")
