"""Version 1 API routes."""

from fastapi import APIRouter

from vectordb.api.routes.v1 import collections, query

router = APIRouter()
router.include_router(collections.router, prefix="/collections", tags=["collections"])
router.include_router(query.router, prefix="/query", tags=["query"])
