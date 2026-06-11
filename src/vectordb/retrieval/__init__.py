"""Vector similarity retrieval layer."""

from vectordb.retrieval.base import RetrievalRequest, RetrievalResult, Retriever
from vectordb.retrieval.brute_force_search import BruteForceSearcher, RankedDocument
from vectordb.retrieval.pipeline import RetrievalPipeline
from vectordb.retrieval.similarity import (
    SimilarityMetric,
    compute_similarity,
    cosine_similarity,
    dot_product_similarity,
    top_k_indices,
)

__all__ = [
    "BruteForceSearcher",
    "RankedDocument",
    "RetrievalPipeline",
    "RetrievalRequest",
    "RetrievalResult",
    "Retriever",
    "SimilarityMetric",
    "compute_similarity",
    "cosine_similarity",
    "dot_product_similarity",
    "top_k_indices",
]
