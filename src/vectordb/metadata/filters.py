"""Metadata filter parsing and evaluation."""

from enum import StrEnum
from typing import Any

from vectordb.storage.models import Document
from vectordb.types import Metadata, MetadataFilter


class FilterOperator(StrEnum):
    """Supported metadata filter operators."""

    EQ = "$eq"
    NE = "$ne"
    GT = "$gt"
    GTE = "$gte"
    LT = "$lt"
    LTE = "$lte"
    AND = "$and"
    OR = "$or"
    NOT = "$not"


RANGE_OPERATORS = {
    FilterOperator.GT,
    FilterOperator.GTE,
    FilterOperator.LT,
    FilterOperator.LTE,
}

LOGICAL_OPERATORS = {
    FilterOperator.AND,
    FilterOperator.OR,
    FilterOperator.NOT,
}


class FilterError(ValueError):
    """Raised when a metadata filter expression is invalid."""


def matches_metadata(metadata: Metadata, filter_expr: MetadataFilter) -> bool:
    """Return True when metadata satisfies the filter expression."""
    if not filter_expr:
        return True

    if FilterOperator.AND in filter_expr:
        return _matches_and(metadata, filter_expr[FilterOperator.AND])

    if FilterOperator.OR in filter_expr:
        return _matches_or(metadata, filter_expr[FilterOperator.OR])

    if FilterOperator.NOT in filter_expr:
        return _matches_not(metadata, filter_expr[FilterOperator.NOT])

    for field, condition in filter_expr.items():
        if field in LOGICAL_OPERATORS:
            continue
        if not _matches_field(metadata, field, condition):
            return False

    return True


def filter_documents(
    documents: list[Document],
    filter_expr: MetadataFilter | None,
) -> list[Document]:
    """Return documents whose metadata matches the filter expression."""
    if filter_expr is None:
        return documents
    return [document for document in documents if matches_metadata(document.metadata, filter_expr)]


def _matches_and(metadata: Metadata, clauses: Any) -> bool:
    if not isinstance(clauses, list):
        raise FilterError("$and requires a list of filter expressions")
    return all(matches_metadata(metadata, clause) for clause in clauses)


def _matches_or(metadata: Metadata, clauses: Any) -> bool:
    if not isinstance(clauses, list):
        raise FilterError("$or requires a list of filter expressions")
    return any(matches_metadata(metadata, clause) for clause in clauses)


def _matches_not(metadata: Metadata, clause: Any) -> bool:
    if not isinstance(clause, dict):
        raise FilterError("$not requires a filter expression object")
    return not matches_metadata(metadata, clause)


def _matches_field(metadata: Metadata, field: str, condition: Any) -> bool:
    if field not in metadata:
        return False

    value = metadata[field]
    if isinstance(condition, dict) and _is_operator_condition(condition):
        return all(
            _evaluate_operator(value, FilterOperator(operator), expected)
            for operator, expected in condition.items()
            if operator in FilterOperator._value2member_map_
        )

    return bool(value == condition)


def _is_operator_condition(condition: dict[str, Any]) -> bool:
    return any(key in FilterOperator._value2member_map_ for key in condition)


def _evaluate_operator(value: Any, operator: FilterOperator, expected: Any) -> bool:
    if operator == FilterOperator.EQ:
        return bool(value == expected)
    if operator == FilterOperator.NE:
        return bool(value != expected)
    if operator in RANGE_OPERATORS:
        return _evaluate_range(value, operator, expected)
    raise FilterError(f"Unsupported operator in field expression: {operator}")


def _evaluate_range(value: Any, operator: FilterOperator, expected: Any) -> bool:
    if not _is_comparable(value, expected):
        return False

    if operator == FilterOperator.GT:
        return bool(value > expected)
    if operator == FilterOperator.GTE:
        return bool(value >= expected)
    if operator == FilterOperator.LT:
        return bool(value < expected)
    if operator == FilterOperator.LTE:
        return bool(value <= expected)
    raise FilterError(f"Unsupported range operator: {operator}")


def _is_comparable(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    return isinstance(left, (int, float, str))
