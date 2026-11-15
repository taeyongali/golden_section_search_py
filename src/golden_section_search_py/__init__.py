"""Golden section search for minimizing unimodal scalar functions."""

from .core import golden_section_search, GoldenSectionSearchError, GoldenSectionSearchResult

__all__ = [
    "golden_section_search",
    "GoldenSectionSearchError",
    "GoldenSectionSearchResult",
]
