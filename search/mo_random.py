from __future__ import annotations

from search.random_search import RandomSearch


class MultiObjectiveRandomSearch(RandomSearch):
    """Random-search budget with explicit non-dominated front extraction."""
