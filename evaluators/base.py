from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, decoded_ind: list[int], n_eval: int) -> dict[str, Any]:
        """
        Evaluate a decoded architecture and return a standardized dictionary.

        Expected minimum structure:
        {
            "score": float,
            "details": dict
        }
        """
        raise NotImplementedError