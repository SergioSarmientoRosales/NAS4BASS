from __future__ import annotations

import random
from typing import Any

import numpy as np
import tensorflow as tf

from search_space.encoding import bstr_to_rstr
from search_space.search_space import decode
from search_space.model_builder import get_model


class NASProblem:
    def __init__(
        self,
        evaluator,
        n_var: int = 84,
        n_obj: int = 2,
        use_decode_cache: bool = True,
        use_score_cache: bool = True,
        use_param_cache: bool = True,
        use_obj_cache: bool = True,
        verbose_cache: bool = False,
    ) -> None:
        self.evaluator = evaluator

        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = np.zeros(self.n_var, dtype=int)
        self.xu = np.ones(self.n_var, dtype=int)

        self.use_decode_cache = use_decode_cache
        self.use_score_cache = use_score_cache
        self.use_param_cache = use_param_cache
        self.use_obj_cache = use_obj_cache
        self.verbose_cache = verbose_cache

        self.decode_cache: dict[tuple[int, ...], list[int]] = {}
        self.score_cache: dict[tuple[int, ...], float] = {}
        self.param_cache: dict[tuple[int, ...], int] = {}
        self.obj_cache: dict[tuple[int, ...], list[float]] = {}
        self.evaluation_details_cache: dict[tuple[int, ...], dict[str, Any]] = {}

    @staticmethod
    def _normalize_ind(ind) -> tuple[int, ...]:
        if isinstance(ind, np.ndarray):
            return tuple(int(x) for x in ind.tolist())
        return tuple(int(x) for x in ind)

    def _print_cache_hit(self, name: str) -> None:
        if self.verbose_cache:
            print(f"[CACHE HIT] {name}")

    def get_decoded_ind(self, ind) -> list[int]:
        key = self._normalize_ind(ind)

        if self.use_decode_cache and key in self.decode_cache:
            self._print_cache_hit("decode")
            return self.decode_cache[key]

        decoded = bstr_to_rstr(list(key))

        if self.use_decode_cache:
            self.decode_cache[key] = decoded

        return decoded

    def evaluate_primary_score(self, ind, n_eval: int) -> float:
        key = self._normalize_ind(ind)

        if self.use_score_cache and key in self.score_cache:
            self._print_cache_hit("score")
            return self.score_cache[key]

        decoded_ind = self.get_decoded_ind(ind)
        result = self.evaluator.evaluate(decoded_ind, n_eval)

        score = float(result["score"])

        if self.use_score_cache:
            self.score_cache[key] = score
            self.evaluation_details_cache[key] = result.get("details", {})

        return score

    def func_eval_params(self, ind, random_seed: int = 1) -> int:
        key = self._normalize_ind(ind)

        if self.use_param_cache and key in self.param_cache:
            self._print_cache_hit("params")
            return self.param_cache[key]

        random.seed(random_seed)

        decoded_ind = self.get_decoded_ind(ind)
        genotype = decode(decoded_ind)

        model = None
        try:
            model = get_model(genotype)
            params = int(model.count_params())
        finally:
            try:
                del model
            except Exception:
                pass
            tf.keras.backend.clear_session()

        if self.use_param_cache:
            self.param_cache[key] = params

        return params

    def _evaluate_multi(self, ind, n_eval: int) -> list[float]:
        key = self._normalize_ind(ind)

        if self.use_obj_cache and key in self.obj_cache:
            self._print_cache_hit("objectives")
            return self.obj_cache[key]

        score = self.evaluate_primary_score(ind, n_eval)
        params = self.func_eval_params(ind)

        objectives = [-float(score), int(params)]

        if self.use_obj_cache:
            self.obj_cache[key] = objectives

        return objectives

    def get_cache_summary(self) -> dict[str, int]:
        return {
            "decode_cache": len(self.decode_cache),
            "score_cache": len(self.score_cache),
            "param_cache": len(self.param_cache),
            "obj_cache": len(self.obj_cache),
            "evaluation_details_cache": len(self.evaluation_details_cache),
        }