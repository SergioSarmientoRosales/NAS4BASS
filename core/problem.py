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

        # decode cache: keyed by RAW bitstring
        self.decode_cache: dict[tuple[int, ...], list[int]] = {}

        # all other caches: keyed by DECODED architecture
        self.score_cache: dict[tuple[int, ...], float] = {}
        self.param_cache: dict[tuple[int, ...], int] = {}
        self.obj_cache: dict[tuple[int, ...], list[float]] = {}
        self.evaluation_details_cache: dict[tuple[int, ...], dict[str, Any]] = {}

        # real cache usage stats
        self.cache_stats: dict[str, int] = {
            "decode_hits": 0,
            "decode_misses": 0,
            "score_hits": 0,
            "score_misses": 0,
            "param_hits": 0,
            "param_misses": 0,
            "obj_hits": 0,
            "obj_misses": 0,
        }

    @staticmethod
    def _normalize_seq(seq) -> tuple[int, ...]:
        if isinstance(seq, np.ndarray):
            return tuple(int(x) for x in seq.tolist())
        return tuple(int(x) for x in seq)

    def _raw_key(self, ind) -> tuple[int, ...]:
        return self._normalize_seq(ind)

    def _arch_key_from_decoded(
        self, decoded_ind: list[int] | tuple[int, ...]
    ) -> tuple[int, ...]:
        return self._normalize_seq(decoded_ind)

    def _print_cache_hit(self, name: str) -> None:
        if self.verbose_cache:
            print(f"[CACHE HIT] {name}")

    def _print_cache_miss(self, name: str) -> None:
        if self.verbose_cache:
            print(f"[CACHE MISS] {name}")

    # ------------------------------------------------------------------
    # Decode: expects RAW 84-bit individual
    # ------------------------------------------------------------------
    def get_decoded_ind(self, ind) -> list[int]:
        raw_key = self._raw_key(ind)

        if self.use_decode_cache and raw_key in self.decode_cache:
            self.cache_stats["decode_hits"] += 1
            self._print_cache_hit("decode")
            return self.decode_cache[raw_key]

        self.cache_stats["decode_misses"] += 1
        self._print_cache_miss("decode")

        decoded = bstr_to_rstr(list(raw_key))

        if self.use_decode_cache:
            self.decode_cache[raw_key] = decoded

        return decoded

    # ------------------------------------------------------------------
    # Internal score evaluation: expects DECODED individual
    # ------------------------------------------------------------------
    def _evaluate_primary_score_from_decoded(
        self, decoded_ind: list[int], n_eval: int
    ) -> float:
        arch_key = self._arch_key_from_decoded(decoded_ind)

        if self.use_score_cache and arch_key in self.score_cache:
            self.cache_stats["score_hits"] += 1
            self._print_cache_hit("score")
            return self.score_cache[arch_key]

        self.cache_stats["score_misses"] += 1
        self._print_cache_miss("score")

        result = self.evaluator.evaluate(decoded_ind, n_eval)
        score = float(result["score"])

        if self.use_score_cache:
            self.score_cache[arch_key] = score
            self.evaluation_details_cache[arch_key] = result.get("details", {})

        return score

    # public version: accepts RAW individual
    def evaluate_primary_score(self, ind, n_eval: int) -> float:
        decoded_ind = self.get_decoded_ind(ind)
        return self._evaluate_primary_score_from_decoded(decoded_ind, n_eval)

    # ------------------------------------------------------------------
    # Internal param evaluation: expects DECODED individual
    # Used as fallback when evaluator does not provide params directly
    # ------------------------------------------------------------------
    def _func_eval_params_from_decoded(
        self, decoded_ind: list[int], random_seed: int = 1
    ) -> int:
        arch_key = self._arch_key_from_decoded(decoded_ind)

        if self.use_param_cache and arch_key in self.param_cache:
            self.cache_stats["param_hits"] += 1
            self._print_cache_hit("params")
            return self.param_cache[arch_key]

        self.cache_stats["param_misses"] += 1
        self._print_cache_miss("params")

        random.seed(random_seed)
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
            self.param_cache[arch_key] = params

        return params

    @staticmethod
    def _make_objectives(score: float, params: float | int) -> list[float]:
        score = float(score)
        params = float(params)

        if not np.isfinite(score):
            return [float("inf"), int(1e18)]

        primary_objective = -score
        params_objective = int(params) if np.isfinite(params) and params >= 0 else int(1e18)

        return [float(primary_objective), params_objective]

    # public version: accepts RAW individual
    def func_eval_params(self, ind, random_seed: int = 1) -> int:
        decoded_ind = self.get_decoded_ind(ind)
        return self._func_eval_params_from_decoded(
            decoded_ind, random_seed=random_seed
        )

    # ------------------------------------------------------------------
    # Multi-objective evaluation: accepts RAW individual
    # ------------------------------------------------------------------
    def _evaluate_multi(self, ind, n_eval: int) -> list[float]:
        decoded_ind = self.get_decoded_ind(ind)
        arch_key = self._arch_key_from_decoded(decoded_ind)

        if self.use_obj_cache and arch_key in self.obj_cache:
            self.cache_stats["obj_hits"] += 1
            self._print_cache_hit("objectives")
            return self.obj_cache[arch_key]

        self.cache_stats["obj_misses"] += 1
        self._print_cache_miss("objectives")

        # Fast path: zero-cost evaluator can return score + params
        if hasattr(self.evaluator, "evaluate_with_params"):
            result = self.evaluator.evaluate_with_params(decoded_ind, n_eval)
            score = float(result["score"])
            params = int(result["params"])

            if self.use_score_cache:
                self.score_cache[arch_key] = score
                self.evaluation_details_cache[arch_key] = result.get("details", {})

            if self.use_param_cache:
                self.param_cache[arch_key] = params

        # Fallback path: generic evaluator API
        else:
            score = self._evaluate_primary_score_from_decoded(decoded_ind, n_eval)
            params = self._func_eval_params_from_decoded(decoded_ind)

        objectives = self._make_objectives(score=score, params=params)

        if self.use_obj_cache:
            self.obj_cache[arch_key] = objectives

        return objectives

    # ------------------------------------------------------------------
    # Cache summary
    # ------------------------------------------------------------------
    def get_cache_summary(self) -> dict[str, int]:
        return {
            **self.cache_stats,
            "decode_cache_size": len(self.decode_cache),
            "score_cache_size": len(self.score_cache),
            "param_cache_size": len(self.param_cache),
            "obj_cache_size": len(self.obj_cache),
            "evaluation_details_cache_size": len(self.evaluation_details_cache),
        }
