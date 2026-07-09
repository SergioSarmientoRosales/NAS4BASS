from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def primary_score_from_objective(obj) -> float:
    primary_obj = float(obj[0])
    return -primary_obj if np.isfinite(primary_obj) else np.nan


def params_from_objective(obj) -> int:
    params = float(obj[1]) if len(obj) > 1 else np.inf
    return int(params) if np.isfinite(params) and params >= 0 else int(1e18)


def dominates(a, b) -> bool:
    a_vals = [float(value) if np.isfinite(value) else np.inf for value in a]
    b_vals = [float(value) if np.isfinite(value) else np.inf for value in b]
    return all(a_i <= b_i for a_i, b_i in zip(a_vals, b_vals)) and any(
        a_i < b_i for a_i, b_i in zip(a_vals, b_vals)
    )


def non_dominated_indices(objectives: Iterable) -> list[int]:
    values = list(objectives)
    indexes: list[int] = []
    for i, obj_i in enumerate(values):
        if not any(j != i and dominates(obj_j, obj_i) for j, obj_j in enumerate(values)):
            indexes.append(i)
    return indexes


def make_front(pop: dict, indexes: list[int]) -> dict:
    return {
        "X": [pop["X"][idx] for idx in indexes],
        "F": [pop["F"][idx] for idx in indexes],
    }


def evaluation_budget(pop_size: int, n_gen: int | None) -> int:
    if n_gen is None:
        return int(pop_size)
    return max(1, int(pop_size) * max(1, int(n_gen)))


def scalarize_objective(obj, weights: tuple[float, float] = (1.0, 0.0)) -> float:
    primary = float(obj[0]) if np.isfinite(obj[0]) else float("inf")
    complexity = float(obj[1]) if len(obj) > 1 and np.isfinite(obj[1]) else float("inf")
    return float(weights[0]) * primary + float(weights[1]) * complexity


def random_binary_individual(rng: np.random.Generator, n_var: int) -> list[int]:
    return rng.integers(0, 2, size=n_var, endpoint=False).astype(int).tolist()


def mutate_one_bit(
    rng: np.random.Generator,
    candidate: list[int],
    *,
    n_mutations: int = 1,
) -> list[int]:
    child = list(candidate)
    n_mutations = max(1, min(len(child), int(n_mutations)))
    for idx in rng.choice(len(child), size=n_mutations, replace=False):
        child[int(idx)] = 1 - int(child[int(idx)])
    return child


@dataclass
class EvaluationRecord:
    eval_id: int
    iteration: int
    candidate_id: int
    candidate: list[int]
    objectives: list[float]
    decoded_architecture: list[int]
    elapsed_sec: float
    budget: int | float = 1
    accepted: bool | str = ""


class EvaluationLogger:
    def __init__(self, output_file: str | None):
        self.output_file = output_file
        self._header_written = False

    def write(self, records: list[EvaluationRecord]) -> None:
        if not self.output_file or not records:
            return

        file_exists = os.path.exists(self.output_file)
        with open(self.output_file, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if not file_exists or not self._header_written:
                writer.writerow(
                    [
                        "evaluation_id",
                        "iteration",
                        "candidate_id",
                        "budget",
                        "accepted",
                        "decoded_architecture",
                        "primary_score",
                        "params",
                        "objective_0",
                        "objective_1",
                        "elapsed_sec",
                    ]
                )
                self._header_written = True

            for record in records:
                writer.writerow(
                    [
                        record.eval_id,
                        record.iteration,
                        record.candidate_id,
                        record.budget,
                        record.accepted,
                        " ".join(map(str, record.decoded_architecture)),
                        primary_score_from_objective(record.objectives),
                        params_from_objective(record.objectives),
                        float(record.objectives[0]),
                        float(record.objectives[1]) if len(record.objectives) > 1 else "",
                        f"{record.elapsed_sec:.6f}",
                    ]
                )


class BudgetedSearchMixin:
    def _init_budgeted_search(self) -> None:
        self.n_eval = 1
        self.rng = np.random.default_rng(getattr(self, "seed", None))
        self.logger = EvaluationLogger(self.output_file)

    @property
    def max_evals(self) -> int:
        if getattr(self, "max_evals_override", None) is not None:
            return max(1, int(self.max_evals_override))
        return evaluation_budget(self.pop_size, self.n_gen)

    def _evaluate_candidate(
        self,
        candidate: list[int],
        *,
        iteration: int,
        candidate_id: int,
        budget: int | float = 1,
        accepted: bool | str = "",
    ) -> tuple[list[float], EvaluationRecord]:
        start = time.time()
        obj = self.problem._evaluate_multi(candidate, self.n_eval)
        elapsed = time.time() - start
        record = EvaluationRecord(
            eval_id=self.n_eval,
            iteration=iteration,
            candidate_id=candidate_id,
            candidate=list(candidate),
            objectives=list(obj),
            decoded_architecture=self.problem.get_decoded_ind(candidate),
            elapsed_sec=elapsed,
            budget=budget,
            accepted=accepted,
        )
        self.n_eval += 1
        return obj, record

    def _finalize_population(self, pop: dict) -> tuple[dict, dict]:
        nds_index = non_dominated_indices(pop["F"])
        return pop, make_front(pop, nds_index)
