from __future__ import annotations

import numpy as np

from search.base import BaseSearch
from search.common import (
    BudgetedSearchMixin,
    random_binary_individual,
    scalarize_objective,
)
from search.progress import tqdm


class BayesianOptimizationSearch(BudgetedSearchMixin, BaseSearch):
    """Lightweight internal surrogate search over binary architecture bits."""

    def __init__(
        self,
        *args,
        seed: int | None = None,
        scalarization_weights=None,
        initial_random: int = 16,
        acquisition_candidates: int = 128,
        exploration_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(*args, seed=seed, **kwargs)
        self.weights = tuple(scalarization_weights or (1.0, 0.0))
        self.initial_random = int(initial_random)
        self.acquisition_candidates = int(acquisition_candidates)
        self.exploration_weight = float(exploration_weight)
        self._init_budgeted_search()

    def _suggest(self, x_seen: list[list[int]], y_seen: list[float]) -> list[int]:
        if len(x_seen) < max(2, self.initial_random):
            return random_binary_individual(self.rng, self.problem.n_var)

        x = np.asarray(x_seen, dtype=np.float64)
        y = np.asarray(y_seen, dtype=np.float64)
        x_aug = np.c_[np.ones(len(x)), x]
        reg = 1e-6 * np.eye(x_aug.shape[1])
        coef = np.linalg.pinv(x_aug.T @ x_aug + reg) @ x_aug.T @ y
        candidates = np.asarray(
            [random_binary_individual(self.rng, self.problem.n_var) for _ in range(self.acquisition_candidates)],
            dtype=np.float64,
        )
        cand_aug = np.c_[np.ones(len(candidates)), candidates]
        pred = cand_aug @ coef
        uncertainty = np.sqrt(np.mean((candidates[:, None, :] - x[None, :, :]) ** 2, axis=(1, 2)))
        acquisition = pred - self.exploration_weight * uncertainty
        return candidates[int(np.argmin(acquisition))].astype(int).tolist()

    def run(self):
        pop = {"X": [], "F": []}
        records = []
        x_seen: list[list[int]] = []
        y_seen: list[float] = []
        seen = set()
        budget = self.max_evals
        pbar = tqdm(total=budget, desc="BayesOpt", unit="eval") if self.verbose else None

        for eval_idx in range(budget):
            candidate = self._suggest(x_seen, y_seen)
            retry = 0
            while tuple(candidate) in seen and retry < 32:
                candidate = random_binary_individual(self.rng, self.problem.n_var)
                retry += 1
            seen.add(tuple(candidate))
            obj, record = self._evaluate_candidate(candidate, iteration=eval_idx, candidate_id=eval_idx)
            scalar = scalarize_objective(obj, self.weights)
            x_seen.append(candidate)
            y_seen.append(scalar)
            pop["X"].append(candidate)
            pop["F"].append(obj)
            records.append(record)
            if pbar is not None:
                pbar.update(1)
            if len(records) >= self.save_flush_every:
                self.logger.write(records)
                records = []

        if pbar is not None:
            pbar.close()
        self.logger.write(records)
        return self._finalize_population(pop)
