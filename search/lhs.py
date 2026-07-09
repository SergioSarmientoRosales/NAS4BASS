from __future__ import annotations

import numpy as np

from search.base import BaseSearch
from search.common import BudgetedSearchMixin
from search.progress import tqdm


class LatinHypercubeSearch(BudgetedSearchMixin, BaseSearch):
    def __init__(self, *args, seed: int | None = None, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self._init_budgeted_search()

    def _sample(self, n_samples: int) -> list[list[int]]:
        samples = np.zeros((n_samples, self.problem.n_var), dtype=int)
        for var_idx in range(self.problem.n_var):
            values = (np.arange(n_samples) + self.rng.random(n_samples)) / n_samples
            self.rng.shuffle(values)
            samples[:, var_idx] = values >= 0.5
        return samples.astype(int).tolist()

    def run(self):
        pop = {"X": [], "F": []}
        records = []
        budget = self.max_evals
        candidates = self._sample(budget)
        pbar = tqdm(total=budget, desc="LHS Search", unit="eval") if self.verbose else None

        for idx, candidate in enumerate(candidates):
            obj, record = self._evaluate_candidate(candidate, iteration=0, candidate_id=idx)
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
