from __future__ import annotations

from search.base import BaseSearch
from search.common import (
    BudgetedSearchMixin,
    random_binary_individual,
    scalarize_objective,
)
from search.progress import tqdm


class SuccessiveHalvingSearch(BudgetedSearchMixin, BaseSearch):
    def __init__(
        self,
        *args,
        seed: int | None = None,
        scalarization_weights=None,
        eta: int = 3,
        min_budget: int = 1,
        **kwargs,
    ):
        super().__init__(*args, seed=seed, **kwargs)
        self.weights = tuple(scalarization_weights or (1.0, 0.0))
        self.eta = max(2, int(eta))
        self.min_budget = max(1, int(min_budget))
        self._init_budgeted_search()

    def run(self):
        pop = {"X": [], "F": []}
        records = []
        remaining = [random_binary_individual(self.rng, self.problem.n_var) for _ in range(self.max_evals)]
        round_idx = 0
        pbar = tqdm(total=self.max_evals, desc="Successive Halving", unit="eval") if self.verbose else None

        while remaining:
            scored = []
            for candidate_id, candidate in enumerate(remaining):
                obj, record = self._evaluate_candidate(
                    candidate,
                    iteration=round_idx,
                    candidate_id=candidate_id,
                    budget=self.min_budget * (self.eta ** round_idx),
                )
                scalar = scalarize_objective(obj, self.weights)
                scored.append((scalar, candidate, obj))
                pop["X"].append(candidate)
                pop["F"].append(obj)
                records.append(record)
                if pbar is not None:
                    pbar.update(1)
                if self.n_eval > self.max_evals:
                    break
            if self.n_eval > self.max_evals:
                break
            scored.sort(key=lambda item: item[0])
            keep = max(1, len(scored) // self.eta)
            if keep >= len(scored):
                break
            remaining = [candidate for _, candidate, _ in scored[:keep]]
            round_idx += 1

        if pbar is not None:
            pbar.close()
        self.logger.write(records)
        return self._finalize_population(pop)
