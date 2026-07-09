from __future__ import annotations

from search.base import BaseSearch
from search.common import (
    BudgetedSearchMixin,
    mutate_one_bit,
    random_binary_individual,
    scalarize_objective,
)
from search.progress import tqdm


class HillClimbingSearch(BudgetedSearchMixin, BaseSearch):
    def __init__(self, *args, seed: int | None = None, scalarization_weights=None, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self.weights = tuple(scalarization_weights or (1.0, 0.0))
        self._init_budgeted_search()

    def run(self):
        pop = {"X": [], "F": []}
        records = []
        budget = self.max_evals
        current = random_binary_individual(self.rng, self.problem.n_var)
        current_obj, record = self._evaluate_candidate(current, iteration=0, candidate_id=0, accepted=True)
        current_score = scalarize_objective(current_obj, self.weights)
        pop["X"].append(current)
        pop["F"].append(current_obj)
        records.append(record)

        pbar = tqdm(total=budget, desc="Hill Climbing", unit="eval") if self.verbose else None
        if pbar is not None:
            pbar.update(1)

        for eval_idx in range(1, budget):
            candidate = mutate_one_bit(self.rng, current)
            obj, record = self._evaluate_candidate(
                candidate,
                iteration=eval_idx,
                candidate_id=eval_idx,
            )
            score = scalarize_objective(obj, self.weights)
            accepted = score <= current_score
            record.accepted = accepted
            if accepted:
                current = candidate
                current_obj = obj
                current_score = score
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
