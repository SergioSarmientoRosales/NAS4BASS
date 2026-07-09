from __future__ import annotations

import math

from search.common import (
    mutate_one_bit,
    random_binary_individual,
    scalarize_objective,
)
from search.hill_climbing import HillClimbingSearch
from search.progress import tqdm


class SimulatedAnnealingSearch(HillClimbingSearch):
    def __init__(
        self,
        *args,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.95,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.initial_temperature = float(initial_temperature)
        self.cooling_rate = float(cooling_rate)

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
        pbar = tqdm(total=budget, desc="Simulated Annealing", unit="eval") if self.verbose else None
        if pbar is not None:
            pbar.update(1)

        for eval_idx in range(1, budget):
            temperature = max(1e-12, self.initial_temperature * (self.cooling_rate ** eval_idx))
            candidate = mutate_one_bit(self.rng, current)
            obj, record = self._evaluate_candidate(candidate, iteration=eval_idx, candidate_id=eval_idx)
            score = scalarize_objective(obj, self.weights)
            delta = score - current_score
            accepted = delta <= 0 or self.rng.random() < math.exp(-delta / temperature)
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
