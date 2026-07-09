from __future__ import annotations

import math

from search.common import random_binary_individual, scalarize_objective
from search.progress import tqdm
from search.successive_halving import SuccessiveHalvingSearch


class HyperbandSearch(SuccessiveHalvingSearch):
    def run(self):
        pop = {"X": [], "F": []}
        records = []
        budget = self.max_evals
        brackets = max(1, int(math.log(max(2, budget), self.eta)))
        eval_count = 0
        pbar = tqdm(total=budget, desc="Hyperband", unit="eval") if self.verbose else None

        for bracket in range(brackets, -1, -1):
            if eval_count >= budget:
                break
            n = max(1, min(budget - eval_count, self.eta ** bracket))
            remaining = [random_binary_individual(self.rng, self.problem.n_var) for _ in range(n)]
            round_idx = 0
            while remaining and eval_count < budget:
                scored = []
                for candidate_id, candidate in enumerate(remaining):
                    obj, record = self._evaluate_candidate(
                        candidate,
                        iteration=bracket * 100 + round_idx,
                        candidate_id=candidate_id,
                        budget=self.min_budget * (self.eta ** round_idx),
                    )
                    eval_count += 1
                    scalar = scalarize_objective(obj, self.weights)
                    scored.append((scalar, candidate))
                    pop["X"].append(candidate)
                    pop["F"].append(obj)
                    records.append(record)
                    if pbar is not None:
                        pbar.update(1)
                    if eval_count >= budget:
                        break
                scored.sort(key=lambda item: item[0])
                keep = max(1, len(scored) // self.eta)
                if keep >= len(scored):
                    break
                remaining = [candidate for _, candidate in scored[:keep]]
                round_idx += 1

        if pbar is not None:
            pbar.close()
        self.logger.write(records)
        return self._finalize_population(pop)
