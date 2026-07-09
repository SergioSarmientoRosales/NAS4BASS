from __future__ import annotations

import numpy as np

from search.base import BaseSearch
from search.common import (
    BudgetedSearchMixin,
    dominates,
    make_front,
    non_dominated_indices,
    random_binary_individual,
)
from search.operators import (
    BitFlipMutation,
    KPointBinaryCrossover,
    TournamentSelection,
)


class NSGA2(BudgetedSearchMixin, BaseSearch):
    """Minimal NSGA-II style baseline using dominance rank and crowding distance."""

    def __init__(self, *args, seed: int | None = None, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self.rng = np.random.default_rng(seed)
        self._init_budgeted_search()
        self._records = []

    @staticmethod
    def _crowding(front_f):
        values = np.asarray(front_f, dtype=np.float64)
        n = len(values)
        if n == 0:
            return []
        distance = np.zeros(n, dtype=np.float64)
        for m in range(values.shape[1]):
            order = np.argsort(values[:, m])
            distance[order[0]] = distance[order[-1]] = np.inf
            low, high = values[order[0], m], values[order[-1], m]
            if high == low:
                continue
            for idx in range(1, n - 1):
                distance[order[idx]] += (values[order[idx + 1], m] - values[order[idx - 1], m]) / (high - low)
        return distance.tolist()

    def _sort_fronts(self, pop):
        remaining = set(range(len(pop["F"])))
        fronts = []
        while remaining:
            front = [
                idx
                for idx in remaining
                if not any(other != idx and dominates(pop["F"][other], pop["F"][idx]) for other in remaining)
            ]
            fronts.append(front)
            remaining.difference_update(front)
        return fronts

    def _select(self, pop):
        fronts = self._sort_fronts(pop)
        selected = []
        for front in fronts:
            if len(selected) + len(front) <= self.pop_size:
                selected.extend(front)
                continue
            crowding = self._crowding([pop["F"][idx] for idx in front])
            ordered = [idx for _, idx in sorted(zip(crowding, front), reverse=True)]
            selected.extend(ordered[: self.pop_size - len(selected)])
            break
        return {"X": [pop["X"][idx] for idx in selected], "F": [pop["F"][idx] for idx in selected]}

    def _initial_pop(self):
        x = []
        f = []
        for idx in range(self.pop_size):
            candidate = random_binary_individual(self.rng, self.problem.n_var)
            obj, record = self._evaluate_candidate(candidate, iteration=0, candidate_id=idx)
            x.append(candidate)
            f.append(obj)
            self._records.append(record)
        return {"X": x, "F": f}

    def run(self):
        generations = max(1, int(self.n_gen or 1))
        pop = self._initial_pop()
        for generation in range(generations):
            offspring = {"X": [], "F": []}
            for _ in range(self.pop_size):
                parents = TournamentSelection(n_parents=2)(pop=pop)
                child = KPointBinaryCrossover(problem=self.problem)(parents, pop)
                child = BitFlipMutation(problem=self.problem, prob=1 / self.problem.n_var)(child).tolist()
                obj, record = self._evaluate_candidate(
                    child,
                    iteration=generation + 1,
                    candidate_id=len(offspring["X"]),
                )
                offspring["X"].append(child)
                offspring["F"].append(obj)
                self._records.append(record)
            combined = {"X": pop["X"] + offspring["X"], "F": pop["F"] + offspring["F"]}
            pop = self._select(combined)
            if len(self._records) >= self.save_flush_every:
                self.logger.write(self._records)
                self._records = []
        self.logger.write(self._records)
        nd_idx = non_dominated_indices(pop["F"])
        return pop, make_front(pop, nd_idx)
