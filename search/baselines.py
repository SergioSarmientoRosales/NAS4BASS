from __future__ import annotations

import math

import numpy as np

from search.base import BaseSearch
from search.common import (
    BudgetedSearchMixin,
    dominates,
    make_front,
    mutate_one_bit,
    non_dominated_indices,
    random_binary_individual,
    scalarize_objective,
)
from search.operators import (
    BitFlipMutation,
    KPointBinaryCrossover,
    TournamentSelection,
)
from search.progress import tqdm
from search.random_search import RandomSearch


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


class MultiObjectiveRandomSearch(RandomSearch):
    """Random-search budget with explicit non-dominated front extraction."""


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


class BayesianOptimizationSearch(BudgetedSearchMixin, BaseSearch):
    """Lightweight internal surrogate search using random Fourier-like features.

    This intentionally avoids adding Optuna/sklearn dependencies. It uses random
    initial points, ridge regression on binary architecture bits, and lower
    confidence bound acquisition over random candidates.
    """

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
