from __future__ import annotations
import os
import csv
import numpy as np
from tqdm import tqdm
from search.operators import TournamentSelection, KPointBinaryCrossover, BitFlipMutation
from search.base import BaseSearch

class SMSEMOA(BaseSearch):
    """
    SMS-EMOA (S-Metric Selection Evolutionary Multi-Objective Algorithm)
    adapted for binary search spaces (e.g., NAS).

    Steady-state strategy (μ + 1): In each generation, ONE offspring is generated,
    added to the population (size pop_size + 1), and the individual with the
    least contribution to the hypervolume of the current Pareto front is removed.

    Reference:
        Beume, Naujoks & Emmerich (2007).
        "SMS-EMOA: Multiobjective selection based on dominated hypervolume."
    """
    def __init__(
        self,
        problem,
        pop_size: int = 100,
        n_gen: int = 1000,
        verbose: bool = False,
        output_file: str | None = None,
        ref_point: list[float] | None = None,
    ):
        super().__init__(problem, pop_size, n_gen, verbose, output_file)
        self.ref_point = np.array(ref_point) if ref_point is not None else None
        self.n_eval = 1


    def _dominates(self, a: np.ndarray, b: np.ndarray) -> bool:
        """True si a domina a b (minimización)."""
        return bool(np.all(a <= b) and np.any(a < b))

    def _fast_non_dominated_sorting(self, objectives: list[list]) -> dict:
        n = len(objectives)
        sp = {i: [] for i in range(n)}
        nq = {i: 0  for i in range(n)}

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(np.array(objectives[i]), np.array(objectives[j])):
                    sp[i].append(j)
                elif self._dominates(np.array(objectives[j]), np.array(objectives[i])):
                    nq[i] += 1

        fronts = {}
        k = 1
        fronts[f"F{k}"] = [i for i in range(n) if nq[i] == 0]

        while fronts[f"F{k}"]:
            next_front = []
            for i in fronts[f"F{k}"]:
                for j in sp[i]:
                    nq[j] -= 1
                    if nq[j] == 0:
                        next_front.append(j)
            k += 1
            fronts[f"F{k}"] = next_front

        return fronts

    def _hypervolume_2d(self, pareto_front: np.ndarray, ref: np.ndarray) -> float:
        """
        Compute the hypervolume of a 2D Pareto front with respect to a reference point.
        """
        pts = pareto_front[
            np.all(pareto_front < ref, axis=1)
        ]
        if len(pts) == 0:
            return 0.0

        pts = pts[np.argsort(pts[:, 0])]

        hv = 0.0
        prev_x = ref[0]
        for pt in reversed(pts):
            hv += (prev_x - pt[0]) * (ref[1] - pt[1])
            prev_x = pt[0]
        return hv

    def _exclusive_contribution(
        self,
        front_objs: np.ndarray,
        idx: int,
        ref: np.ndarray,
    ) -> float:
        """
        Exclusive Hypervolume Contribution of the individual at index idx in front_objs.
        """
        hv_full = self._hypervolume_2d(front_objs, ref)
        reduced = np.delete(front_objs, idx, axis=0)
        if len(reduced) == 0:
            return hv_full
        hv_reduced = self._hypervolume_2d(reduced, ref)
        return hv_full - hv_reduced

    def _get_ref_point(self, objectives: list) -> np.ndarray:
        """Ref point = nadir + offset (10% range)"""
        if self.ref_point is not None:
            return self.ref_point
        objs = np.array(objectives)
        nadir = objs.max(axis=0)
        margin = np.abs(nadir) * 0.1 + 1e-6
        return nadir + margin

    def _select_worst(self, objectives: list) -> int:
        """
        1. Perform fast non-dominated sorting to identify the last front.
        2. If the last front has only one individual, return it.
        """
        fronts = self._fast_non_dominated_sorting(objectives)
        ref = self._get_ref_point(objectives)

        last_front_key = max(
            (k for k, v in fronts.items() if v),
            key=lambda k: int(k[1:])
        )
        last_front = fronts[last_front_key]

        if len(last_front) == 1:
            return last_front[0]

        front_objs = np.array([objectives[i] for i in last_front])
        hvc = [
            self._exclusive_contribution(front_objs, k, ref)
            for k in range(len(last_front))
        ]

        worst_local = int(np.argmin(hvc))
        return last_front[worst_local]

    def _initialize_pop(self):
        x, x_f = [], []
        for _ in range(self.pop_size):
            ind = np.random.randint(0, 2, size=self.problem.n_var).tolist()
            x.append(ind)
            x_f.append(self.problem._evaluate_multi(ind, self.n_eval))
            self.n_eval += 1
        return {"X": x, "F": x_f}

    def _new_individual(self, pop: dict) -> dict:
        """Generate one offspring using tournament selection, crossover, and mutation."""
        parents  = TournamentSelection(n_parents=2)(pop=pop)
        offspring = KPointBinaryCrossover(problem=self.problem)(parents, pop)
        x = BitFlipMutation(problem=self.problem, prob=1 / self.problem.n_var)(offspring)
        x_f = self.problem._evaluate_multi(x.tolist(), self.n_eval)
        self.n_eval += 1
        return {"X": [x.tolist()], "F": [x_f]}

    
    def _non_dominated_samples(self, pop: dict) -> list[int]:
        indexes = []
        for i, p in enumerate(pop["F"]):
            p = np.array(p)
            dominated = any(
                self._dominates(np.array(pop["F"][j]), p)
                for j in range(len(pop["F"])) if i != j
            )
            if not dominated:
                indexes.append(i)
        return indexes

    def _save_population(self, pop: dict, generation: int):
        if not self.output_file:
            return
        file_exists = os.path.exists(self.output_file)
        with open(self.output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "generation", "individual_id",
                    "decoded_architecture", "predicted_psnr", "params",
                ])
            for idx, (x, obj) in enumerate(zip(pop["X"], pop["F"])):
                decoded_arch = self.problem.get_decoded_ind(x)
                writer.writerow([
                    generation, idx,
                    " ".join(map(str, decoded_arch)),
                    float(obj[0]), int(obj[1]),
                ])

    def _do(self):
        """Executes the SMS-EMOA optimization process."""
        pop = self._initialize_pop()
        self._save_population(pop, generation=0)

        pbar = tqdm(total=self.n_gen, desc="SMS-EMOA Progress", unit="gen") if self.verbose else None

        for gen in range(1, self.n_gen + 1):

            # 1) Generate ONE offspring and add it to the population (size pop+1)
            child = self._new_individual(pop)
            pop["X"] = pop["X"] + child["X"]
            pop["F"] = pop["F"] + child["F"]

            # 2) Select the worst contributor to the hypervolume and remove it
            worst_idx = self._select_worst(pop["F"])
            pop["X"].pop(worst_idx)
            pop["F"].pop(worst_idx)
            self.problem.notify_generation(gen)
            self._save_population(pop, generation=gen)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        nds_idx = self._non_dominated_samples(pop)
        nds = {key: [pop[key][i] for i in nds_idx] for key in pop}
        return pop, nds

    def run(self):
        return self._do()