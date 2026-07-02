from __future__ import annotations
import os
import csv
import copy
import numpy as np
from tqdm import tqdm
from search.operators import TournamentSelection, KPointBinaryCrossover, BitFlipMutation
from search.base import BaseSearch

# --- Internal Utility Classes ---
class _Indicators:
    """Collection of multi-objective indicators and dominance utilities used by islands."""
    @staticmethod
    def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
        """Standard Pareto dominance check."""
        return bool(np.all(a <= b) and np.any(a < b))

    @staticmethod
    def _non_dominated(objectives: np.ndarray) -> np.ndarray:
        """Returns a boolean mask of non-dominated solutions."""
        n = len(objectives)
        is_nd = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_nd[i]:
                continue
            dominated = (
                np.all(objectives <= objectives[i], axis=1) &
                np.any(objectives <  objectives[i], axis=1)
            )
            dominated[i] = False
            if dominated.any():
                is_nd[i] = False
        return is_nd

    @staticmethod
    def _normalize(objectives: np.ndarray):
        """Normalizes each objective to the range [0,1]."""
        if len(objectives) == 0:
            return objectives
        z_min = objectives.min(axis=0)
        z_max = objectives.max(axis=0)
        rng   = z_max - z_min
        rng[rng == 0] = 1e-10
        return (objectives - z_min) / rng

    @staticmethod
    def hv_contribution(objectives: np.ndarray, idx: int) -> float:
        if len(objectives) <= 1:
            return 0.0
        ref  = objectives.max(axis=0) * 1.1 + 1e-6
        def _hv2d(pts, r):
            pts = pts[np.all(pts < r, axis=1)]
            if len(pts) == 0:
                return 0.0
            pts = pts[np.argsort(pts[:, 0])]
            hv, px = 0.0, r[0]
            for p in reversed(pts):
                hv += (px - p[0]) * (r[1] - p[1])
                px  = p[0]
            return hv
        full    = _hv2d(objectives, ref)
        reduced = _hv2d(np.delete(objectives, idx, axis=0), ref)
        return full - reduced

    @staticmethod
    def r2_contribution(objectives: np.ndarray, idx: int,
                        n_weights: int = 50) -> float:
        if len(objectives) <= 1:
            return 0.0
        m = objectives.shape[1]
        # Pesos uniformes (simplex lattice 1D para 2 obj)
        w = np.linspace(0, 1, n_weights)
        W = np.column_stack([w, 1 - w]) if m == 2 else \
            np.random.dirichlet(np.ones(m), n_weights)

        def _r2(objs):
            if len(objs) == 0:
                return 0.0
            # Función de escalarización: achievement scalarizing (Chebyshev)
            vals = np.max(objs[:, None, :] * W[None, :, :], axis=2)  # (n, nw)
            return -np.mean(np.min(vals, axis=0))

        full    = _r2(objectives)
        reduced = _r2(np.delete(objectives, idx, axis=0))
        return abs(full - reduced)

    @staticmethod
    def igd_plus_contribution(objectives: np.ndarray, idx: int) -> float:
        if len(objectives) <= 1:
            return 0.0
        # Usa el frente actual como reference set (proxy)
        nd_mask = _Indicators._non_dominated(objectives)
        ref_set = objectives[nd_mask]

        def _igd_plus(objs):
            if len(objs) == 0 or len(ref_set) == 0:
                return 0.0
            total = 0.0
            for z in ref_set:
                diff  = np.maximum(objs - z, 0)
                dists = np.linalg.norm(diff, axis=1)
                total += dists.min()
            return total / len(ref_set)

        full    = _igd_plus(objectives)
        reduced = _igd_plus(np.delete(objectives, idx, axis=0))
        return abs(full - reduced)

    @staticmethod
    def eps_plus_contribution(objectives: np.ndarray, idx: int) -> float:
        if len(objectives) <= 1:
            return 0.0
        nd_mask = _Indicators._non_dominated(objectives)
        ref_set = objectives[nd_mask]

        def _eps_plus(objs):
            if len(objs) == 0 or len(ref_set) == 0:
                return 0.0
            # ε+(A,Z) = max_z min_a max_i (a_i - z_i)
            vals = []
            for z in ref_set:
                vals.append(np.min(np.max(objs - z, axis=1)))
            return max(vals) if vals else 0.0

        full    = _eps_plus(objectives)
        reduced = _eps_plus(np.delete(objectives, idx, axis=0))
        return abs(full - reduced)

    @staticmethod
    def delta_p_contribution(objectives: np.ndarray, idx: int,
                             p: int = 2) -> float:
        if len(objectives) <= 1:
            return 0.0
        nd_mask = _Indicators._non_dominated(objectives)
        ref_set = objectives[nd_mask]

        def _gdp(A, Z):
            if len(A) == 0 or len(Z) == 0:
                return 0.0
            dists = [np.min(np.linalg.norm(Z - a, axis=1)) for a in A]
            return (np.mean(np.array(dists) ** p)) ** (1 / p)

        def _delta_p(objs):
            return max(_gdp(objs, ref_set), _gdp(ref_set, objs))

        full    = _delta_p(objectives)
        reduced = _delta_p(np.delete(objectives, idx, axis=0))
        return abs(full - reduced)

    @classmethod
    def contribution(cls, indicator: str,
                     objectives: np.ndarray, idx: int) -> float:
        if len(objectives) <= 1:
            return 0.0
        if indicator == "hv":
            return cls.hv_contribution(objectives, idx)
        if indicator == "r2":
            return cls.r2_contribution(objectives, idx)
        if indicator == "igd_plus":
            return cls.igd_plus_contribution(objectives, idx)
        if indicator == "eps_plus":
            return cls.eps_plus_contribution(objectives, idx)
        if indicator == "delta_p":
            return cls.delta_p_contribution(objectives, idx)
        raise ValueError(f"Unknown indicator: {indicator}")

class _RieszEnergy:

    @staticmethod
    def energy(objectives: np.ndarray, s: float = 1.0) -> float:
        n   = len(objectives)
        if n <= 1:
            return 0.0
        tot = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(objectives[i] - objectives[j])
                if d > 1e-12:
                    tot += d ** (-s)
        return 2 * tot   # simétrico

    @staticmethod
    def worst_contribution(objectives: np.ndarray, s: float = 1.0) -> int:
        """Returns the index of the element that contributes most to the energy (the most 'clustered')."""
        n    = len(objectives)
        if n <= 1:
            return 0

        norm_objs = _Indicators._normalize(objectives)
        full = _RieszEnergy.energy(norm_objs, s)
        contribs = []
        for i in range(n):
            reduced = _RieszEnergy.energy(
                np.delete(norm_objs, i, axis=0), s
            )
            # C_Es(a, A) = 0.5 * (E(A) - E(A\{a}))
            contribs.append(0.5 * (full - reduced))
        return int(np.argmax(contribs))

class _Archive:
    """ Keep a bounded set of non-dominated solutions with Riesz s-energy pruning.  (algorithm 3) """
    def __init__(self, max_size: int, s: float = 1.0):
        self.max_size = max_size
        self.s        = s
        self.pos: list[np.ndarray] = []
        self.obj: list[np.ndarray] = []

    def __len__(self):
        return len(self.obj)

    @staticmethod
    def _dominates(a, b):
        return bool(np.all(a <= b) and np.any(a < b))

    def _weakly_dominates(self, a, b):
        return bool(np.all(a <= b))

    def insert(self, pos: np.ndarray, obj: np.ndarray) -> None:
        """Algorithm 3: insert only if not dominated, then prune."""
        keep = []
        for i, r_obj in enumerate(self.obj):
            if self._dominates(obj, r_obj):
                continue                      # dominated → discard old
            if self._weakly_dominates(r_obj, obj):
                return                        # the new one is dominated or equal → discard new
            keep.append(i)

        self.pos = [self.pos[i] for i in keep]
        self.obj = [self.obj[i] for i in keep]
        self.pos.append(pos.copy())
        self.obj.append(obj.copy())

        while len(self.obj) > self.max_size:
            objs  = np.array(self.obj)
            worst = _RieszEnergy.worst_contribution(objs, self.s)
            self.pos.pop(worst)
            self.obj.pop(worst)

    def get_all(self):
        return list(self.pos), list(self.obj)

class _Island:
    """Represents a single island in the IMIA algorithm, evolving a micro-population with a specific indicator and maintaining an external archive."""
    INDICATORS = ["hv", "r2", "igd_plus", "eps_plus", "delta_p"]

    def __init__(
        self,
        indicator: str,
        problem,
        micro_pop_size: int,
        archive_max_size: int,
        f_mig: int,
        riesz_s: float = 1.0,
    ):
        assert indicator in self.INDICATORS, f"Unknown indicator: {indicator}"
        self.indicator       = indicator
        self.problem         = problem
        self.micro_pop_size  = micro_pop_size
        self.archive         = _Archive(archive_max_size, s=riesz_s)
        self.f_mig           = f_mig
        self.n_eval_ref      = [1]

        self.pop_X: list[np.ndarray] = []
        self.pop_F: list[np.ndarray] = []

    def _fast_nds(self, objectives: np.ndarray) -> dict[str, list[int]]:
        """
        This is a standard implementation of the Fast Non-Dominated Sorting algorithm (NSGA-II style).
        It returns a dictionary mapping front names (e.g., "F1", "F2", ...) to lists of indices of solutions in that front.
        """
        n  = len(objectives)
        if n == 0:
            return {"F1": []}
        sp = {i: [] for i in range(n)}
        nq = {i: 0  for i in range(n)}
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = objectives[i], objectives[j]
                if _Indicators._dominates(a, b):
                    sp[i].append(j)
                elif _Indicators._dominates(b, a):
                    nq[i] += 1
        fronts, k = {}, 1
        fronts[f"F{k}"] = [i for i in range(n) if nq[i] == 0]
        while fronts[f"F{k}"]:
            nxt = []
            for i in fronts[f"F{k}"]:
                for j in sp[i]:
                    nq[j] -= 1
                    if nq[j] == 0:
                        nxt.append(j)
            k += 1
            if nxt:
                fronts[f"F{k}"] = nxt
            else:
                break
        return fronts

    def _generate_offspring(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a single offspring solution using tournament selection, crossover, and mutation.
        Evaluates the offspring and returns its decision variables and objective values.
        """
        pop = {"X": [x.tolist() for x in self.pop_X],
               "F": [f.tolist() for f in self.pop_F]}
        parents   = TournamentSelection(n_parents=2)(pop=pop)
        offspring = KPointBinaryCrossover(problem=self.problem)(parents, pop)
        x = BitFlipMutation(
            problem=self.problem, prob=1 / self.problem.n_var
        )(offspring)
        x_f = self.problem._evaluate_multi(
            x.tolist(), self.n_eval_ref[0]
        )
        self.n_eval_ref[0] += 1
        return x.astype(float), np.array(x_f, dtype=float)

    def _worst_in_front(self, front_indices: list[int],
                        normalized_objectives: np.ndarray) -> int:
        """
        find the index of the solution in front_indices that contributes least to the island's indicator.
        """
        if len(front_indices) == 1:
            return front_indices[0]
        
        front_objs = normalized_objectives[front_indices]
        if len(front_objs) <= 1:
            return front_indices[0]
            
        contribs   = [
            _Indicators.contribution(self.indicator, front_objs, k)
            for k in range(len(front_indices))
        ]
        return front_indices[int(np.argmin(contribs))]


    def run_isolated(self) -> None:
        """Evolves the island's micro-population for f_mig iterations in isolation, following Algorithm 2 from the paper."""
        for _ in range(self.f_mig):
            # 1) Generates a single offspring solution q (Algorithm 2)
            q_x, q_f = self._generate_offspring()
            
            # 2) Combine the offspring q with the current population P to form Q = P ∪ {q} (Algorithm 2)
            Q_X = self.pop_X + [q_x]
            Q_F = self.pop_F + [q_f]
            
            # 3) Normalize the objective values of Q (Algorithm 2)
            Q_F_array = np.array([f.tolist() for f in Q_F])
            Q_F_norm = _Indicators._normalize(Q_F_array)

            # 4) Non-dominated sorting
            fronts = self._fast_nds(Q_F_norm)

            # 5) Identify the last front F_t (the one with the highest index) that contains solutions (Algorithm 2)
            last_k = max(
                (k for k, v in fronts.items() if v),
                key=lambda k: int(k[1:])
            )
            last_front = fronts[last_k]

            # 6) Gets the index r_worst of the solution in F_t that contributes least to the island's indicator (Algorithm 2)
            r_worst = self._worst_in_front(last_front, Q_F_norm)

            # 7) If the offspring q is not the worst in F_t, insert it into the archive (Algorithm 2)
            q_idx = len(Q_X) - 1
            if q_idx != r_worst:
                self.archive.insert(q_x, q_f)

            # 8) Delete the solution at index r_worst from Q to form the new population P (Algorithm 2)
            self.pop_X = [Q_X[i] for i in range(len(Q_X)) if i != r_worst]
            self.pop_F = [Q_F[i] for i in range(len(Q_F)) if i != r_worst]

    def receive_migrants(
        self,
        migrants_X: list[np.ndarray],
        migrants_F: list[np.ndarray],
        n_mig: int,
    ) -> None:
        """
        Replace iteratively the worst contributing solutions to the indicator in the population with immigrants, recalculating contributions after each removal.
        Contributions are calculated on the normalized population (Algorithm 4).
        """
        if len(migrants_X) == 0:
            return

        n_replace = min(len(migrants_X), len(self.pop_F))
        
        if n_replace == 0:
            return

        for _ in range(n_replace):
            if len(self.pop_F) <= 1:
                break
                
            all_objs = np.array([f.tolist() for f in self.pop_F])
            norm_objs = _Indicators._normalize(all_objs)
            
            contribs = [
                _Indicators.contribution(self.indicator, norm_objs, i)
                for i in range(len(norm_objs))
            ]
            
            worst_idx = int(np.argmin(contribs))
            self.pop_X.pop(worst_idx)
            self.pop_F.pop(worst_idx)

        max_to_insert = self.micro_pop_size - len(self.pop_F)
        for x, f in zip(migrants_X[:max_to_insert], migrants_F[:max_to_insert]):
            self.archive.insert(x, f)
            self.pop_X.append(x.copy())
            self.pop_F.append(f.copy())



class IMIA(BaseSearch):
    """
    Island-based Multi-Indicator Algorithm (IMIA).

    Falcon-Cardona et al., IEEE TEVC 2020.
    5 islas steady-state (HV, R2, IGD+, ε+, Δp) con topología
    fully-connected, micro-poblaciones de tamaño pop_size // 5,
    archivo externo por isla con poda Riesz s-energy.
    """

    INDICATOR_NAMES = ["hv", "r2", "igd_plus", "eps_plus", "delta_p"]

    def __init__(
        self,
        problem,
        pop_size: int   = 100,
        n_gen: int      = 1000,
        verbose: bool   = False,
        output_file: str | None = None,
        f_mig: int      = 10,    
        n_mig: int      = 1,     
        riesz_s: float  = 1.0, 
    ):
        super().__init__(problem, pop_size, n_gen, verbose, output_file)
        self.f_mig   = f_mig
        self.n_mig   = n_mig
        self.riesz_s = riesz_s

        # Size of the micro-population per island (pop_size // 5)
        self.micro_size = max(2, pop_size // len(self.INDICATOR_NAMES))

        # Counter of evaluations shared between islands
        self._n_eval = [1]

    # Inicialización ------------------------------------------------------------------

    def _init_islands(self) -> list[_Island]:
        islands = []
        for ind in self.INDICATOR_NAMES:
            island             = _Island(
                indicator        = ind,
                problem          = self.problem,
                micro_pop_size   = self.micro_size,
                archive_max_size = self.pop_size,
                f_mig            = self.f_mig,
                riesz_s          = self.riesz_s,
            )
            island.n_eval_ref  = self._n_eval  # Shared evaluation counter
            islands.append(island)
        return islands

    def _populate_islands(self, islands: list[_Island]) -> None:
        """Init each island with random solutions, evaluate them, and insert into the archive."""
        for island in islands:
            island.pop_X, island.pop_F = [], []
            for _ in range(self.micro_size):
                x   = np.random.randint(0, 2, size=self.problem.n_var).astype(float)
                x_f = np.array(
                    self.problem._evaluate_multi(x.tolist(), self._n_eval[0]),
                    dtype=float,
                )
                self._n_eval[0] += 1
                island.pop_X.append(x)
                island.pop_F.append(x_f)
            for x, f in zip(island.pop_X, island.pop_F):
                island.archive.insert(x, f)

    def _migrate(self, islands: list[_Island]) -> None:
        """
        Every f_mig iterations, each island sends n_mig emigrants to all other islands. Emigrants are selected randomly from the micro-population before any modifications. Each island receives migrants from all other islands and integrates them using the receive_migrants method, which replaces the worst contributing solutions in the population
        """
        k = len(islands)
        emigrants: list[tuple[list, list]] = []
        for island in islands:
            n_send = min(self.n_mig, len(island.pop_X))
            if n_send == 0:
                emigrants.append(([], []))
                continue
            idxs   = np.random.choice(len(island.pop_X), n_send, replace=False)
            em_X   = [island.pop_X[i] for i in idxs]
            em_F   = [island.pop_F[i] for i in idxs]
            emigrants.append((em_X, em_F))

        for j, island in enumerate(islands):
            all_mig_X, all_mig_F = [], []
            for src in range(k):
                if src == j:
                    continue
                all_mig_X.extend(emigrants[src][0])
                all_mig_F.extend(emigrants[src][1])
            if all_mig_X:
                island.receive_migrants(all_mig_X, all_mig_F, self.n_mig)

    def _merge_archives(self, islands: list[_Island]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """ Combine all sub-populations and archives, filter non-dominated solutions, and prune with Riesz s-energy if needed. (lines 8-16) """
        merged_X: list[np.ndarray] = []
        merged_F: list[np.ndarray] = []

        for island in islands:
            # Sub-población
            for x, f in zip(island.pop_X, island.pop_F):
                merged_X.append(np.array(x, dtype=float))
                merged_F.append(np.array(f, dtype=float))
            a_pos, a_obj = island.archive.get_all()
            for x, f in zip(a_pos, a_obj):
                merged_X.append(np.array(x, dtype=float))
                merged_F.append(np.array(f, dtype=float))

        if len(merged_F) == 0:
            return [], []

        objs = np.array([f.tolist() for f in merged_F], dtype=float)
        nd   = _Indicators._non_dominated(objs)
        nd_X = [merged_X[i] for i in range(len(merged_X)) if nd[i]]
        nd_F = [merged_F[i] for i in range(len(merged_F)) if nd[i]]

        while len(nd_F) > self.pop_size:
            objs_nd = np.array([f.tolist() for f in nd_F], dtype=float)
            worst   = _RieszEnergy.worst_contribution(objs_nd, self.riesz_s)
            nd_X.pop(worst)
            nd_F.pop(worst)

        return nd_X, nd_F

    def _save_population(self, pop: dict, generation: int) -> None:
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
                decoded = self.problem.get_decoded_ind(
                    x.astype(int).tolist() if isinstance(x, np.ndarray) else x
                )
                writer.writerow([
                    generation, idx,
                    " ".join(map(str, decoded)),
                    float(obj[0]), int(obj[1]),
                ])

    def _do(self):
        """Executes the complete IMIA optimization process, following the structure of Algorithm 1 from the paper."""
        islands = self._init_islands()
        self._populate_islands(islands)   # line 1

        all_X0 = [x for isl in islands for x in isl.pop_X]
        all_F0 = [f for isl in islands for f in isl.pop_F]
        self._save_population({"X": all_X0, "F": all_F0}, generation=0)

        n_cycles = max(1, self.n_gen // self.f_mig)
        pbar     = tqdm(
            total=n_cycles, desc="IMIA Progress", unit="cycle"
        ) if self.verbose else None

        for cycle in range(1, n_cycles + 1):

            for island in islands:
                island.run_isolated()

            self._migrate(islands)

            all_X = [x for isl in islands for x in isl.pop_X]
            all_F = [f for isl in islands for f in isl.pop_F]
            self.problem.notify_generation(cycle)
            self._save_population(
                {"X": all_X, "F": all_F},
                generation=cycle * self.f_mig,
            )

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        nds_X, nds_F = self._merge_archives(islands)

        final_X = [x for isl in islands for x in isl.pop_X]
        final_F = [f for isl in islands for f in isl.pop_F]

        final_pop = {"X": final_X, "F": final_F}
        nds       = {"X": nds_X,   "F": nds_F}
        return final_pop, nds

    def run(self):
        return self._do()