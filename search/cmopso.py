from __future__ import annotations
import os
import csv
import numpy as np
from tqdm import tqdm
from search.base import BaseSearch


class CMOPSO(BaseSearch):
    """
    CMOPSO (Competitive Multi-Objective Particle Swarm Optimization) 
    implementation adapted for binary search spaces.
    """

    def __init__(
        self,
        problem,
        pop_size: int = 20,
        n_gen: int = 1250,
        verbose: bool = False,
        output_file: str | None = None,
        v_max: float = 6.0,          # FIX 4: rango sigmoid más útil para binario
    ):
        super().__init__(problem, pop_size, n_gen, verbose, output_file)
        self.v_max = v_max
        self.n_eval = 1
        # Set elite set size (gamma) as 30% of the population
        self.gamma = max(2, int(0.3 * self.pop_size))

    def _initialize(self):
        """Initializes particles with binary positions and continuous velocities."""
        n = self.problem.n_var
        positions = np.random.randint(0, 2, size=(self.pop_size, n)).astype(float)
        velocities = np.random.uniform(-1.0, 1.0, size=(self.pop_size, n))

        objectives = []
        for i in range(self.pop_size):
            f = self.problem._evaluate_multi(positions[i].astype(int).tolist(), self.n_eval)
            objectives.append(f)
            self.n_eval += 1
        return positions, velocities, np.array(objectives)
    
    def _dominates(self, a, b):
        """Checks if objective vector 'a' dominates vector 'b'."""
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort_f0(self, objectives):
        """Standard non-dominated sorting to partition the population into fronts."""
        n = len(objectives)
        dominated = [False] * n
        for i in range(n):
            for j in range(n):
                if i != j and self._dominates(objectives[j], objectives[i]):
                    dominated[i] = True
                    break
        return [i for i in range(n) if not dominated[i]]

    def _non_dominated_sort(self, objectives):
        """NSGA-II fast non-dominated sort. Devuelve lista de frentes."""
        n = len(objectives)
        S      = [[] for _ in range(n)]
        n_dom  = [0] * n
        fronts = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(objectives[i], objectives[j]):
                    S[i].append(j)
                elif self._dominates(objectives[j], objectives[i]):
                    n_dom[i] += 1
            if n_dom[i] == 0:
                fronts[0].append(i)

        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_dom[q] -= 1
                    if n_dom[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
                i += 1
            else:
                break

        return fronts

    def _crowding_distance(self, objectives):
        """Calculates crowding distance to maintain diversity in the Pareto front."""
        n, m   = objectives.shape
        dist   = np.zeros(n)

        for j in range(m):
            order = np.argsort(objectives[:, j])
            dist[order[0]] = dist[order[-1]] = np.inf
            denom = objectives[order[-1], j] - objectives[order[0], j]
            if denom == 0:
                continue
            for k in range(1, n - 1):
                dist[order[k]] += (
                    objectives[order[k + 1], j] - objectives[order[k - 1], j]
                ) / denom

        return dist

    def _select_elite(self, positions, objectives):
        """Selects the top gamma individuals to form the elite set."""
        fronts     = self._non_dominated_sort(objectives)
        elite_idx  = []

        for front in fronts:
            if len(elite_idx) + len(front) <= self.gamma:
                elite_idx.extend(front)
                if len(elite_idx) == self.gamma:
                    break
            else:
                faltantes   = self.gamma - len(elite_idx)
                front_objs  = objectives[front]
                cd          = self._crowding_distance(front_objs)
                sorted_idx  = np.argsort(-cd)[:faltantes]
                for i in sorted_idx:
                    elite_idx.append(front[i])
                break

        return positions[elite_idx], objectives[elite_idx]   # <-- devuelve ambos

    def _competition(self, p_obj, elite_obj, elite_pos):
        """Tournament selection based on angular similarity in objective space."""
        idx_a, idx_b = np.random.choice(len(elite_obj), 2, replace=False)
        obj_a = elite_obj[idx_a]
        obj_b = elite_obj[idx_b]

        def angle(x, y):
            num = np.dot(x, y)
            den = np.linalg.norm(x) * np.linalg.norm(y) + 1e-12
            return np.arccos(np.clip(num / den, -1.0, 1.0))

        winner_idx = idx_a if angle(p_obj, obj_a) < angle(p_obj, obj_b) else idx_b
        return elite_pos[winner_idx]

    def _update_particle(self, pos, vel, winner_pos):
        """Updates velocity and maps it to binary position via sigmoid function."""
        r1, r2 = np.random.rand(self.problem.n_var), np.random.rand(self.problem.n_var)
        new_vel = r1 * vel + r2 * (winner_pos - pos)
        new_vel = np.clip(new_vel, -self.v_max, self.v_max)

        prob    = 1.0 / (1.0 + np.exp(-new_vel))
        new_pos = (np.random.rand(self.problem.n_var) < prob).astype(float)
        return new_pos, new_vel

    def _mutate(self, pos, gen):
        """Applies binary mutation with a rate that decays over generations."""
        rate = (0.50 / self.problem.n_var) * (1.0 - gen / self.n_gen)
        mask = np.random.rand(self.problem.n_var) < rate
        pos[mask] = 1.0 - pos[mask]
        return pos

    def _environmental_selection(self, all_pos, all_obj):
        """Selects the best individuals from the combined population."""
        fronts     = self._non_dominated_sort(all_obj)
        selected   = []
        for front in fronts:
            if len(selected) + len(front) <= self.pop_size:
                selected.extend(front)
                if len(selected) == self.pop_size:
                    break
            else:
                faltantes  = self.pop_size - len(selected)
                front_objs = all_obj[front]
                cd         = self._crowding_distance(front_objs)
                sorted_idx = np.argsort(-cd)[:faltantes]
                for i in sorted_idx:
                    selected.append(front[i])
                break

        return all_pos[selected], all_obj[selected]

    def _save_population(self, positions, objectives, generation):
        if not self.output_file:
            return

        file_exists = os.path.exists(self.output_file)

        with open(self.output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "generation", "individual_id",
                    "decoded_architecture", "predicted_psnr", "params"
                ])

            for i, (x, obj) in enumerate(zip(positions, objectives)):
                decoded = self.problem.get_decoded_ind(x.astype(int).tolist())
                writer.writerow([
                    generation,
                    i,
                    " ".join(map(str, decoded)),
                    float(obj[0]),
                    int(obj[1]),
                ])

    def _do(self):
        """Executes the complete optimization process."""
        positions, velocities, objectives = self._initialize()
        self._save_population(positions, objectives, 0)

        pbar = tqdm(total=self.n_gen, desc="CMOPSO", unit="gen") if self.verbose else None

        for gen in range(1, self.n_gen + 1):
            elite_pos, elite_obj = self._select_elite(positions, objectives)

            new_positions  = []
            new_velocities = []
            new_objectives = []

            for i in range(self.pop_size):
                winner_pos = self._competition(objectives[i], elite_obj, elite_pos)

                new_pos, new_vel = self._update_particle(
                    positions[i], velocities[i], winner_pos
                )
                new_pos = self._mutate(new_pos, gen)

                new_obj = np.array(
                    self.problem._evaluate_multi(
                        new_pos.astype(int).tolist(), self.n_eval
                    )
                )
                self.n_eval += 1

                new_positions.append(new_pos)
                new_velocities.append(new_vel)
                new_objectives.append(new_obj)

            new_positions  = np.array(new_positions)
            new_velocities = np.array(new_velocities)
            new_objectives = np.array(new_objectives)

            # Combine parents and children for selection
            combined_pos = np.vstack([positions,     new_positions])
            combined_obj = np.vstack([objectives,    new_objectives])
            combined_vel = np.vstack([velocities,    new_velocities])

            positions, objectives = self._environmental_selection(combined_pos, combined_obj)

            # Synchronize velocities with the selected particles
            sel_idx = []
            for p in positions:
                for k, cp in enumerate(combined_pos):
                    if np.array_equal(p, cp) and k not in sel_idx:
                        sel_idx.append(k)
                        break
            velocities = combined_vel[sel_idx]

            self.problem.notify_generation(gen)
            self._save_population(positions, objectives, gen)

            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()
        # Final non-dominated extraction for output
        nd_idx = self._non_dominated_sort_f0(objectives)

        nds = {
            "X": [positions[i].astype(int).tolist() for i in nd_idx],
            "F": [objectives[i].tolist()             for i in nd_idx],
        }

        final_pop = {
            "X": [p.astype(int).tolist() for p in positions],
            "F": [o.tolist()             for o in objectives],
        }

        return final_pop, nds

    def run(self):
        return self._do()