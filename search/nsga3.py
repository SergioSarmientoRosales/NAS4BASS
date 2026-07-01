import copy
import os
import csv
import numpy as np
from tqdm import tqdm

from search.operators import TournamentSelection, KPointBinaryCrossover, BitFlipMutation
from search.base import BaseSearch


class ReferencePoint:
    def __init__(self, position):
        self.position = position


class NSGA3(BaseSearch):
    def __init__(
            self,
            pop_size=100,
            n_gen=None,
            problem=None,
            verbose=False,
            output_file=None,
            save_flush_every: int = 100,
            early_stop: bool = False,
            early_stop_min_gen: int = 200,
            early_stop_repeat_patience: int = 10,
    ):
        super().__init__(
            problem=problem,
            pop_size=pop_size,
            n_gen=n_gen,
            verbose=verbose,
            output_file=output_file,
            save_flush_every=save_flush_every,
            early_stop=early_stop,
            early_stop_min_gen=early_stop_min_gen,
            early_stop_repeat_patience=early_stop_repeat_patience,
        )
        self.n_eval = 1
        self._history_buffer = []
        self._header_written = False

        self._last_population_signature = None
        self._same_population_count = 0

    @staticmethod
    def _dominate(p, q):
        return all(p_i < q_i for p_i, q_i in zip(p, q))

    def _fast_non_dominated_sorting(self, pop):
        f1 = []
        fronts = {}
        sp = {}
        nq = {}

        for i in range(len(pop["F"])):
            p = pop["F"][i]
            s = []
            n = 0
            for j in range(len(pop["F"])):
                q = pop["F"][j]
                if self._dominate(p, q):
                    s.append(j)
                elif self._dominate(q, p):
                    n += 1
            sp[f"p_{i}"] = s
            nq[f"q_{i}"] = n
            if n == 0:
                f1.append(i)

        k = 1
        fronts[f"F{k}"] = f1
        while fronts[f"F{k}"] != []:
            next_front = []
            for i in fronts[f"F{k}"]:
                for j in sp[f"p_{i}"]:
                    nq[f"q_{j}"] -= 1
                    if nq[f"q_{j}"] == 0:
                        next_front.append(j)
            k += 1
            fronts[f"F{k}"] = next_front

        return fronts

    def _initialize_pop(self):
        x, x_f = [], []
        for _ in range(self.pop_size):
            ind = np.random.randint(0, 2, size=self.problem.n_var).tolist()
            x.append(ind)
            x_f.append(self.problem._evaluate_multi(ind, self.n_eval))
            self.n_eval += 1
        return {"X": x, "F": x_f}

    def _new_individual(self, pop):
        parents = TournamentSelection(n_parents=2)(pop=pop)
        offspring = KPointBinaryCrossover(problem=self.problem)(parents, pop)
        x = BitFlipMutation(problem=self.problem, prob=1 / self.problem.n_var)(offspring)
        x_f = self.problem._evaluate_multi(x.tolist(), self.n_eval)
        self.n_eval += 1
        return {"X": [x.tolist()], "F": [x_f]}

    def _generate_q(self, pop):
        x = []
        x_f = []
        for _ in range(self.pop_size):
            q = self._new_individual(pop)
            x.append(q["X"][0])
            x_f.append(q["F"][0])
        return {"X": x, "F": x_f}

    def generate_reference_points(self, num_objs, num_divisions_per_obj=4):
        def gen_refs_recursive(work_point, num_objs, left, total, depth):
            if depth == num_objs - 1:
                work_point[depth] = left / total
                ref = ReferencePoint(copy.deepcopy(work_point))
                return [ref]
            else:
                res = []
                for i in range(left + 1):
                    work_point[depth] = i / total
                    res = res + gen_refs_recursive(work_point, num_objs, left - i, total, depth + 1)
                return res

        return gen_refs_recursive(
            [0] * num_objs,
            num_objs,
            num_objs * num_divisions_per_obj,
            num_objs * num_divisions_per_obj,
            0,
        )

    def _weights_vector(self):
        reference_points = self.generate_reference_points(num_objs=self.problem.n_obj)
        return {f"w{i}": ref.position for i, ref in enumerate(reference_points)}

    def _normalize(self, s, pop):
        min_obj = [np.inf] * self.problem.n_obj
        max_obj = [-np.inf] * self.problem.n_obj

        valid_s = [index for index in s if 0 <= index < len(pop)]

        for index in valid_s:
            for obj in range(self.problem.n_obj):
                if obj < len(pop[index]):
                    if pop[index][obj] < min_obj[obj]:
                        min_obj[obj] = pop[index][obj]
                    if pop[index][obj] > max_obj[obj]:
                        max_obj[obj] = pop[index][obj]

        normalized = []
        for index in valid_s:
            ind = [
                (pop[index][obj] - min_obj[obj]) / (max_obj[obj] - min_obj[obj])
                if max_obj[obj] != min_obj[obj] else 0.0
                for obj in range(self.problem.n_obj)
            ]
            ind.append(index)
            normalized.append(ind)

        return normalized

    def perpendicular_distance(self, direction, point):
        if len(direction) != len(point):
            return np.inf

        direction = np.asarray(direction, dtype=np.float64)
        point = np.asarray(point, dtype=np.float64)

        denom = np.sum(np.power(direction, 2))
        if denom == 0:
            return np.inf

        k = np.dot(direction, point) / denom
        d = np.sum(np.power(np.subtract(direction * k, point), 2))
        return np.sqrt(d)

    def _associate(self, norm):
        a = {}
        for index in range(len(norm)):
            ind = norm[index][:self.problem.n_obj]
            d_min = np.inf
            w_min = None

            for w_index in range(len(self.ref_points)):
                w = self.ref_points[f"w{w_index}"]
                d = self.perpendicular_distance(w, ind)
                if d < d_min:
                    d_min = d
                    w_min = f"w{w_index}"

            if w_min is None:
                raise RuntimeError(f"Failed to associate normalized point: {ind}")

            a[f"{norm[index][self.problem.n_obj]}"] = [d_min, w_min]

        return a

    def _niching(self, l_front, niche, a, pop_index):
        k = self.pop_size - len(pop_index)
        count = 0
        remaining = list(l_front)

        while count < k and remaining:
            l_ref = list(np.unique([a[f"{index}"][1] for index in remaining if f"{index}" in a]))
            if not l_ref:
                break

            empty_ref = [key for key in niche.keys() if niche[key] == 0]
            candidate_refs = [w for w in empty_ref if w in l_ref] if empty_ref else []

            if not candidate_refs:
                candidate_refs = l_ref

            selected = None

            for w in candidate_refs:
                candidates = [i for i in remaining if f"{i}" in a and a[f"{i}"][1] == w]
                if not candidates:
                    continue

                selected = min(candidates, key=lambda i: a[f"{i}"][0])
                pop_index.append(selected)
                remaining.remove(selected)
                niche[w] += 1
                count += 1
                break

            if selected is None:
                fallback_candidates = [i for i in remaining if f"{i}" in a]
                if not fallback_candidates:
                    break

                selected = fallback_candidates[0]
                ref_key = a[f"{selected}"][1]

                pop_index.append(selected)
                remaining.remove(selected)
                if ref_key in niche:
                    niche[ref_key] += 1
                count += 1

        return pop_index

    def _non_dominated_samples(self, front):
        indexes = []
        for i in range(len(front["F"])):
            p = front["F"][i]
            n = 0
            for j in range(len(front["F"])):
                q = front["F"][j]
                if self._dominate(q, p):
                    n += 1
            if n == 0:
                indexes.append(i)
        return indexes

    def _population_signature(self, pop):
        return tuple(
            sorted(tuple(self.problem.get_decoded_ind(x)) for x in pop["X"])
        )

    def _append_population_to_buffer(self, pop, generation: int):
        if not self.output_file:
            return

        for idx, (x, obj) in enumerate(zip(pop["X"], pop["F"])):
            decoded_arch = self.problem.get_decoded_ind(x)
            self._history_buffer.append([
                generation,
                idx,
                " ".join(map(str, decoded_arch)),
                float(obj[0]),
                int(obj[1]),
            ])

    def _flush_population_buffer(self):
        if not self.output_file or not self._history_buffer:
            return

        file_exists = os.path.exists(self.output_file)

        with open(self.output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists or not self._header_written:
                writer.writerow([
                    "generation",
                    "individual_id",
                    "decoded_architecture",
                    "predicted_psnr",
                    "params",
                ])
                self._header_written = True

            writer.writerows(self._history_buffer)

        self._history_buffer = []

    def _do(self):
        c = 0
        nds = {}
        self.ref_points = self._weights_vector()
        pop = self._initialize_pop()
        self._append_population_to_buffer(pop, generation=0)

        pbar = tqdm(total=self.n_gen, desc="NSGA-III Progress", unit="gen") if self.verbose else None

        while self.n_gen is None or c < self.n_gen:
            q_pop = self._generate_q(pop)

            r_pop = {}
            for key in pop.keys():
                r_pop[key] = pop[key] + q_pop[key]

            fronts = self._fast_non_dominated_sorting(r_pop)

            s, f = [], 0
            while len(s) < self.pop_size:
                f += 1
                for x in fronts[f"F{f}"]:
                    s.append(x)

            if len(s) == self.pop_size:
                for key in r_pop.keys():
                    pop[key] = [r_pop[key][index] for index in s]
            else:
                pop_index = [item for j in range(1, f) for item in fronts[f"F{j}"]]
                last_front = fronts[f"F{f}"].copy()

                normal = self._normalize(s, r_pop["F"])
                a = self._associate(normal)

                niche_c = {key: 0 for key in self.ref_points}
                for index in pop_index:
                    niche_c[a[f"{index}"][1]] += 1

                pop_index = self._niching(last_front, niche_c, a, pop_index)
                pop_index = [i for i in pop_index if i is not None]

                if len(pop_index) != self.pop_size:
                    raise RuntimeError(
                        f"Invalid pop_index size after niching: got {len(pop_index)}, expected {self.pop_size}"
                    )

                for key in pop.keys():
                    pop[key] = [r_pop[key][i] for i in pop_index]

            nds_index = self._non_dominated_samples(pop)
            for key in pop.keys():
                nds[key] = [pop[key][index] for index in nds_index]

            c += 1
            self._append_population_to_buffer(pop, generation=c)

            if self.save_flush_every > 0 and c % self.save_flush_every == 0:
                self._flush_population_buffer()

            if self.early_stop and c >= self.early_stop_min_gen:
                current_signature = self._population_signature(pop)

                if current_signature == self._last_population_signature:
                    self._same_population_count += 1
                else:
                    self._same_population_count = 0

                self._last_population_signature = current_signature

                if self._same_population_count >= self.early_stop_repeat_patience:
                    print(
                        f"[INFO] Early stop triggered at generation {c} "
                        f"after {self.early_stop_repeat_patience} consecutive identical populations."
                    )
                    break

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self._flush_population_buffer()

        return pop, nds

    def run(self):
        return self._do()