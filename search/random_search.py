from __future__ import annotations

import os
import csv
import numpy as np
from tqdm import tqdm

from search.base import BaseSearch


class RandomSearch(BaseSearch):
    def _save_population(self, pop, generation: int):
        if not self.output_file:
            return

        file_exists = os.path.exists(self.output_file)

        with open(self.output_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "generation",
                    "individual_id",
                    "decoded_architecture",
                    "predicted_psnr",
                    "params",
                ])

            for idx, (x, obj) in enumerate(zip(pop["X"], pop["F"])):
                decoded_arch = self.problem.get_decoded_ind(x)

                writer.writerow([
                    generation,
                    idx,
                    " ".join(map(str, decoded_arch)),
                    float(obj[0]),
                    int(obj[1]),
                ])

    @staticmethod
    def _non_dominated_samples(front):
        def dominates(p, q):
            return all(p_i <= q_i for p_i, q_i in zip(p, q)) and any(
                p_i < q_i for p_i, q_i in zip(p, q)
            )

        indexes = []
        for i in range(len(front["F"])):
            p = front["F"][i]
            dominated = False
            for j in range(len(front["F"])):
                if i == j:
                    continue
                q = front["F"][j]
                if dominates(q, p):
                    dominated = True
                    break
            if not dominated:
                indexes.append(i)
        return indexes

    def run(self):
        pop = {"X": [], "F": []}

        pbar = tqdm(total=self.pop_size, desc="Random Search", unit="ind") if self.verbose else None

        for i in range(self.pop_size):
            ind = np.random.randint(0, 2, size=self.problem.n_var).tolist()
            obj = self.problem._evaluate_multi(ind, i + 1)

            pop["X"].append(ind)
            pop["F"].append(obj)

            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self._save_population(pop, generation=0)

        nds_index = self._non_dominated_samples(pop)
        nds = {
            "X": [pop["X"][i] for i in nds_index],
            "F": [pop["F"][i] for i in nds_index],
        }

        return pop, nds