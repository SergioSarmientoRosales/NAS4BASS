from search.base import BaseSearch
from search.common import (
    BudgetedSearchMixin,
    random_binary_individual,
)
from search.progress import tqdm


class RandomSearch(BudgetedSearchMixin, BaseSearch):
    def __init__(self, *args, seed: int | None = None, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        self._init_budgeted_search()

    def run(self):
        pop = {"X": [], "F": []}
        records = []
        seen = set()
        budget = self.max_evals

        pbar = tqdm(total=budget, desc="Random Search", unit="eval") if self.verbose else None

        for i in range(budget):
            ind = random_binary_individual(self.rng, self.problem.n_var)
            key = tuple(ind)
            retry = 0
            while key in seen and retry < 32:
                ind = random_binary_individual(self.rng, self.problem.n_var)
                key = tuple(ind)
                retry += 1
            seen.add(key)

            obj, record = self._evaluate_candidate(
                ind,
                iteration=0,
                candidate_id=i,
            )
            pop["X"].append(ind)
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
