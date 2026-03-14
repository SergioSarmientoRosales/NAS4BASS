from __future__ import annotations


class BaseSearch:
    def __init__(
        self,
        problem,
        pop_size: int,
        n_gen: int,
        verbose: bool = False,
        output_file: str | None = None,
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.verbose = verbose
        self.output_file = output_file

    def run(self):
        raise NotImplementedError

    def __call__(self):
        return self.run()