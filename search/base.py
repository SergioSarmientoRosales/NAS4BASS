from __future__ import annotations


class BaseSearch:
    def __init__(
        self,
        problem,
        pop_size: int,
        n_gen: int | None,
        verbose: bool = False,
        output_file: str | None = None,
        save_flush_every: int = 100,
        early_stop: bool = False,
        early_stop_min_gen: int = 200,
        early_stop_repeat_patience: int = 10,
    ):
        self.problem = problem
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.verbose = verbose
        self.output_file = output_file
        self.save_flush_every = save_flush_every

        self.early_stop = early_stop
        self.early_stop_min_gen = early_stop_min_gen
        self.early_stop_repeat_patience = early_stop_repeat_patience

    def run(self):
        raise NotImplementedError

    def __call__(self):
        return self.run()