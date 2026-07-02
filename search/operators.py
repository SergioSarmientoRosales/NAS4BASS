import math
import numpy as np


def _finite_objectives(values):
    sanitized = []
    has_invalid = False

    for value in values:
        if np.isfinite(value):
            sanitized.append(float(value))
        else:
            sanitized.append(np.inf)
            has_invalid = True

    return [np.inf] * len(sanitized) if has_invalid else sanitized


def dominance(a_f, b_f):
    a_better = False
    b_better = False

    for a, b in zip(_finite_objectives(a_f), _finite_objectives(b_f)):
        if a < b:
            a_better = True
        elif b < a:
            b_better = True

        if a_better and b_better:
            return 0

    if a_better:
        return 1
    elif b_better:
        return -1
    else:
        return 0


class TournamentSelection:
    def __init__(self, n_select=1, n_parents=1, pressure=2):
        self.n_select = n_select
        self.n_parents = n_parents
        self.pressure = pressure
        self.n_random = n_select * n_parents * pressure

    def random_permutations(self, length, concat=True):
        p = []
        for _ in range(self.n_perms):
            p.append(np.random.permutation(length))
        return np.concatenate(p) if concat else p

    def _do(self, pop):
        self.n_perms = math.ceil(self.n_random / len(pop["X"]))
        p = self.random_permutations(len(pop["X"]))[:self.n_random]
        p = np.reshape(p, (self.n_select * self.n_parents, self.pressure))

        n_tournaments, _ = p.shape
        s = np.full(n_tournaments, np.nan)

        for i in range(n_tournaments):
            a, b = p[i, 0], p[i, 1]
            a_f, b_f = pop["F"][a], pop["F"][b]

            rel = dominance(a_f, b_f)
            if rel == 1:
                s[i] = a
            elif rel == -1:
                s[i] = b

            if np.isnan(s[i]):
                s[i] = np.random.choice([a, b])

        return s[:, None].astype(int, copy=False)

    def __call__(self, pop):
        return self._do(pop)


class KPointBinaryCrossover:
    def __init__(self, problem, k=2, prob=0.9):
        self.problem = problem
        self.k = k
        self.prob = prob

    def _crossover_points(self):
        points = np.random.choice(range(1, self.problem.n_var), self.k, replace=False)
        return sorted(points)

    def _repair(self, x):
        for i in range(len(x)):
            if x[i] < self.problem.xl[i]:
                x[i] = self.problem.xl[i]
            elif x[i] > self.problem.xu[i]:
                x[i] = self.problem.xu[i]
        return x

    def _do(self, i_par, pop):
        if np.random.rand() <= self.prob:
            parent1 = pop["X"][i_par[0][0]]
            parent2 = pop["X"][i_par[1][0]]
            off = np.full(self.problem.n_var, np.nan)

            crossover_points = self._crossover_points()
            crossover_points.append(self.problem.n_var)

            start = 0
            for i, point in enumerate(crossover_points):
                off[start:point] = parent1[start:point] if i % 2 == 0 else parent2[start:point]
                start = point

            off = self._repair(off)
        else:
            off = pop["X"][np.random.choice([i_par[0][0], i_par[1][0]])]

        return np.asarray(off, dtype=int)

    def __call__(self, parents, pop):
        return self._do(i_par=parents, pop=pop)


class BitFlipMutation:
    def __init__(self, problem, prob=0.1):
        self.problem = problem
        self.prob = prob

    def _flip(self):
        return np.random.rand(self.problem.n_var) < self.prob

    def _do(self, parent):
        flip_decision = self._flip()
        off = np.copy(parent)

        for i in range(self.problem.n_var):
            if flip_decision[i]:
                off[i] = 1 if off[i] == 0 else 0

        return np.asarray(off, dtype=int)

    def __call__(self, parent):
        return self._do(parent)
