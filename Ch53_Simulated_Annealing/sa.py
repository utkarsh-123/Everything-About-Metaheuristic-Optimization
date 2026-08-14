import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class SA(BaseAlgorithm):
    """
    Simulated Annealing (SA)
    Single-solution trajectory-based metaheuristic.
    Accepts worse solutions with probability exp(-dE/T) to escape local optima.
    Temperature decreases geometrically each iteration.
    """
    def __init__(self, objective_func, bounds, dim, pop_size=1, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.T0 = 1000.0   # initial temperature
        self.alpha = 0.95  # cooling rate
        self.n_inner = 10  # inner loop iterations per temperature step

    def optimize(self):
        # SA is single-solution; current best for plotting at each outer iteration
        current = np.random.uniform(self.lb, self.ub, self.dim)
        current_fit = self.objective_func(current)

        self.global_best_pos = np.copy(current)
        self.global_best_score = current_fit
        self.convergence_curve.append(self.global_best_score)

        T = self.T0

        for t in range(1, self.max_iter):
            for _ in range(self.n_inner):
                # Generate neighbour by Gaussian perturbation
                neighbour = current + np.random.randn(self.dim) * 0.1 * (self.ub - self.lb)
                neighbour = np.clip(neighbour, self.lb, self.ub)
                neighbour_fit = self.objective_func(neighbour)

                dE = neighbour_fit - current_fit

                # Accept if better, or with Boltzmann probability if worse
                if dE < 0 or np.random.rand() < np.exp(-dE / (T + 1e-10)):
                    current = np.copy(neighbour)
                    current_fit = neighbour_fit

                if current_fit < self.global_best_score:
                    self.global_best_pos = np.copy(current)
                    self.global_best_score = current_fit

            # Cool down
            T *= self.alpha
            self.convergence_curve.append(self.global_best_score)

        return self.global_best_pos, self.global_best_score
