import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class TS(BaseAlgorithm):
    """
    Tabu Search (TS)
    Single-solution trajectory-based metaheuristic.
    Maintains a tabu list to prevent revisiting recently visited solutions,
    helping escape local optima. Uses aspiration criteria to override tabu.
    """
    def __init__(self, objective_func, bounds, dim, pop_size=1, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.tabu_size = 10        # maximum tabu list length
        self.n_neighbours = 20     # neighbourhood size per iteration
        self.step_size = 0.1       # perturbation step (fraction of range)

    def optimize(self):
        current = np.random.uniform(self.lb, self.ub, self.dim)
        current_fit = self.objective_func(current)

        self.global_best_pos = np.copy(current)
        self.global_best_score = current_fit
        self.convergence_curve.append(self.global_best_score)

        tabu_list = []   # stores recent solutions as tuples for hash comparison

        for t in range(1, self.max_iter):
            step = self.step_size * (self.ub - self.lb) * (1 - 0.5 * t / self.max_iter)

            # Generate neighbourhood
            neighbours = []
            for _ in range(self.n_neighbours):
                delta = np.random.uniform(-step, step, self.dim)
                nb = np.clip(current + delta, self.lb, self.ub)
                nb_fit = self.objective_func(nb)
                neighbours.append((nb, nb_fit))

            # Sort neighbours by fitness
            neighbours.sort(key=lambda x: x[1])

            # Select best non-tabu neighbour (or override by aspiration criterion)
            moved = False
            for nb, nb_fit in neighbours:
                nb_key = tuple(np.round(nb, 4))  # discretised key for tabu list

                is_tabu = nb_key in tabu_list

                # Aspiration criterion: override tabu if this is a global improvement
                if not is_tabu or nb_fit < self.global_best_score:
                    current = np.copy(nb)
                    current_fit = nb_fit

                    # Update tabu list
                    tabu_list.append(nb_key)
                    if len(tabu_list) > self.tabu_size:
                        tabu_list.pop(0)

                    if current_fit < self.global_best_score:
                        self.global_best_pos = np.copy(current)
                        self.global_best_score = current_fit

                    moved = True
                    break

            # If all neighbours are tabu, take the least bad one
            if not moved and neighbours:
                nb, nb_fit = neighbours[0]
                current = np.copy(nb)
                current_fit = nb_fit

            self.convergence_curve.append(self.global_best_score)

        return self.global_best_pos, self.global_best_score
