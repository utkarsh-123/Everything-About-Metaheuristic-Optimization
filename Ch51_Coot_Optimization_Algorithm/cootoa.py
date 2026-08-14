import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class CootOA(BaseAlgorithm):
    """
    Coot Optimization Algorithm (CootOA)
    Inspired by the swimming and flying behaviour of coot birds.
    Chain movement (followers follow leaders) + random walk + flight.
    """
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]

            new_population = np.copy(population)

            for i in range(self.pop_size):
                if i == 0:
                    # Leader: moves toward global best
                    r1 = np.random.rand()
                    new_pos = population[i] + r1 * (self.global_best_pos - population[i])
                else:
                    # Follower: chain movement
                    r2 = np.random.rand()
                    new_pos = population[i] + r2 * (population[i - 1] - population[i])

                # Random walk (swimming perturbation)
                R_walk = (2 * np.random.rand(self.dim) - 1) * 0.05 * (self.ub - self.lb)
                new_pos += R_walk

                # Occasional flight (large random displacement)
                if np.random.rand() < 0.1:
                    new_pos = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)

                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)

                if new_fit < fitness[i]:
                    new_population[i] = new_pos
                    fitness[i] = new_fit

            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)

        return self.global_best_pos, self.global_best_score
