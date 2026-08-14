import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class FDO(BaseAlgorithm):
    """
    Fitness Dependent Optimizer (FDO)
    Inspired by the swarming behaviour of bees.
    Explores and exploits based on fitness-weighted position updates.
    """
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            # Compute fitness weights (inverse, for minimisation)
            min_f = np.min(fitness)
            max_f = np.max(fitness)
            if max_f == min_f:
                weights = np.ones(self.pop_size)
            else:
                weights = (max_f - fitness) / (max_f - min_f + 1e-10)
            weights /= (np.sum(weights) + 1e-10)

            # Weighted centre (fitness-dependent centre of mass)
            X_c = np.sum(weights[:, np.newaxis] * population, axis=0)

            new_population = np.copy(population)
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                # Move toward fitness-weighted centre and global best
                step = r1 * (X_c - population[i]) + r2 * (self.global_best_pos - population[i])
                new_pos = population[i] + step

                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)

                if new_fit < fitness[i]:
                    new_population[i] = new_pos
                    fitness[i] = new_fit

            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)

        return self.global_best_pos, self.global_best_score
