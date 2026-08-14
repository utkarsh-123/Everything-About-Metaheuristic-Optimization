import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class AVOA(BaseAlgorithm):
    """
    African Vultures Optimization Algorithm (AVOA)
    Based on foraging and navigation strategies of African vultures.
    Two types of vultures; satiated vs hungry behaviour drives exploration/exploitation.
    """
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.p1 = 0.6   # probability phase 1
        self.p2 = 0.4   # probability phase 2
        self.p3 = 0.6   # probability of strategy within phase

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            sort_idx = np.argsort(fitness)
            best1 = population[sort_idx[0]]  # best vulture
            best2 = population[sort_idx[1]]  # second best vulture

            new_population = np.copy(population)

            for i in range(self.pop_size):
                # Select reference vulture
                F = (2 * np.random.rand() + 1) * (1 - t / self.max_iter) + np.random.randn() * 0.5
                # Choose between best1 and best2 probabilistically
                p = np.random.rand()
                R = best1 if p < 0.5 else best2

                if np.abs(F) >= 1:
                    # Exploration
                    r1 = np.random.rand()
                    D = np.abs(2 * np.random.rand() * R - population[i])
                    new_pos = R - F * D
                else:
                    # Exploitation
                    if np.random.rand() < self.p1:
                        # Phase 1: rotating flight
                        D = np.abs(R - population[i])
                        theta = np.random.uniform(-np.pi, np.pi, self.dim)
                        new_pos = D * np.exp(1j * theta).real * np.cos(theta) + R
                    else:
                        # Phase 2: siege
                        if np.random.rand() < self.p2:
                            L1 = np.abs(2 * np.random.rand() * self.global_best_pos - population[i])
                            L2 = np.abs(2 * np.random.rand() * R - population[i])
                            new_pos = (population[i]
                                       - (L1 + L2)
                                       + 2 * np.random.rand() * (R - population[i]))
                        else:
                            t2 = np.random.rand()
                            new_pos = (R - (np.abs(R - population[i]) * (F + np.random.rand()))
                                       * np.tanh(np.random.rand() * 2 - 1))

                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)

                if new_fit < fitness[i]:
                    new_population[i] = new_pos
                    fitness[i] = new_fit

            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)

        return self.global_best_pos, self.global_best_score
