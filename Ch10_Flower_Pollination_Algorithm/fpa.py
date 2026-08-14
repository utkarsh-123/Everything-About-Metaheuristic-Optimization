import numpy as np
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class FPA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.p = 0.8 # switching probability

    def _levy_flight(self):
        lam = 1.5
        sigma = (math.gamma(1 + lam) * math.sin(math.pi * lam / 2) / 
                 (math.gamma((1 + lam) / 2) * lam * 2 ** ((lam - 1) / 2))) ** (1 / lam)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / lam))
        return step

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(self.max_iter - 1):
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                if np.random.rand() < self.p:
                    # Global pollination
                    alpha = 0.1
                    L = self._levy_flight()
                    new_population[i] = population[i] + alpha * L * (self.global_best_pos - population[i])
                else:
                    # Local pollination
                    epsilon = np.random.rand()
                    j = np.random.randint(0, self.pop_size)
                    k = np.random.randint(0, self.pop_size)
                    while j == k:
                        k = np.random.randint(0, self.pop_size)
                    new_population[i] = population[i] + epsilon * (population[j] - population[k])

            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
