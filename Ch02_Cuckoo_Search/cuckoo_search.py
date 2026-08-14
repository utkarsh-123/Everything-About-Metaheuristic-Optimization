import numpy as np
import math
import sys
import os

# Add parent directory to path to import base_algorithm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class CuckooSearch(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30, pa=0.25, alpha=1.0, lam=1.5):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.pa = pa
        self.alpha = alpha
        self.lam = lam

    def _levy_flight(self):
        sigma_u = (math.gamma(1 + self.lam) * math.sin(math.pi * self.lam / 2) / 
                   (math.gamma((1 + self.lam) / 2) * self.lam * 2 ** ((self.lam - 1) / 2))) ** (1 / self.lam)
        u = np.random.normal(0, sigma_u, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / self.lam))
        return step

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(self.max_iter - 1):
            new_population = np.copy(population)
            for i in range(self.pop_size):
                step = self._levy_flight()
                # alpha * Levy(lambda) * (x_i - x_best)
                step_size = self.alpha * step * (population[i] - self.global_best_pos)
                new_population[i] = population[i] + step_size
            
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)

            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            num_abandon = int(self.pa * self.pop_size)
            if num_abandon > 0:
                new_nests = np.random.uniform(self.lb, self.ub, (num_abandon, self.dim))
                population[-num_abandon:] = new_nests
                fitness[-num_abandon:] = self.evaluate(new_nests)

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
