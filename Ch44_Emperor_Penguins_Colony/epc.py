import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class EPC(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            T_eff = 1 - t / self.max_iter # temperature profile
            
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                # Calculate distance to huddle center (global best)
                D_ep = np.abs(self.global_best_pos - population[i])
                
                # Attract to center and random walk
                A = 2 * T_eff * np.random.rand(self.dim) - T_eff
                C = 2 * np.random.rand(self.dim)
                
                new_pos = self.global_best_pos - A * D_ep
                
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)
                
                if new_fit < fitness[i]:
                    new_population[i] = new_pos
                    fitness[i] = new_fit

            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
