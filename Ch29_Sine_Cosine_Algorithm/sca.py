import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class SCA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.a = 2.0

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            r1 = self.a - t * (self.a / self.max_iter)
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                for j in range(self.dim):
                    r2 = 2 * np.pi * np.random.rand()
                    r3 = 2 * np.random.rand()
                    r4 = np.random.rand()
                    
                    if r4 < 0.5:
                        new_population[i, j] = population[i, j] + r1 * np.sin(r2) * np.abs(r3 * self.global_best_pos[j] - population[i, j])
                    else:
                        new_population[i, j] = population[i, j] + r1 * np.cos(r2) * np.abs(r3 * self.global_best_pos[j] - population[i, j])
                        
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
