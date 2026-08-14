import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class GOA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.c_max = 1.0
        self.c_min = 0.00001
        self.f = 0.5
        self.l = 1.5

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            c = self.c_max - t * ((self.c_max - self.c_min) / self.max_iter)
            
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                s_sum = np.zeros(self.dim)
                for j in range(self.pop_size):
                    if i != j:
                        dist = np.linalg.norm(population[j] - population[i])
                        if dist == 0: dist = 1e-10
                        # social forces
                        s_val = self.f * np.exp(-dist / self.l) - np.exp(-dist)
                        direction = (population[j] - population[i]) / dist
                        
                        s_sum += c * ((self.ub - self.lb) / 2) * s_val * direction
                        
                new_pos = c * s_sum + self.global_best_pos
                
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)
                
                if new_fit < fitness[i]:
                    new_population[i] = new_pos
                    fitness[i] = new_fit

            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
