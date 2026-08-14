import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class FSA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        local_best = np.copy(population)
        local_fitness = np.copy(fitness)

        for t in range(1, self.max_iter):
            for i in range(self.pop_size):
                # Local Search
                S_L = (local_best[i] - population[i]) * np.random.rand(self.dim)
                # Global Search
                S_G = (self.global_best_pos - population[i]) * np.random.rand(self.dim)
                
                # New Position
                new_pos = population[i] + S_L + S_G
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)
                
                # Update Local Best
                if new_fit < local_fitness[i]:
                    local_best[i] = np.copy(new_pos)
                    local_fitness[i] = new_fit
                    
                # Update current population via exploration phase
                population[i] = self.global_best_pos + (self.global_best_pos - local_best[i]) * np.random.rand(self.dim)
                population[i] = np.clip(population[i], self.lb, self.ub)
                fitness[i] = self.objective_func(population[i])

            self.update_global_best(population, fitness)
            self.update_global_best(local_best, local_fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
