import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class CA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.accept_ratio = 0.2

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        # Initialize Normative Knowledge (Nt)
        lower_bounds = np.full(self.dim, self.lb)
        upper_bounds = np.full(self.dim, self.ub)

        for t in range(1, self.max_iter):
            # Acceptance function: select top performing individuals
            sort_idx = np.argsort(fitness)
            num_accepted = max(1, int(self.pop_size * self.accept_ratio))
            accepted_pop = population[sort_idx[:num_accepted]]
            
            # Update Normative Knowledge
            for j in range(self.dim):
                min_val = np.min(accepted_pop[:, j])
                max_val = np.max(accepted_pop[:, j])
                
                # Expand or contract normative bounds conservatively
                if min_val < lower_bounds[j]:
                    lower_bounds[j] = min_val
                if max_val > upper_bounds[j]:
                    upper_bounds[j] = max_val
                    
            # Influence Function
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                for j in range(self.dim):
                    size = upper_bounds[j] - lower_bounds[j]
                    if size == 0:
                        size = 1e-5
                        
                    # Hybrid influence: uses normative step size and situational direction
                    if np.random.rand() < 0.5:
                        direction = 1 if population[i, j] < self.global_best_pos[j] else -1
                        new_population[i, j] = population[i, j] + direction * np.abs(np.random.randn()) * size
                    else:
                        new_population[i, j] = population[i, j] + np.random.randn() * size
                        
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
