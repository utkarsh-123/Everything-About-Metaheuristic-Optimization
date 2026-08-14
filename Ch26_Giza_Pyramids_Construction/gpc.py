import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class GPC(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        g = 9.8

        for t in range(1, self.max_iter):
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                v0 = np.random.rand()
                mu_k = np.random.rand()
                theta_deg = np.random.uniform(0, 15)
                theta = np.radians(theta_deg)
                
                # Calculate displacement
                d = (v0 ** 2) / (2 * g * (np.sin(theta) + mu_k * np.cos(theta)) + 1e-10)
                x_w = (v0 ** 2) / (2 * g * np.sin(theta) + 1e-10)
                
                # Random vector R
                R = np.random.randn(self.dim)
                
                new_pos = population[i] + d + x_w + R
                
                # Crossover
                if np.random.rand() < 0.5:
                    other_idx = np.random.randint(0, self.pop_size)
                    crossover_mask = np.random.rand(self.dim) < 0.5
                    new_pos = np.where(crossover_mask, new_pos, population[other_idx])
                    
                new_population[i] = new_pos
                
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]: # minimizing fitness cost
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
