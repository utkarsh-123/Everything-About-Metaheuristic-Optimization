import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class LSA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            # Sort population
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # Transition projectile (worst), Space projectile (middle), Lead projectile (best)
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                if i == 0:
                    # Lead Projectile: Normal Distribution
                    new_pos = np.random.normal(population[i], np.abs(self.global_best_pos - population[i]) + 1e-10)
                elif i < self.pop_size // 2:
                    # Space Projectile: Exponential Distribution
                    lam = 1.0 / (np.abs(population[i] - self.global_best_pos) + 1e-10)
                    new_pos = population[i] + np.random.exponential(1/lam) * np.sign(np.random.randn(self.dim))
                else:
                    # Transition Projectile: Uniform Distribution
                    new_pos = population[i] + np.random.uniform(-1, 1, self.dim) * (self.ub - self.lb)
                    
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)
                
                if new_fit < fitness[i]:
                    new_population[i] = new_pos
                    fitness[i] = new_fit
                    
            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
