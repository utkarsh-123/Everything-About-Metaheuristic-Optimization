import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class FA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30, 
                 alpha=0.5, beta0=1.0, gamma=1.0, delta=0.97):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.delta = delta

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        current_alpha = self.alpha

        for t in range(self.max_iter - 1):
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                moved = False
                for j in range(self.pop_size):
                    if fitness[j] < fitness[i]: # Minimization
                        r = np.linalg.norm(population[i] - population[j])
                        beta = self.beta0 * np.exp(-self.gamma * (r ** 2))
                        
                        # Random movement vector
                        rand_vec = np.random.uniform(-0.5, 0.5, self.dim)
                        
                        new_pos = population[i] + beta * (population[j] - population[i]) + current_alpha * rand_vec
                        new_population[i] = new_pos
                        moved = True
                
                if not moved:
                    # Move randomly if no one is brighter
                    rand_vec = np.random.uniform(-0.5, 0.5, self.dim)
                    new_population[i] = population[i] + current_alpha * rand_vec
                    
            # Enforce bounds
            new_population = self.enforce_bounds(new_population)
            
            # Evaluate
            new_fitness = self.evaluate(new_population)
            
            # Update population (Firefly updates unconditionally, but we can preserve bests or just replace)
            population = new_population
            fitness = new_fitness
            
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
            # Cool down alpha
            current_alpha = current_alpha * self.delta
            
        return self.global_best_pos, self.global_best_score
