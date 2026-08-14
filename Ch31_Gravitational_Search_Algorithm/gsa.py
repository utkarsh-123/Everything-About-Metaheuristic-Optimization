import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class GSA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.G0 = 100.0
        self.alpha = 20.0

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        velocity = np.zeros((self.pop_size, self.dim))

        for t in range(1, self.max_iter):
            best_fit = np.min(fitness)
            worst_fit = np.max(fitness)
            
            # Calculate Mass
            if best_fit == worst_fit:
                M = np.ones(self.pop_size) / self.pop_size
            else:
                m = (fitness - worst_fit) / (best_fit - worst_fit)
                M = m / np.sum(m)
                
            G = self.G0 * np.exp(-self.alpha * t / self.max_iter)
            
            # Update Kbest (exploration vs exploitation)
            Kbest = int(self.pop_size * (1 - t / self.max_iter)) + 1
            sort_idx = np.argsort(fitness)
            kbest_indices = sort_idx[:Kbest]
            
            acceleration = np.zeros((self.pop_size, self.dim))
            
            for i in range(self.pop_size):
                force = np.zeros(self.dim)
                for j in kbest_indices:
                    if i != j:
                        R = np.linalg.norm(population[i] - population[j])
                        r_vec = np.random.rand(self.dim)
                        force += r_vec * G * (M[i] * M[j]) / (R + 1e-10) * (population[j] - population[i])
                acceleration[i] = force / (M[i] + 1e-10)
                
            velocity = np.random.rand(self.pop_size, self.dim) * velocity + acceleration
            population = population + velocity
            
            population = self.enforce_bounds(population)
            fitness = self.evaluate(population)

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
