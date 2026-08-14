import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class IGMM(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        
    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        T = np.full(self.pop_size, 273.0)
        k_b = 1.0 / self.pop_size

        for t in range(1, self.max_iter):
            # calculate masses
            mass = 1.0 / (fitness + 1e-10)
            
            MCP = 1 - np.exp(-5 * (1 - t / self.max_iter))
            beta = 1 - (t / self.max_iter)
            
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                r = np.random.rand()
                if r < MCP:
                    # Collision
                    j = np.random.randint(0, self.pop_size)
                    while j == i:
                        j = np.random.randint(0, self.pop_size)
                        
                    if mass[i] < mass[j]:
                        m1, m2 = mass[i], mass[j]
                        x1, x2 = population[i], population[j]
                        idx1, idx2 = i, j
                    else:
                        m1, m2 = mass[j], mass[i]
                        x1, x2 = population[j], population[i]
                        idx1, idx2 = j, i
                        
                    v1 = x2 - x1
                    
                    v1_prime = v1 * (m1 - beta * m2) / (m1 + m2)
                    v2_prime = v1 * (1 + beta) * m1 / (m1 + m2)
                    
                    new_population[idx1] = x2 + np.random.rand(self.dim) * v1_prime
                    new_population[idx2] = x2 + np.random.rand(self.dim) * v2_prime
                else:
                    # No collision
                    T[i] = T[i] - 273.0 / self.max_iter
                    if T[i] <= 0:
                        T[i] = 0.001
                        
                    v_i = np.sqrt(3 * k_b * T[i] / mass[i])
                    new_population[i] = population[i] + v_i * np.random.randn(self.dim)
                    
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
