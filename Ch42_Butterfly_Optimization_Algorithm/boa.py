import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class BOA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.c = 0.01
        self.a = 0.1
        self.p = 0.8

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            # Update a
            a = self.a + (0.3 - self.a) * (t / self.max_iter)
            
            new_population = np.copy(population)
            
            # Intensity is mapped to fitness
            max_fit = np.max(fitness)
            I = max_fit - fitness + 1e-10
            
            for i in range(self.pop_size):
                fragrance = self.c * (I[i] ** a)
                r = np.random.rand()
                
                if r < self.p:
                    # Move towards global best
                    new_pos = population[i] + (np.random.rand()**2 * self.global_best_pos - population[i]) * fragrance
                else:
                    # Move randomly
                    j, k = np.random.randint(0, self.pop_size, 2)
                    new_pos = population[i] + (np.random.rand()**2 * population[j] - population[k]) * fragrance
                    
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)
                
                if new_fit < fitness[i]:
                    new_population[i] = new_pos
                    fitness[i] = new_fit

            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
