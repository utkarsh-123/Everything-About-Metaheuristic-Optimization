import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class EPO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            # Similar behavior to EPC but with polygon boundaries and temperature
            T = (t > self.max_iter // 2) # Temperature phase flag
            
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                if T == 0:
                    # Exploring
                    A = 2 * (1 - t/self.max_iter) * np.random.rand(self.dim) - (1 - t/self.max_iter)
                    C = 2 * np.random.rand(self.dim)
                    j = np.random.randint(0, self.pop_size)
                    D = np.abs(C * population[j] - population[i])
                    new_pos = population[j] - A * D
                else:
                    # Exploiting
                    A = 2 * (t/self.max_iter) * np.random.rand(self.dim) - (t/self.max_iter)
                    C = 2 * np.random.rand(self.dim)
                    D = np.abs(C * self.global_best_pos - population[i])
                    new_pos = self.global_best_pos - A * D
                    
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)
                
                if new_fit < fitness[i]:
                    new_population[i] = new_pos
                    fitness[i] = new_fit

            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
