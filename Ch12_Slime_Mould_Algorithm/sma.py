import numpy as np
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class SMA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.z = 0.03

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            # calculate a and vb, vc
            a = math.atanh(-(t / self.max_iter) + 1) if t < self.max_iter else 0
            bF = np.min(fitness)
            wF = np.max(fitness)
            
            W = np.zeros(self.pop_size)
            sort_idx = np.argsort(fitness)
            
            # calculate W
            for i in range(self.pop_size):
                idx = sort_idx[i]
                if i < self.pop_size / 2:
                    W[idx] = 1 + np.random.rand() * np.log10((bF - fitness[idx]) / (bF - wF + 1e-10) + 1)
                else:
                    W[idx] = 1 - np.random.rand() * np.log10((bF - fitness[idx]) / (bF - wF + 1e-10) + 1)
                    
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                if np.random.rand() < self.z:
                    new_population[i] = np.random.uniform(self.lb, self.ub, self.dim)
                else:
                    p = np.tanh(np.abs(fitness[i] - self.global_best_score))
                    vb = np.random.uniform(-a, a, self.dim)
                    vc = np.random.uniform(-1, 1, self.dim)
                    
                    A = np.random.randint(0, self.pop_size)
                    B = np.random.randint(0, self.pop_size)
                    
                    r = np.random.rand()
                    if r < p:
                        new_population[i] = self.global_best_pos + vb * (W[i] * population[A] - population[B])
                    else:
                        new_population[i] = vc * population[i]
                        
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
