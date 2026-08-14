import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class DE(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.F = 0.5
        self.CR = 0.9

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                r1, r2, r3 = np.random.choice(indices, 3, replace=False)
                
                # Mutation
                v_i = population[r3] + self.F * (population[r1] - population[r2])
                
                # Recombination
                j_rand = np.random.randint(0, self.dim)
                u_i = np.copy(population[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR or j == j_rand:
                        u_i[j] = v_i[j]
                        
                new_population[i] = u_i
                
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] <= fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
