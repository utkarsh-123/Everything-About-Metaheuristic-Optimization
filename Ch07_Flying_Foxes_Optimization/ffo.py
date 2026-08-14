import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class FFO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.A = 0.5
        self.C = 1.5

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        RList = [] # Replacement List

        for t in range(self.max_iter - 1):
            new_population = np.copy(population)
            
            # Identify coolest and hottest
            xcool_idx = np.argmin(fitness)
            xhot_idx = np.argmax(fitness)
            xcool = population[xcool_idx]
            
            # Populate RList (keeping best unique solutions)
            if len(RList) == 0:
                RList.append(np.copy(xcool))
            else:
                is_unique = True
                for sol in RList:
                    if np.array_equal(sol, xcool):
                        is_unique = False
                        break
                if is_unique:
                    RList.append(np.copy(xcool))
                    if len(RList) > 10: # Keep only top R
                        RList.pop(0)

            for i in range(self.pop_size):
                distance = np.linalg.norm(population[i] - xcool)
                
                if distance > self.A:
                    new_population[i] = population[i] + self.C * (xcool - population[i])
                else:
                    m = np.random.randint(0, self.pop_size)
                    n = np.random.randint(0, self.pop_size)
                    j_rand = np.random.randint(0, self.dim)
                    r = np.random.rand()
                    pa = 0.5
                    
                    for j in range(self.dim):
                        if j == j_rand or np.random.rand() <= pa:
                            new_population[i, j] = population[i, j] + r * (population[m, j] - population[n, j])

            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                else:
                    # Heat exhaustion
                    if np.random.rand() < 0.1: # Threshold for extremely hot region
                        k = np.random.randint(0, len(RList))
                        population[i] = np.copy(RList[k])
                        fitness[i] = self.objective_func(population[i])
            
            # Suffocation death
            ncool = sum([1 for f in fitness if f == fitness[xcool_idx]])
            pdeath = ncool / self.pop_size
            
            for i in range(self.pop_size):
                if np.random.rand() < pdeath:
                    if np.random.rand() < 0.5:
                        # RList
                        k = np.random.randint(0, len(RList))
                        population[i] = np.copy(RList[k])
                    else:
                        # Crossover
                        xp = population[np.random.randint(0, self.pop_size)]
                        xq = population[np.random.randint(0, self.pop_size)]
                        r = np.random.rand()
                        population[i] = r * xp + (1 - r) * xq
                    
                    population[i] = np.clip(population[i], self.lb, self.ub)
                    fitness[i] = self.objective_func(population[i])

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
