import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class LGSI(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                dice = np.random.randint(1, 7) # 1 to 6
                
                if dice == 6:
                    # Move towards global best with large step
                    step = np.random.rand(self.dim) * (self.global_best_pos - population[i])
                else:
                    # Random exploration or move towards a better random solution
                    j = np.random.randint(0, self.pop_size)
                    if fitness[j] < fitness[i]:
                        step = (dice / 6.0) * np.random.rand(self.dim) * (population[j] - population[i])
                    else:
                        step = (dice / 6.0) * np.random.uniform(-1, 1, self.dim) * (self.ub - self.lb) * (1 - t/self.max_iter)
                        
                new_pos = population[i] + step
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)
                
                if new_fit < fitness[i]:
                    new_population[i] = new_pos
                    fitness[i] = new_fit

            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
