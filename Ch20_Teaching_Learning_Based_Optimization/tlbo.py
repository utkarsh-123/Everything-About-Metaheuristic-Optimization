import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class TLBO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            # Teacher Phase
            mean_learner = np.mean(population, axis=0)
            teacher = population[np.argmin(fitness)]
            
            for i in range(self.pop_size):
                TF = np.random.choice([1, 2])
                diff = np.random.rand(self.dim) * (teacher - TF * mean_learner)
                new_pos = population[i] + diff
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)
                
                if new_fit < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fit
                    
            # Learner Phase
            for i in range(self.pop_size):
                j = np.random.randint(0, self.pop_size)
                while j == i:
                    j = np.random.randint(0, self.pop_size)
                    
                if fitness[i] < fitness[j]:
                    new_pos = population[i] + np.random.rand(self.dim) * (population[i] - population[j])
                else:
                    new_pos = population[i] + np.random.rand(self.dim) * (population[j] - population[i])
                    
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)
                
                if new_fit < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fit

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
