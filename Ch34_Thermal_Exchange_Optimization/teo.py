import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class TEO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        TM = np.copy(population) # Thermal Memory
        TM_fit = np.copy(fitness)

        for t in range(1, self.max_iter):
            sort_idx = np.argsort(TM_fit)
            TM = TM[sort_idx]
            TM_fit = TM_fit[sort_idx]
            
            new_population = np.copy(population)
            
            c = t / self.max_iter
            
            for i in range(self.pop_size):
                if np.random.rand() < 0.5:
                    # Exchange with TM
                    j = np.random.randint(0, self.pop_size)
                    beta = np.random.rand()
                    new_pos = population[i] + beta * c * (TM[j] - population[i])
                else:
                    # Exchange with global best
                    beta = np.random.rand()
                    new_pos = population[i] + beta * c * (self.global_best_pos - population[i])
                    
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)
                
                new_population[i] = new_pos
                fitness[i] = new_fit
                
                # Update TM
                worst_TM_idx = np.argmax(TM_fit)
                if new_fit < TM_fit[worst_TM_idx]:
                    TM[worst_TM_idx] = new_pos
                    TM_fit[worst_TM_idx] = new_fit

            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
