import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class HGS(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        hungry = np.zeros(self.pop_size)
        
        for t in range(1, self.max_iter):
            # Update BF, WF
            BF = np.min(fitness)
            WF = np.max(fitness)
            
            # Update Hungry
            for i in range(self.pop_size):
                if fitness[i] == BF:
                    hungry[i] = 0
                else:
                    # simplified H
                    H = ((fitness[i] - BF) / (WF - BF + 1e-10)) * np.random.rand() * 2 * (self.ub - self.lb)
                    hungry[i] += H
                    
            SHungry = np.sum(hungry)
            if SHungry == 0:
                SHungry = 1e-10
                
            shrink = 2 * (1 - t / self.max_iter)
            
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                r1 = np.random.rand()
                r2 = np.random.rand()
                r3 = np.random.rand()
                r4 = np.random.rand()
                r5 = np.random.rand()
                l = 0.03 # small probability parameter
                
                # Calculate W1, W2
                if r3 < l:
                    W1 = hungry[i] * self.pop_size / SHungry * r4
                else:
                    W1 = 1
                    
                W2 = (1 - np.exp(-np.abs(hungry[i] - SHungry))) * r5 * 2
                
                E = 1.0 / np.cosh(np.abs(fitness[i] - BF)) # sech = 1/cosh
                R = 2 * shrink * np.random.rand() - shrink
                
                if r1 < l:
                    new_population[i] = population[i] * (1 + np.random.randn())
                else:
                    if r2 > E:
                        new_population[i] = W1 * self.global_best_pos + R * W2 * np.abs(self.global_best_pos - population[i])
                    else:
                        new_population[i] = W1 * self.global_best_pos - R * W2 * np.abs(self.global_best_pos - population[i])
                        
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
