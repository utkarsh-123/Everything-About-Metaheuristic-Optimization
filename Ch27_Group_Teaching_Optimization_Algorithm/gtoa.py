import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class GTOA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # Divide into outstanding and average
            half = self.pop_size // 2
            outstanding = population[:half]
            average = population[half:]
            
            # Calculate mean
            mean_group = np.mean(population, axis=0)
            
            # Teacher allocation (simplified to best)
            teacher = population[0]
            
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                if i < half:
                    # Outstanding group teacher phase
                    a, b, c = np.random.rand(), np.random.rand(), np.random.rand()
                    F = np.random.choice([1, 2])
                    new_pos_t = population[i] + a * (teacher - F * (b * mean_group + c * population[i]))
                else:
                    # Average group teacher phase
                    d = np.random.rand()
                    new_pos_t = population[i] + 2 * d * (teacher - population[i])
                    
                new_pos_t = np.clip(new_pos_t, self.lb, self.ub)
                fit_t = self.objective_func(new_pos_t)
                
                # Student phase
                j = np.random.randint(0, self.pop_size)
                while j == i: j = np.random.randint(0, self.pop_size)
                e, g = np.random.rand(), np.random.rand()
                
                if fit_t < fitness[j]:
                    new_pos_s = new_pos_t + e * (new_pos_t - population[j]) + g * (new_pos_t - population[i])
                else:
                    new_pos_s = new_pos_t - e * (new_pos_t - population[j]) + g * (new_pos_t - population[i])
                    
                new_pos_s = np.clip(new_pos_s, self.lb, self.ub)
                fit_s = self.objective_func(new_pos_s)
                
                # Final evaluation
                if fit_t < fitness[i] and fit_t < fit_s:
                    new_population[i] = new_pos_t
                elif fit_s < fitness[i]:
                    new_population[i] = new_pos_s

            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
