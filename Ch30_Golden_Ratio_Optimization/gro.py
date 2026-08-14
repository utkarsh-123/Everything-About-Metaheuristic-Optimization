import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class GRO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.GF = 1.618

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            # Phase 1: Mean-guided Exploration
            X_mean = np.mean(population, axis=0)
            f_mean = self.objective_func(self.enforce_bounds(X_mean))
            
            worst_idx = np.argmax(fitness)
            if f_mean < fitness[worst_idx]:
                population[worst_idx] = np.copy(X_mean)
                fitness[worst_idx] = f_mean
                
            Ft = self.GF * (self.GF**t - (1 - self.GF)**t) / np.sqrt(5)
            Ft_norm = t / self.max_iter # using simplified Ft mapping for convergence
            
            new_population = np.copy(population)
            for i in range(self.pop_size):
                rand_vec = np.random.rand(self.dim)
                X_rand = population[np.random.randint(0, self.pop_size)]
                new_population[i] = (1 - Ft_norm) * self.global_best_pos + rand_vec * Ft_norm * X_rand
                
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    
            # Phase 2: Best-Worst Attraction
            worst_pos = population[np.argmax(fitness)]
            new_population = np.copy(population)
            for i in range(self.pop_size):
                new_population[i] = population[i] + np.random.rand(self.dim) * (1 / self.GF) * (self.global_best_pos - worst_pos)
                
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
