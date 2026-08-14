import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class GA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.crossover_rate = 0.8
        self.mutation_rate = 0.1

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            # Normalize fitness for roulette wheel selection (minimization: invert fitness)
            max_fit = np.max(fitness)
            min_fit = np.min(fitness)
            
            # Avoid division by zero
            if max_fit == min_fit:
                probs = np.ones(self.pop_size) / self.pop_size
            else:
                inv_fitness = max_fit - fitness
                probs = inv_fitness / np.sum(inv_fitness)
                
            new_population = np.zeros_like(population)
            
            for i in range(0, self.pop_size, 2):
                # Selection
                parent1_idx = np.random.choice(self.pop_size, p=probs)
                parent2_idx = np.random.choice(self.pop_size, p=probs)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # Crossover
                if np.random.rand() < self.crossover_rate:
                    alpha = np.random.rand(self.dim)
                    child1 = alpha * parent1 + (1 - alpha) * parent2
                    child2 = alpha * parent2 + (1 - alpha) * parent1
                else:
                    child1 = np.copy(parent1)
                    child2 = np.copy(parent2)
                    
                # Mutation
                for j in range(self.dim):
                    if np.random.rand() < self.mutation_rate:
                        child1[j] += np.random.randn() * 0.1 * (self.ub - self.lb)
                    if np.random.rand() < self.mutation_rate:
                        child2[j] += np.random.randn() * 0.1 * (self.ub - self.lb)
                        
                new_population[i] = child1
                if i + 1 < self.pop_size:
                    new_population[i+1] = child2

            new_population = self.enforce_bounds(new_population)
            fitness = self.evaluate(new_population)
            population = np.copy(new_population)

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
