import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class ACO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.evaporation_rate = 0.5
        self.q = 1.0

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        # Pheromone matrix for continuous domain abstracted as variances
        pheromones = np.ones(self.dim) * (self.ub - self.lb)

        for t in range(1, self.max_iter):
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # Update pheromones based on best solutions
            best_pos = population[0]
            for j in range(self.dim):
                pheromones[j] = (1 - self.evaporation_rate) * pheromones[j] + self.q / (fitness[0] + 1e-10)
                
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                for j in range(self.dim):
                    # Sample new position using Gaussian distribution around best position
                    # standard deviation is influenced by pheromone
                    std_dev = pheromones[j] / np.sqrt(i + 1)
                    new_population[i, j] = best_pos[j] + np.random.randn() * std_dev
                    
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
