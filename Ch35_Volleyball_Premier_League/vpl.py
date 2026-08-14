import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class VPL(BaseAlgorithm):
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
            
            new_population = np.copy(population)
            
            # Teams play against each other (i vs i+1)
            for i in range(0, self.pop_size - 1, 2):
                team1 = population[i]
                team2 = population[i+1]
                
                winner = team1
                loser = team2
                
                # Winner takes properties from global best
                new_winner = winner + np.random.rand(self.dim) * (self.global_best_pos - winner)
                # Loser learns from winner
                new_loser = loser + np.random.rand(self.dim) * (winner - loser)
                
                # Knowledge sharing among players
                crossover_mask = np.random.rand(self.dim) < 0.2
                new_winner = np.where(crossover_mask, team2, new_winner)
                new_loser = np.where(crossover_mask, team1, new_loser)
                
                new_population[i] = new_winner
                new_population[i+1] = new_loser
                
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
