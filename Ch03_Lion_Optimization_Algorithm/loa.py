import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class LOA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        # Simplified LOA parameters based on Chapter 3
        self.num_nomads = int(pop_size * 0.2)
        self.num_prides = pop_size - self.num_nomads

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(self.max_iter - 1):
            new_population = np.copy(population)
            
            # Split into prides and nomads for abstraction
            prides = new_population[:self.num_prides]
            nomads = new_population[self.num_prides:]
            
            # Hunting process for prides (Eq 3.1, 3.2, 3.3 abstracted)
            # Center hunters moving toward prey (global best)
            for i in range(len(prides)):
                hunter = prides[i]
                prey = self.global_best_pos
                
                # Simplified Eq 3.1 & 3.2: Hunters moving towards prey
                rand_val = np.random.rand()
                if rand_val < 0.5:
                    new_hunter = np.random.rand(self.dim) * (2 * prey - hunter) + prey
                else:
                    new_hunter = np.random.rand(self.dim) * prey + (2 * prey - hunter)
                
                prides[i] = new_hunter
                
            # Nomads random movement (Eq 3.5 abstracted)
            for i in range(len(nomads)):
                prob = np.random.rand()
                if prob > 0.5:
                    nomads[i] = np.random.uniform(self.lb, self.ub, self.dim)
                else:
                    nomads[i] = nomads[i] + np.random.randn(self.dim) * 0.1 * (self.ub - self.lb)
            
            # Combine back
            new_population[:self.num_prides] = prides
            new_population[self.num_prides:] = nomads
            
            # Enforce bounds
            new_population = self.enforce_bounds(new_population)
            
            # Evaluate
            new_fitness = self.evaluate(new_population)
            
            # Selection
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
