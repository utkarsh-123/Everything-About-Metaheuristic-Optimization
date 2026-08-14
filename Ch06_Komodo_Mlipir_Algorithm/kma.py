import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class KMA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.p = 0.5
        self.m = 0.5 # mlipir rate

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(self.max_iter - 1):
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            # Stratification
            n1 = max(1, int(np.round(self.pop_size * self.p)))
            n_female = 1
            n2 = self.pop_size - n1 - n_female
            
            big_males = population[:n1]
            female = population[n1]
            small_males = population[n1 + n_female:]
            
            # Big Male Movement
            new_big_males = np.copy(big_males)
            for i in range(n1):
                j = np.random.randint(0, n1)
                r1, r2 = np.random.rand(), np.random.rand()
                if fitness[i] < fitness[j] or r2 < 0.5:
                    new_big_males[i] = big_males[i] + r1 * (big_males[j] - big_males[i])
                else:
                    new_big_males[i] = big_males[i] + r1 * (big_males[i] - big_males[j])
            
            # Female Reproduction (Sexual Mating)
            r1 = np.random.rand()
            if np.random.rand() < 0.5:
                # Sexual Mating with best male (index 0)
                M = big_males[0]
                O1 = r1 * M + (1 - r1) * female
                O2 = r1 * female + (1 - r1) * M
                
                O1_fit = self.objective_func(self.enforce_bounds(O1))
                O2_fit = self.objective_func(self.enforce_bounds(O2))
                
                new_female = O1 if O1_fit < O2_fit else O2
            else:
                # Parthenogenesis
                r1 = np.random.rand()
                new_female = female + (2 * r1 - 1) * 0.1 * (self.ub - self.lb)
                
            # Small Male Movement
            new_small_males = np.copy(small_males)
            for i in range(n2):
                j = np.random.randint(0, n1)
                for l in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    if r2 < self.m:
                        new_small_males[i, l] = small_males[i, l] + r1 * (big_males[j, l] - small_males[i, l])
            
            # Reassemble population
            new_population = np.vstack([new_big_males, new_female, new_small_males])
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
