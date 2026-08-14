import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class EO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.a1 = 2
        self.a2 = 1
        self.GP = 0.5

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for Iter in range(1, self.max_iter):
            # Identify the four best particles
            sort_idx = np.argsort(fitness)
            Ceq1 = population[sort_idx[0]]
            Ceq2 = population[sort_idx[1]]
            Ceq3 = population[sort_idx[2]]
            Ceq4 = population[sort_idx[3]]
            Ceq_ave = (Ceq1 + Ceq2 + Ceq3 + Ceq4) / 4.0
            
            Ceq_pool = [Ceq1, Ceq2, Ceq3, Ceq4, Ceq_ave]
            
            t = (1 - Iter / self.max_iter) ** (self.a2 * Iter / self.max_iter)
            
            new_population = np.copy(population)
            
            for i in range(self.pop_size):
                Ceq = Ceq_pool[np.random.randint(0, len(Ceq_pool))]
                
                lam = np.random.rand(self.dim)
                r = np.random.rand(self.dim)
                
                # F = a1 * sign(r - 0.5) * (exp(-lam * t) - 1)
                F = self.a1 * np.sign(r - 0.5) * (np.exp(-lam * t) - 1)
                
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                GCP = 0.5 * r1 * (r2 >= self.GP)
                G0 = GCP * (Ceq - lam * population[i])
                G = G0 * F # Simplified from G0 * exp(...)
                
                new_population[i] = Ceq + (population[i] - Ceq) * F + (G / (lam + 1e-10)) * (1 - F)
                
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
