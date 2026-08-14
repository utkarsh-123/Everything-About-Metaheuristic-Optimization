import numpy as np
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class AO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def _levy_flight(self):
        beta = 1.5
        sigma = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) / 
                 (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        step = u / (np.abs(v) ** (1 / beta))
        return step

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            new_population = np.copy(population)
            Xmean = np.mean(population, axis=0)
            
            for i in range(self.pop_size):
                if t <= (2/3) * self.max_iter:
                    # Exploration phases
                    if np.random.rand() <= 0.5:
                        # Phase 1: Expanded exploration
                        r1 = np.random.rand()
                        new_population[i] = self.global_best_pos * (1 - t / self.max_iter) + (Xmean - self.global_best_pos * r1)
                    else:
                        # Phase 2: Narrowed exploration
                        r2 = np.random.rand()
                        levy = self._levy_flight()
                        Xrand = population[np.random.randint(0, self.pop_size)]
                        
                        # Spiral calculation (simplified)
                        r_spiral = 10 + 0.05 * i
                        theta = -0.05 * i + (3 * math.pi / 2)
                        x_spiral = r_spiral * math.sin(theta)
                        y_spiral = r_spiral * math.cos(theta)
                        
                        new_population[i] = self.global_best_pos * levy + Xrand + (y_spiral - x_spiral) * r2
                else:
                    # Exploitation phases
                    if np.random.rand() <= 0.5:
                        # Phase 3: Expanded exploitation
                        alpha = 0.1
                        delta = 0.1
                        r3 = np.random.rand()
                        r4 = np.random.rand()
                        new_population[i] = (self.global_best_pos - Xmean) * alpha - r3 + ((self.ub - self.lb) * r4 + self.lb) * delta
                    else:
                        # Phase 4: Narrowed exploitation
                        Q = t ** ((2 * np.random.rand()) - 1) / (1 - self.max_iter) ** 2 if self.max_iter > 1 else 1
                        G1 = 2 * np.random.rand() - 1
                        G2 = 2 * (1 - t / self.max_iter)
                        r5 = np.random.rand()
                        r6 = np.random.rand()
                        levy = self._levy_flight()
                        
                        new_population[i] = Q * self.global_best_pos - (G1 * population[i] * r5) - G2 * levy + r6 * G1
            
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
