import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class ChOA(BaseAlgorithm):
    """
    Chimp Optimization Algorithm (ChOA)
    Inspired by the individual intelligence and sexual motivation of chimps.
    Four roles: Attacker, Barrier, Chaser, Driver.
    """
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            # Sort and assign roles to top 4 individuals
            sort_idx = np.argsort(fitness)
            attacker = population[sort_idx[0]]   # best
            barrier  = population[sort_idx[1]]
            chaser   = population[sort_idx[2]]
            driver   = population[sort_idx[3]]

            f = np.random.uniform(-2, 2)  # chaotic map-like factor
            new_population = np.copy(population)

            for i in range(self.pop_size):
                # Each agent updates position w.r.t. the four leaders
                m1, m2 = np.random.rand(self.dim), np.random.rand(self.dim)
                m3, m4 = np.random.rand(self.dim), np.random.rand(self.dim)

                D_att = np.abs(m1 * attacker - f * population[i])
                D_bar = np.abs(m2 * barrier  - f * population[i])
                D_cha = np.abs(m3 * chaser   - f * population[i])
                D_dri = np.abs(m4 * driver   - f * population[i])

                a = 2 - t * (2.0 / self.max_iter)
                c = 2 * np.random.rand(self.dim)

                X1 = attacker - a * D_att * c
                X2 = barrier  - a * D_bar * c
                X3 = chaser   - a * D_cha * c
                X4 = driver   - a * D_dri * c

                new_pos = (X1 + X2 + X3 + X4) / 4.0

                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)

                if new_fit < fitness[i]:
                    new_population[i] = new_pos
                    fitness[i] = new_fit

            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)

        return self.global_best_pos, self.global_best_score
