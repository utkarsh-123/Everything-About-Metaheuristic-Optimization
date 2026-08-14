import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class BO(BaseAlgorithm):
    """
    Bonobo Optimizer (BO)
    Inspired by the sexual and social behaviour of Bonobos.
    Three strategies: Progressive Directed Strategy, Promiscuous Strategy, Restrictive Strategy.
    """
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.alpha = 0.9  # Initial probability for PDS

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            # Probability of each strategy varies with iteration
            alpha_t = self.alpha - (self.alpha - 0.1) * (t / self.max_iter)  # PDS probability decreases
            beta_t = (1 - alpha_t) * 0.5  # Promiscuous
            # Restrictive probability = 1 - alpha_t - beta_t

            new_population = np.copy(population)

            for i in range(self.pop_size):
                r = np.random.rand()

                if r < alpha_t:
                    # Progressive Directed Strategy (PDS): move toward global best
                    phi = np.random.uniform(-1, 1, self.dim)
                    new_pos = population[i] + phi * (self.global_best_pos - np.abs(population[i]))
                elif r < alpha_t + beta_t:
                    # Promiscuous Strategy: mate with random partner
                    j = np.random.randint(0, self.pop_size)
                    while j == i:
                        j = np.random.randint(0, self.pop_size)
                    partner = population[j]
                    # Crossover-like blend
                    phi = np.random.rand(self.dim)
                    new_pos = phi * population[i] + (1 - phi) * partner
                else:
                    # Restrictive Strategy: mate within close neighbourhood
                    # pick the nearest neighbour
                    distances = np.linalg.norm(population - population[i], axis=1)
                    distances[i] = np.inf
                    nearest = np.argmin(distances)
                    phi = np.random.rand(self.dim)
                    new_pos = phi * population[i] + (1 - phi) * population[nearest]

                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)

                if new_fit < fitness[i]:
                    new_population[i] = new_pos
                    fitness[i] = new_fit

            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)

        return self.global_best_pos, self.global_best_score
