import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class GEO(BaseAlgorithm):
    """
    Golden Eagle Optimizer (GEO)
    Based on the hunting strategy of golden eagles — cruise, attack, and soar phases.
    Eagles attack prey and adjust attack/cruise vectors each iteration.
    """
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.attack_coef  = 0.5  # initial attack proportion
        self.cruise_coef  = 0.5  # initial cruise proportion

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        # Each eagle has an attack and cruise vector
        attack_vec = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        cruise_vec = np.random.uniform(-1, 1, (self.pop_size, self.dim))

        for t in range(1, self.max_iter):
            # Linearly decrease attack, increase cruise (shift exploitation → exploration)
            attack_ratio = self.attack_coef * (1 - t / self.max_iter)
            cruise_ratio = self.cruise_coef * (t / self.max_iter)

            new_population = np.copy(population)

            for i in range(self.pop_size):
                # Choose random prey (another eagle or global best)
                j = np.random.randint(0, self.pop_size)
                prey = population[j] if fitness[j] > fitness[i] else self.global_best_pos

                # Attack step: move toward prey
                attack_step = attack_ratio * attack_vec[i] * (prey - population[i])

                # Cruise step: soaring movement perpendicular to attack
                cruise_vec[i] = np.random.uniform(-1, 1, self.dim)
                cruise_step = cruise_ratio * cruise_vec[i] * (self.ub - self.lb)

                new_pos = population[i] + attack_step + cruise_step

                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)

                if new_fit < fitness[i]:
                    new_population[i] = new_pos
                    fitness[i] = new_fit
                    # Update attack vector toward successful direction
                    attack_vec[i] = (new_pos - population[i]) / (np.linalg.norm(new_pos - population[i]) + 1e-10)

            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)

        return self.global_best_pos, self.global_best_score
