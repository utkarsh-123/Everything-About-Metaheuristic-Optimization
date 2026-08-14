import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class MA(BaseAlgorithm):
    """
    Mayfly Algorithm (MA)
    Inspired by the mating behaviour of mayflies.
    Males fly attracted to the best position; females move toward males.
    Nuptial dance is performed when no improvement occurs.
    """
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.g = 0.8    # attraction coefficient
        self.a1 = 1.0   # cognitive coefficient
        self.a2 = 1.5   # social coefficient
        self.beta = 2.0 # visibility coefficient
        self.d = 0.1    # nuptial dance step

    def optimize(self):
        n = self.pop_size // 2  # half males, half females
        males   = self.initialize_population()[:n]
        females = self.initialize_population()[:n]

        vel_m = np.zeros_like(males)
        vel_f = np.zeros_like(females)

        fit_m = self.evaluate(males)
        fit_f = self.evaluate(females)

        pbest_m     = np.copy(males)
        pbest_fit_m = np.copy(fit_m)

        all_pop = np.vstack([males, females])
        all_fit = np.concatenate([fit_m, fit_f])
        self.update_global_best(all_pop, all_fit)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            # --- Update Males ---
            for i in range(n):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                dist_pbest = np.linalg.norm(pbest_m[i] - males[i])
                dist_gbest = np.linalg.norm(self.global_best_pos - males[i])

                vel_m[i] = (self.g * vel_m[i]
                            + self.a1 * r1 * np.exp(-self.beta * dist_pbest**2) * (pbest_m[i] - males[i])
                            + self.a2 * r2 * np.exp(-self.beta * dist_gbest**2) * (self.global_best_pos - males[i]))
                males[i] = np.clip(males[i] + vel_m[i], self.lb, self.ub)
                fit_m[i] = self.objective_func(males[i])

                if fit_m[i] < pbest_fit_m[i]:
                    pbest_m[i]     = np.copy(males[i])
                    pbest_fit_m[i] = fit_m[i]

            # --- Update Females (attracted to best male) ---
            # Sort males by fitness to find each female's nearest/best male
            male_rank = np.argsort(fit_m)
            for i in range(n):
                j = male_rank[i % n]  # pair with ranked male
                dist_mf = np.linalg.norm(males[j] - females[i])
                r3 = np.random.rand(self.dim)

                vel_f[i] = (self.g * vel_f[i]
                            + self.a2 * r3 * np.exp(-self.beta * dist_mf**2) * (males[j] - females[i]))
                females[i] = np.clip(females[i] + vel_f[i], self.lb, self.ub)
                fit_f[i] = self.objective_func(females[i])

            # --- Nuptial dance if no improvement ---
            if t % 5 == 0:
                for i in range(n):
                    males[i]   += self.d * np.random.uniform(-1, 1, self.dim) * (self.ub - self.lb)
                    females[i] += self.d * np.random.uniform(-1, 1, self.dim) * (self.ub - self.lb)
                    males[i]   = np.clip(males[i],   self.lb, self.ub)
                    females[i] = np.clip(females[i], self.lb, self.ub)
                    fit_m[i] = self.objective_func(males[i])
                    fit_f[i] = self.objective_func(females[i])

            all_pop = np.vstack([males, females])
            all_fit = np.concatenate([fit_m, fit_f])
            self.update_global_best(all_pop, all_fit)
            self.convergence_curve.append(self.global_best_score)

        return self.global_best_pos, self.global_best_score
