import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class HSA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.HMCR = 0.9
        self.PAR = 0.3
        self.bw = 0.01

    def optimize(self):
        HM = self.initialize_population()
        fitness = self.evaluate(HM)
        self.update_global_best(HM, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            # We treat max_iter as number of improvizations here
            new_harmony = np.zeros(self.dim)
            
            for j in range(self.dim):
                if np.random.rand() < self.HMCR:
                    idx = np.random.randint(0, self.pop_size)
                    new_harmony[j] = HM[idx, j]
                    if np.random.rand() < self.PAR:
                        new_harmony[j] += np.random.uniform(-1, 1) * self.bw * (self.ub - self.lb)
                else:
                    new_harmony[j] = np.random.uniform(self.lb, self.ub)
                    
            new_harmony = np.clip(new_harmony, self.lb, self.ub)
            new_fit = self.objective_func(new_harmony)
            
            worst_idx = np.argmax(fitness)
            if new_fit < fitness[worst_idx]:
                HM[worst_idx] = new_harmony
                fitness[worst_idx] = new_fit
                if new_fit < self.global_best_score:
                    self.global_best_score = new_fit
                    self.global_best_pos = np.copy(new_harmony)

            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
