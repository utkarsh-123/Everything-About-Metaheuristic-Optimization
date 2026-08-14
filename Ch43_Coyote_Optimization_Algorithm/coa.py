import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class COA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.num_packs = 5
        self.pack_size = pop_size // self.num_packs

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        packs = np.repeat(np.arange(self.num_packs), self.pack_size)
        if len(packs) < self.pop_size:
            packs = np.append(packs, np.random.randint(0, self.num_packs, self.pop_size - len(packs)))

        for t in range(1, self.max_iter):
            new_population = np.copy(population)
            
            for p in range(self.num_packs):
                p_indices = np.where(packs == p)[0]
                if len(p_indices) == 0: continue
                
                p_fit = fitness[p_indices]
                alpha_idx = p_indices[np.argmin(p_fit)]
                alpha = population[alpha_idx]
                p_mean = np.mean(population[p_indices], axis=0)
                
                for i in p_indices:
                    r1, r2 = np.random.rand(), np.random.rand()
                    c1, c2 = population[np.random.choice(p_indices)], population[np.random.choice(p_indices)]
                    
                    new_pos = population[i] + r1 * (alpha - c1) + r2 * (p_mean - c2)
                    new_pos = np.clip(new_pos, self.lb, self.ub)
                    new_fit = self.objective_func(new_pos)
                    
                    if new_fit < fitness[i]:
                        new_population[i] = new_pos
                        fitness[i] = new_fit
                        
            # Evict some coyotes from packs
            if np.random.rand() < 0.1:
                idx = np.random.randint(0, self.pop_size)
                new_pack = np.random.randint(0, self.num_packs)
                packs[idx] = new_pack

            population = np.copy(new_population)
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
