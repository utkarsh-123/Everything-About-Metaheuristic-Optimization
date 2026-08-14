import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class HBO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            # Sort population to build heap hierarchy
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            c = self.max_iter / 25.0
            gamma = np.abs(2 - (t % (self.max_iter / c)) / (self.max_iter / (4 * c)))
            
            p1 = 1 - t / self.max_iter
            p2 = p1 + (1 - p1) / 2
            
            new_population = np.copy(population)
            
            for i in range(1, self.pop_size):
                # immediate boss is parent node in heap: (i-1)//2
                boss_idx = (i - 1) // 2
                boss = population[boss_idx]
                
                # colleague is random node at same depth
                depth = int(np.log2(i + 1))
                start_idx = 2**depth - 1
                end_idx = min(2**(depth + 1) - 1, self.pop_size)
                if end_idx - start_idx > 1:
                    colleague_idx = np.random.randint(start_idx, end_idx)
                    while colleague_idx == i:
                        colleague_idx = np.random.randint(start_idx, end_idx)
                else:
                    colleague_idx = i
                colleague = population[colleague_idx]
                
                for j in range(self.dim):
                    lam = 2 * np.random.rand() - 1
                    p = np.random.rand()
                    
                    if p <= p1:
                        # Interaction with immediate boss
                        new_population[i, j] = boss[j] + gamma * lam * np.abs(boss[j] - population[i, j])
                    elif p <= p2:
                        # Interaction with colleague
                        if fitness[colleague_idx] < fitness[i]:
                            new_population[i, j] = colleague[j] + gamma * lam * np.abs(colleague[j] - population[i, j])
                        else:
                            new_population[i, j] = population[i, j] + gamma * lam * np.abs(colleague[j] - population[i, j])
                    else:
                        # Self-contribution
                        new_population[i, j] = population[i, j]
                        
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
