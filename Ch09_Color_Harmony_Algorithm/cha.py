import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class CHA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.div0 = None
        self.mem_new = []
        self.mem_main = []

    def _calculate_diversity(self, population):
        centroid = np.mean(population, axis=0)
        distances = np.abs(population - centroid)
        div = np.mean(np.sum(distances, axis=1))
        return div, centroid

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        self.div0, _ = self._calculate_diversity(population)
        
        for t in range(self.max_iter - 1):
            div, centroid = self._calculate_diversity(population)
            beta = 0.95
            div_end = self.div0 * (beta ** t) if self.div0 else 1e-5
            
            new_population = np.copy(population)
            
            # Sort to separate agents (top) from non-agents (bottom)
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            num_agents = self.pop_size // 4
            
            # Concentration Phase
            pa = max(0.1, min(0.9, div / (self.div0 + 1e-10)))
            for i in range(self.pop_size):
                if np.random.rand() < pa:
                    # Agent + Non-Agent
                    idx1 = np.random.randint(0, num_agents)
                    idx2 = np.random.randint(num_agents, self.pop_size)
                else:
                    # Agent + Agent
                    idx1 = np.random.randint(0, num_agents)
                    idx2 = np.random.randint(0, num_agents)
                    
                r1 = np.random.rand()
                r2 = 1 - r1
                new_pos = r1 * population[idx1] + r2 * population[idx2]
                new_population[i] = new_pos
                
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                else:
                    self.mem_new.append(new_population[i])
                    
            # Memory maintenance
            if len(self.mem_new) > self.pop_size:
                self.mem_main.extend(self.mem_new)
                self.mem_new = []
                if len(self.mem_main) > self.pop_size * 2:
                    # Keep most diverse in mem_main
                    _, main_centroid = self._calculate_diversity(np.array(self.mem_main))
                    main_dists = np.sum(np.abs(self.mem_main - main_centroid), axis=1)
                    diverse_idx = np.argsort(main_dists)[-self.pop_size:]
                    self.mem_main = [self.mem_main[i] for i in diverse_idx]

            # Dispersion Phase Check
            if div < div_end and len(self.mem_main) > 0:
                # Extract central clustered colors
                dists_to_centroid = np.sum(np.abs(population - centroid), axis=1)
                central_idx = np.argsort(dists_to_centroid)[:num_agents]
                
                for i in central_idx:
                    r = 0.7
                    mem_idx = np.random.randint(0, len(self.mem_main))
                    x_mem = self.mem_main[mem_idx]
                    population[i] = r * x_mem + (1 - r) * population[i]
                
                population = self.enforce_bounds(population)
                fitness = self.evaluate(population)
                
            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
