import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class HGSO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.num_clusters = 5
        self.cluster_size = pop_size // self.num_clusters

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        # Initialize constants
        H = 0.05 * np.random.rand(self.num_clusters)
        P = 100 * np.random.rand(self.pop_size)
        C = 0.01 * np.random.rand(self.num_clusters)
        T_theta = 298.15
        K = 1.0
        
        for t in range(1, self.max_iter):
            new_population = np.copy(population)
            T = np.exp(-t / self.max_iter)
            
            # Find cluster bests
            cluster_bests = np.zeros((self.num_clusters, self.dim))
            for j in range(self.num_clusters):
                cluster_fit = fitness[j*self.cluster_size : (j+1)*self.cluster_size]
                cluster_pop = population[j*self.cluster_size : (j+1)*self.cluster_size]
                cluster_bests[j] = cluster_pop[np.argmin(cluster_fit)]
                
                # Update Henry's coefficient
                H[j] = H[j] * np.exp(-C[j] * (1.0/(T+1e-10) - 1.0/T_theta))
                
            for j in range(self.num_clusters):
                for i in range(self.cluster_size):
                    idx = j * self.cluster_size + i
                    S = K * H[j] * P[idx]
                    
                    F = 1 if np.random.rand() < 0.5 else -1
                    r = np.random.rand()
                    gamma = 1.0
                    alpha = 1.0
                    
                    new_pos = population[idx] + F * r * gamma * (cluster_bests[j] - population[idx]) \
                              + F * r * alpha * (S * self.global_best_pos - population[idx])
                    
                    new_population[idx] = new_pos
                    
            # Escaping Local Optimum
            N_w = int(self.pop_size * (np.random.rand() * 0.1 + 0.1))
            worst_indices = np.argsort(fitness)[-N_w:]
            for idx in worst_indices:
                new_population[idx] = np.random.uniform(self.lb, self.ub, self.dim)
                
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
