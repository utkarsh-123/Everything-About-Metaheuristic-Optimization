import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class WCA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.N_sr = 4 # Number of rivers + sea
        self.dmax = 1e-5

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)

        for t in range(1, self.max_iter):
            sort_idx = np.argsort(fitness)
            population = population[sort_idx]
            fitness = fitness[sort_idx]
            
            sea = population[0]
            sea_fit = fitness[0]
            rivers = population[1:self.N_sr]
            rivers_fit = fitness[1:self.N_sr]
            streams = population[self.N_sr:]
            streams_fit = fitness[self.N_sr:]
            
            N_streams = len(streams)
            # Assign streams to rivers and sea
            cost_sr = np.concatenate(([sea_fit], rivers_fit))
            cost_sr_norm = np.max(cost_sr) - cost_sr
            if np.sum(cost_sr_norm) == 0:
                cost_sr_norm = np.ones(self.N_sr)
            probs = cost_sr_norm / np.sum(cost_sr_norm)
            counts = np.round(probs * N_streams).astype(int)
            while np.sum(counts) < N_streams: counts[0] += 1
            while np.sum(counts) > N_streams: counts[np.argmax(counts)] -= 1
            
            C = 2.0
            idx = 0
            
            # Streams flow to sea (0) and rivers (1..)
            for j in range(self.N_sr):
                for k in range(counts[j]):
                    if idx >= N_streams: break
                    target = sea if j == 0 else rivers[j-1]
                    streams[idx] += np.random.rand(self.dim) * C * (target - streams[idx])
                    streams[idx] = np.clip(streams[idx], self.lb, self.ub)
                    streams_fit[idx] = self.objective_func(streams[idx])
                    
                    # Update target if stream is better
                    if streams_fit[idx] < (sea_fit if j == 0 else rivers_fit[j-1]):
                        temp_pos, temp_fit = np.copy(target), (sea_fit if j == 0 else rivers_fit[j-1])
                        if j == 0:
                            sea, sea_fit = np.copy(streams[idx]), streams_fit[idx]
                        else:
                            rivers[j-1], rivers_fit[j-1] = np.copy(streams[idx]), streams_fit[idx]
                        streams[idx], streams_fit[idx] = temp_pos, temp_fit
                    idx += 1
                    
            # Rivers flow to sea
            for j in range(len(rivers)):
                rivers[j] += np.random.rand(self.dim) * C * (sea - rivers[j])
                rivers[j] = np.clip(rivers[j], self.lb, self.ub)
                rivers_fit[j] = self.objective_func(rivers[j])
                if rivers_fit[j] < sea_fit:
                    sea, sea_fit, rivers[j], rivers_fit[j] = rivers[j], rivers_fit[j], sea, sea_fit
                    
            # Evaporation
            dmax = self.dmax - t * (self.dmax / self.max_iter)
            for j in range(len(rivers)):
                if np.linalg.norm(sea - rivers[j]) < dmax:
                    for _ in range(counts[j+1]): # approximate regen
                        s_idx = np.random.randint(0, N_streams)
                        streams[s_idx] = self.lb + np.random.rand(self.dim) * (self.ub - self.lb)
                        streams_fit[s_idx] = self.objective_func(streams[s_idx])
                        
            # Reconstruct population
            population[0] = sea
            fitness[0] = sea_fit
            population[1:self.N_sr] = rivers
            fitness[1:self.N_sr] = rivers_fit
            population[self.N_sr:] = streams
            fitness[self.N_sr:] = streams_fit

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
