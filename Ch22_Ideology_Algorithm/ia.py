import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class IA(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.num_parties = 5
        self.party_size = pop_size // self.num_parties

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        parties = np.repeat(np.arange(self.num_parties), self.party_size)
        if len(parties) < self.pop_size:
            parties = np.append(parties, np.random.randint(0, self.num_parties, self.pop_size - len(parties)))

        for t in range(1, self.max_iter):
            w1, w2, w3, w4, w5 = 0.1, 0.5, 0.5, 0.1, 0.5
            d_thresh = 0.05 * (self.ub - self.lb)
            
            for p in range(self.num_parties):
                p_indices = np.where(parties == p)[0]
                if len(p_indices) < 2: continue
                
                # Sort party members
                p_fitness = fitness[p_indices]
                p_sort_idx = np.argsort(p_fitness)
                sorted_p_indices = p_indices[p_sort_idx]
                
                leader_idx = sorted_p_indices[0]
                second_best_idx = sorted_p_indices[1]
                worst_idx = sorted_p_indices[-1]
                second_worst_idx = sorted_p_indices[-2]
                
                # Update Local Leader
                strategy = np.random.randint(0, 3)
                if strategy == 0: # Introspection
                    new_pos = population[leader_idx] + w1 * np.random.uniform(-1, 1, self.dim) * (self.ub - self.lb)
                elif strategy == 1: # Local competition
                    new_pos = population[leader_idx] + w2 * np.random.rand(self.dim) * (population[leader_idx] - population[second_best_idx])
                else: # Global competition
                    new_pos = population[leader_idx] + w3 * np.random.rand(self.dim) * (self.global_best_pos - population[leader_idx])
                    
                new_pos = np.clip(new_pos, self.lb, self.ub)
                new_fit = self.objective_func(new_pos)
                if new_fit < fitness[leader_idx]:
                    population[leader_idx] = new_pos
                    fitness[leader_idx] = new_fit
                    
                # Update Deserted Individual
                d = np.linalg.norm(population[worst_idx] - population[second_worst_idx])
                if d > np.linalg.norm(d_thresh):
                    # Jump to rival party
                    rival = np.random.randint(0, self.num_parties)
                    while rival == p: rival = np.random.randint(0, self.num_parties)
                    parties[worst_idx] = rival
                else:
                    # Shift toward ordinary member
                    ordinary_idx = np.random.choice(sorted_p_indices[1:-1]) if len(sorted_p_indices) > 2 else leader_idx
                    population[worst_idx] += np.random.rand(self.dim) * (population[ordinary_idx] - population[worst_idx])
                    population[worst_idx] = np.clip(population[worst_idx], self.lb, self.ub)
                    fitness[worst_idx] = self.objective_func(population[worst_idx])
                    
                # Update Ordinary Individuals
                for i in sorted_p_indices[1:-1]:
                    pos_self = population[i] + w4 * np.random.uniform(-1, 1, self.dim) * (self.ub - self.lb)
                    pos_leader = population[i] + w5 * np.random.rand(self.dim) * (population[leader_idx] - population[i])
                    
                    pos_self = np.clip(pos_self, self.lb, self.ub)
                    pos_leader = np.clip(pos_leader, self.lb, self.ub)
                    
                    fit_self = self.objective_func(pos_self)
                    fit_leader = self.objective_func(pos_leader)
                    
                    if fit_self < fitness[i] and fit_self < fit_leader:
                        population[i] = pos_self
                        fitness[i] = fit_self
                    elif fit_leader < fitness[i]:
                        population[i] = pos_leader
                        fitness[i] = fit_leader

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
