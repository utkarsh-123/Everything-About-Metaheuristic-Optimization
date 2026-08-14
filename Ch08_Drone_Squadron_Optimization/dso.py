import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class DSO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.num_teams = 5
        self.drones_per_team = pop_size // self.num_teams

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        team_firmwares = np.random.randint(0, 3, self.num_teams)
        pbest_pos = np.copy(population)
        pbest_scores = np.copy(fitness)
        
        stagnation_counter = 0

        for t in range(self.max_iter - 1):
            new_population = np.copy(population)
            violations = np.zeros(self.num_teams)
            
            for team_idx in range(self.num_teams):
                fw = team_firmwares[team_idx]
                for d_idx in range(self.drones_per_team):
                    i = team_idx * self.drones_per_team + d_idx
                    
                    if fw == 0:
                        new_pos = self.global_best_pos + np.random.randn(self.dim) * 0.1 * (self.ub - self.lb)
                    elif fw == 1:
                        new_pos = pbest_pos[i] + np.random.rand(self.dim) * 0.2 * (self.ub - self.lb)
                    else:
                        new_pos = population[i] + np.random.uniform(-1, 1, self.dim) * 0.1 * (self.ub - self.lb)
                        
                    # Calculate violations before clipping
                    viol = np.sum(np.maximum(0, new_pos - self.ub)) + np.sum(np.maximum(0, self.lb - new_pos))
                    violations[team_idx] += viol
                    
                    new_population[i] = np.clip(new_pos, self.lb, self.ub)
                    
            new_fitness = self.evaluate(new_population)
            
            improved = False
            for i in range(self.pop_size):
                if new_fitness[i] < pbest_scores[i]:
                    pbest_scores[i] = new_fitness[i]
                    pbest_pos[i] = new_population[i]
                    if new_fitness[i] < self.global_best_score:
                        self.global_best_score = new_fitness[i]
                        self.global_best_pos = np.copy(new_population[i])
                        improved = True
                
                population[i] = new_population[i]
                fitness[i] = new_fitness[i]
                
            if improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            if stagnation_counter > 5:
                # Soft selection
                if np.random.rand() < 0.2:
                    random_worse = population[np.random.randint(0, self.pop_size)]
                    self.global_best_pos = np.copy(random_worse)
                    self.global_best_score = self.objective_func(self.global_best_pos)
                stagnation_counter = 0
                
            # Firmware Update
            team_ranks = np.zeros(self.num_teams)
            for team_idx in range(self.num_teams):
                team_fitness = np.mean(fitness[team_idx*self.drones_per_team : (team_idx+1)*self.drones_per_team])
                team_ranks[team_idx] = team_fitness + violations[team_idx] * 10
                
            best_team = np.argmin(team_ranks)
            worst_team = np.argmax(team_ranks)
            
            if np.random.rand() < 0.5:
                team_firmwares[worst_team] = team_firmwares[best_team]

            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
