import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class ABC(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30, limit=100):
        # In ABC, pop_size is typically the total number of bees (employed + onlookers)
        # Half of the swarm is employed, half is onlookers.
        self.num_employed = pop_size // 2
        super().__init__(objective_func, bounds, dim, self.num_employed, max_iter)
        self.limit = limit
        
    def _calculate_fitness(self, f_vals):
        """ Calculate fitness for roulette wheel selection (minimization problem) """
        fit = np.zeros_like(f_vals)
        for i in range(len(f_vals)):
            if f_vals[i] >= 0:
                fit[i] = 1.0 / (1.0 + f_vals[i])
            else:
                fit[i] = 1.0 + np.abs(f_vals[i])
        return fit

    def optimize(self):
        # Initialize food sources (positions)
        population = self.initialize_population()
        f_vals = self.evaluate(population)
        fitness = self._calculate_fitness(f_vals)
        
        trial_counters = np.zeros(self.pop_size)
        
        # Update global best initially
        self.update_global_best(population, f_vals)
        self.convergence_curve.append(self.global_best_score)

        for t in range(self.max_iter - 1):
            
            # --- Employed Bee Phase ---
            for i in range(self.pop_size):
                k = np.random.randint(0, self.pop_size)
                while k == i:
                    k = np.random.randint(0, self.pop_size)
                    
                j = np.random.randint(0, self.dim)
                phi = np.random.uniform(-1, 1)
                
                v_i = np.copy(population[i])
                v_i[j] = population[i][j] + phi * (population[i][j] - population[k][j])
                v_i = np.clip(v_i, self.lb, self.ub)
                
                f_vi = self.objective_func(v_i)
                fit_vi = self._calculate_fitness(np.array([f_vi]))[0]
                
                if fit_vi > fitness[i]:
                    population[i] = v_i
                    f_vals[i] = f_vi
                    fitness[i] = fit_vi
                    trial_counters[i] = 0
                else:
                    trial_counters[i] += 1
            
            # --- Onlooker Bee Phase ---
            prob = fitness / np.sum(fitness)
            m = 0
            i = 0
            while m < self.pop_size:
                if np.random.rand() < prob[i]:
                    m += 1
                    k = np.random.randint(0, self.pop_size)
                    while k == i:
                        k = np.random.randint(0, self.pop_size)
                        
                    j = np.random.randint(0, self.dim)
                    phi = np.random.uniform(-1, 1)
                    
                    v_i = np.copy(population[i])
                    v_i[j] = population[i][j] + phi * (population[i][j] - population[k][j])
                    v_i = np.clip(v_i, self.lb, self.ub)
                    
                    f_vi = self.objective_func(v_i)
                    fit_vi = self._calculate_fitness(np.array([f_vi]))[0]
                    
                    if fit_vi > fitness[i]:
                        population[i] = v_i
                        f_vals[i] = f_vi
                        fitness[i] = fit_vi
                        trial_counters[i] = 0
                    else:
                        trial_counters[i] += 1
                        
                i = (i + 1) % self.pop_size
                
            # --- Scout Bee Phase ---
            max_trial_idx = np.argmax(trial_counters)
            if trial_counters[max_trial_idx] > self.limit:
                population[max_trial_idx] = np.random.uniform(self.lb, self.ub, self.dim)
                f_vals[max_trial_idx] = self.objective_func(population[max_trial_idx])
                fitness[max_trial_idx] = self._calculate_fitness(np.array([f_vals[max_trial_idx]]))[0]
                trial_counters[max_trial_idx] = 0
                
            self.update_global_best(population, f_vals)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
