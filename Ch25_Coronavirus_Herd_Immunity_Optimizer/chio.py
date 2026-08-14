import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class CHIO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.BRr = 0.5
        self.max_age = 10
        self.C0 = 1

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        status = np.zeros(self.pop_size) # 0=Susceptible, 1=Infected, 2=Immune
        age = np.zeros(self.pop_size)
        
        # Initialize Infected
        infected_indices = np.random.choice(self.pop_size, self.C0, replace=False)
        status[infected_indices] = 1

        for t in range(1, self.max_iter):
            new_population = np.copy(population)
            
            susceptible = np.where(status == 0)[0]
            infected = np.where(status == 1)[0]
            immune = np.where(status == 2)[0]
            
            for i in range(self.pop_size):
                r = np.random.rand()
                
                if r < (1/3) * self.BRr and len(infected) > 0:
                    # Infected interaction
                    idx = np.random.choice(infected)
                    new_population[i] = population[i] + np.random.rand(self.dim) * (population[i] - population[idx])
                elif r < (2/3) * self.BRr and len(susceptible) > 0:
                    # Susceptible interaction
                    idx = np.random.choice(susceptible)
                    new_population[i] = population[i] + np.random.rand(self.dim) * (population[i] - population[idx])
                elif len(immune) > 0:
                    # Immune interaction
                    idx = immune[np.argmin(fitness[immune])]
                    new_population[i] = population[i] + np.random.rand(self.dim) * (population[i] - population[idx])
                else:
                    new_population[i] = population[i]
                    
            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    
                if status[i] == 1:
                    age[i] += 1
                    if age[i] >= self.max_age:
                        status[i] = 2 # Become immune or regenerate
                        if np.random.rand() < 0.5: # Regenerate
                            population[i] = np.random.uniform(self.lb, self.ub, self.dim)
                            fitness[i] = self.objective_func(population[i])
                            status[i] = 0
                            age[i] = 0

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
