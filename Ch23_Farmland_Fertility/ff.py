import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class FF(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.k_sections = 5
        self.section_size = pop_size // self.k_sections

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        local_memory = np.zeros((self.k_sections, self.dim))
        
        for t in range(1, self.max_iter):
            alpha = np.random.rand()
            beta = np.random.rand()
            
            # Evaluate sections and update local memory
            section_quality = np.zeros(self.k_sections)
            for s in range(self.k_sections):
                s_fit = fitness[s*self.section_size : (s+1)*self.section_size]
                s_pop = population[s*self.section_size : (s+1)*self.section_size]
                section_quality[s] = np.mean(s_fit)
                local_memory[s] = s_pop[np.argmin(s_fit)]
                
            worst_section = np.argmax(section_quality)
            
            new_population = np.copy(population)
            
            for s in range(self.k_sections):
                for i in range(self.section_size):
                    idx = s * self.section_size + i
                    
                    if s == worst_section:
                        # Combine with global memory (best_pos)
                        h = alpha * np.random.rand()
                        new_population[idx] = population[idx] + h * (population[idx] - self.global_best_pos)
                    else:
                        # Combine with random solution
                        h = beta * np.random.rand()
                        u_idx = np.random.randint(0, self.pop_size)
                        new_population[idx] = population[idx] + h * (population[idx] - population[u_idx])
                        
                    # Also randomly mix with local memory
                    if np.random.rand() < 0.5:
                        new_population[idx] += np.random.rand() * (local_memory[s] - new_population[idx])

            new_population = self.enforce_bounds(new_population)
            new_fitness = self.evaluate(new_population)
            
            for i in range(self.pop_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
