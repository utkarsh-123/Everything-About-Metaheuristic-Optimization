import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class HBBO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.num_fields = 5
        self.field_size = pop_size // self.num_fields

    def optimize(self):
        population = self.initialize_population()
        fitness = self.evaluate(population)
        self.update_global_best(population, fitness)
        self.convergence_curve.append(self.global_best_score)
        
        # Assign fields
        fields = np.repeat(np.arange(self.num_fields), self.field_size)
        if len(fields) < self.pop_size:
            fields = np.append(fields, np.random.randint(0, self.num_fields, self.pop_size - len(fields)))

        for t in range(1, self.max_iter):
            # Education
            for f in range(self.num_fields):
                field_indices = np.where(fields == f)[0]
                if len(field_indices) == 0: continue
                
                expert_idx = field_indices[np.argmin(fitness[field_indices])]
                expert = population[expert_idx]
                
                for i in field_indices:
                    if i == expert_idx: continue
                    dist = np.linalg.norm(population[i] - expert)
                    r = np.random.rand() * dist
                    direction = np.random.randn(self.dim)
                    direction /= (np.linalg.norm(direction) + 1e-10)
                    population[i] += r * direction
                    population[i] = np.clip(population[i], self.lb, self.ub)
                    fitness[i] = self.objective_func(population[i])
                    
            # Consultation
            for i in range(self.pop_size):
                if i == np.argmin(fitness): continue
                advisor_idx = np.random.randint(0, self.pop_size)
                while advisor_idx == i:
                    advisor_idx = np.random.randint(0, self.pop_size)
                    
                temp_pos = np.copy(population[i])
                num_vars_change = int(0.5 * self.dim) # Ccf * n
                change_indices = np.random.choice(self.dim, num_vars_change, replace=False)
                temp_pos[change_indices] = population[advisor_idx][change_indices]
                
                temp_fit = self.objective_func(temp_pos)
                if temp_fit < fitness[i]:
                    population[i] = temp_pos
                    fitness[i] = temp_fit
                    
            # Field Changing Probability
            field_expert_fitness = np.zeros(self.num_fields)
            for f in range(self.num_fields):
                field_indices = np.where(fields == f)[0]
                if len(field_indices) > 0:
                    field_expert_fitness[f] = np.min(fitness[field_indices])
                else:
                    field_expert_fitness[f] = np.inf
                    
            field_ranks = np.argsort(np.argsort(-field_expert_fitness)) + 1 # 1=worst
            
            for f in range(self.num_fields):
                P_f = 1 - field_ranks[f] / self.num_fields
                if np.random.rand() < P_f:
                    field_indices = np.where(fields == f)[0]
                    if len(field_indices) == 0: continue
                    
                    probs = fitness[field_indices] / (np.sum(fitness[field_indices]) + 1e-10)
                    leaver_idx = np.random.choice(field_indices, p=probs)
                    new_field = np.random.randint(0, self.num_fields)
                    fields[leaver_idx] = new_field

            self.update_global_best(population, fitness)
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
