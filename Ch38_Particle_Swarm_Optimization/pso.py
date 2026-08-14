import numpy as np

class PSO:
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30, c1=2.0, c2=2.0, w=0.7):
        self.objective_func = objective_func
        self.lb = bounds[0]
        self.ub = bounds[1]
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w
        
        self.global_best_pos = None
        self.global_best_score = float('inf')
        self.convergence_curve = []

    def optimize(self):
        # Initialize positions and velocities
        population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        velocities = np.zeros((self.pop_size, self.dim))
        
        # Initialize personal bests
        pbest_pos = np.copy(population)
        pbest_scores = np.array([self.objective_func(ind) for ind in population])
        
        # Initialize global best
        min_idx = np.argmin(pbest_scores)
        self.global_best_score = pbest_scores[min_idx]
        self.global_best_pos = np.copy(pbest_pos[min_idx])
        
        self.convergence_curve.append(self.global_best_score)

        for t in range(self.max_iter - 1):
            for i in range(self.pop_size):
                # Update velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (self.w * velocities[i] + 
                                 self.c1 * r1 * (pbest_pos[i] - population[i]) + 
                                 self.c2 * r2 * (self.global_best_pos - population[i]))
                
                # Update position
                population[i] = population[i] + velocities[i]
                
                # Enforce bounds
                population[i] = np.clip(population[i], self.lb, self.ub)
                
                # Evaluate new position
                score = self.objective_func(population[i])
                
                # Update personal best
                if score < pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_pos[i] = np.copy(population[i])
                    
                    # Update global best
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_pos = np.copy(population[i])
            
            self.convergence_curve.append(self.global_best_score)
            
        return self.global_best_pos, self.global_best_score
