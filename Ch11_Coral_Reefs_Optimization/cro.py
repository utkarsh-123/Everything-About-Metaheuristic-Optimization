import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_algorithm import BaseAlgorithm

class CRO(BaseAlgorithm):
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=30):
        super().__init__(objective_func, bounds, dim, pop_size, max_iter)
        self.rho0 = 0.6  # Initial occupation rate
        self.Fb = 0.8    # Spawning fraction
        self.Fa = 0.1    # Asexual reproduction fraction
        self.Fd = 0.1    # Depredation fraction
        self.Pd = 0.1    # Depredation probability
        self.k_attempts = 3
        # Grid dimensions (N x M >= pop_size)
        self.grid_N = int(np.ceil(np.sqrt(pop_size / self.rho0)))
        self.grid_M = self.grid_N
        self.total_cells = self.grid_N * self.grid_M

    def _settle_larva(self, larva, larva_fit, grid, grid_fit, occupied):
        for _ in range(self.k_attempts):
            cell_idx = np.random.randint(0, self.total_cells)
            if not occupied[cell_idx]:
                grid[cell_idx] = larva
                grid_fit[cell_idx] = larva_fit
                occupied[cell_idx] = True
                return True
            elif larva_fit < grid_fit[cell_idx]:
                grid[cell_idx] = larva
                grid_fit[cell_idx] = larva_fit
                return True
        return False

    def optimize(self):
        # Initialize grid
        grid = np.zeros((self.total_cells, self.dim))
        grid_fit = np.full(self.total_cells, np.inf)
        occupied = np.zeros(self.total_cells, dtype=bool)
        
        # Initial population
        initial_pop_size = int(self.rho0 * self.total_cells)
        initial_positions = np.random.uniform(self.lb, self.ub, (initial_pop_size, self.dim))
        initial_fitness = self.evaluate(initial_positions)
        
        # Settle initial population
        available_indices = np.random.choice(self.total_cells, initial_pop_size, replace=False)
        for i, idx in enumerate(available_indices):
            grid[idx] = initial_positions[i]
            grid_fit[idx] = initial_fitness[i]
            occupied[idx] = True

        self.global_best_score = np.min(grid_fit[occupied])
        self.global_best_pos = np.copy(grid[occupied][np.argmin(grid_fit[occupied])])
        self.convergence_curve.append(self.global_best_score)

        for t in range(self.max_iter - 1):
            active_indices = np.where(occupied)[0]
            num_active = len(active_indices)
            
            if num_active < 2:
                break
                
            # 1. Broadcast Spawning
            num_spawners = int(np.round(self.Fb * num_active))
            # ensure even number of spawners
            if num_spawners % 2 != 0:
                num_spawners -= 1
                
            spawner_indices = np.random.choice(active_indices, num_spawners, replace=False)
            brooder_indices = np.setdiff1d(active_indices, spawner_indices)
            
            larvae = []
            
            for i in range(0, num_spawners, 2):
                p1 = grid[spawner_indices[i]]
                p2 = grid[spawner_indices[i+1]]
                crossover_mask = np.random.rand(self.dim) < 0.5
                larva = np.where(crossover_mask, p1, p2)
                larvae.append(larva)
                
            # 2. Brooding
            for idx in brooder_indices:
                p = grid[idx]
                mutation_mask = np.random.rand(self.dim) < 0.1
                mutation_vals = p + np.random.randn(self.dim) * 0.1 * (self.ub - self.lb)
                larva = np.where(mutation_mask, mutation_vals, p)
                larvae.append(larva)
                
            # 3. Larvae Setting
            if larvae:
                larvae = np.array(larvae)
                larvae = self.enforce_bounds(larvae)
                larvae_fit = self.evaluate(larvae)
                
                for larva, fit in zip(larvae, larvae_fit):
                    self._settle_larva(larva, fit, grid, grid_fit, occupied)
                    
            # 4. Asexual Reproduction
            active_indices = np.where(occupied)[0]
            num_active = len(active_indices)
            if num_active > 0:
                sorted_active = active_indices[np.argsort(grid_fit[active_indices])]
                num_clones = int(np.round(self.Fa * num_active))
                for idx in sorted_active[:num_clones]:
                    self._settle_larva(np.copy(grid[idx]), grid_fit[idx], grid, grid_fit, occupied)

            # 5. Depredation
            active_indices = np.where(occupied)[0]
            num_active = len(active_indices)
            if num_active > 0:
                sorted_active = active_indices[np.argsort(grid_fit[active_indices])]
                num_depredation = int(np.round(self.Fd * num_active))
                weak_indices = sorted_active[-num_depredation:]
                
                for idx in weak_indices:
                    if np.random.rand() < self.Pd:
                        occupied[idx] = False
                        grid_fit[idx] = np.inf
                        
            active_indices = np.where(occupied)[0]
            if len(active_indices) > 0:
                current_best_idx = active_indices[np.argmin(grid_fit[active_indices])]
                if grid_fit[current_best_idx] < self.global_best_score:
                    self.global_best_score = grid_fit[current_best_idx]
                    self.global_best_pos = np.copy(grid[current_best_idx])
                    
            self.convergence_curve.append(self.global_best_score)

        return self.global_best_pos, self.global_best_score
