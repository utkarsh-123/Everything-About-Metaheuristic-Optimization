import numpy as np

class BaseAlgorithm:
    def __init__(self, objective_func, bounds, dim, pop_size=50, max_iter=5):
        """
        Base class for metaheuristic algorithms.
        
        :param objective_func: The objective function to minimize.
        :param bounds: A tuple of (lower_bound, upper_bound), e.g., (-100, 100).
        :param dim: The dimensionality of the problem.
        :param pop_size: Number of agents in the population.
        :param max_iter: Maximum number of iterations (generations).
        """
        self.objective_func = objective_func
        self.lb = bounds[0]
        self.ub = bounds[1]
        self.dim = dim
        self.pop_size = pop_size
        self.max_iter = max_iter
        
        self.global_best_pos = None
        self.global_best_score = float('inf')
        self.convergence_curve = []

    def initialize_population(self):
        """
        Initializes a random population within the bounds.
        """
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def evaluate(self, population):
        """
        Evaluates the fitness of the entire population.
        """
        return np.array([self.objective_func(ind) for ind in population])

    def enforce_bounds(self, population):
        """
        Clips the population to the defined search space boundaries.
        """
        return np.clip(population, self.lb, self.ub)

    def update_global_best(self, population, fitness):
        """
        Updates the global best solution found so far.
        """
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.global_best_score:
            self.global_best_score = fitness[min_idx]
            self.global_best_pos = population[min_idx].copy()

    def optimize(self):
        """
        The main optimization loop to be implemented by child classes.
        Must populate `self.convergence_curve` and `self.global_best_score`.
        """
        raise NotImplementedError("optimize() must be implemented in the child class.")
