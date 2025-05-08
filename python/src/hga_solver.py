from hga_local import LocalSearch
from hga_split import Split
from hga_population import Population
from hga_crossover import HGACrossover
from hga_structures import Individual, AlgoParams

import numpy as np
import copy
import random
import heapq
import time

# random.seed(10)
# np.random.seed(10)

"""
We will represent the solution in an n-vehicle problem
as [[route1], [route2], [route3], ..., [routen]]
"""


class HGASolver:
    # {{{ __init__
    def __init__(self, inst, params):
        self.inst = inst
        self.params = params
        self.capacity_penalty = self.inst.default_capacity_penalty

        self.inst.initNeighbors(params.neighborhood_size)

        self.local_search = LocalSearch(self)
        self.splitter = Split(self)

        self.ox = HGACrossover(self)

        self.population = Population(self)

    # }}}

    # {{{ solve
    def solve(self):
        start_time = time.time()

        self.population.generatePopulation()

        num_iter = 0
        num_iter_no_improvement = 1

        while num_iter_no_improvement <= self.params.num_iter:

            # SELECTION AND CROSSOVER
            parent1 = self.population.getBinaryTournament()
            parent2 = self.population.getBinaryTournament()
            offspring = self.ox.crossover_ox(parent1, parent2)

            # LOCAL SEARCH
            self.local_search.run(offspring, self.capacity_penalty)
            is_new_best = self.population.addIndividual(offspring, True)

            # Attempt to repair half the infeasible solutions
            if not offspring.eval.is_feasible and random.random() < 0.5:
                self.local_search.run(offspring, self.capacity_penalty * 10)
                if offspring.eval.is_feasible:
                    # We do not override isNewBest if the second add is new best
                    is_new_best = (
                        self.population.addIndividual(offspring, False)
                        or is_new_best
                    )

            # TRACKING ITERATIONS SINCE LAST IMPROVEMENT
            if is_new_best:
                num_iter_no_improvement = 1
            else:
                num_iter_no_improvement += 1

            # DIVERSIFICATION, PENALTY MANAGEMENT
            if num_iter % self.params.num_iter_until_penalty == 0:
                self.population.managePenalties()

            if time.time() - start_time > self.params.max_time:
                break
        return self.population.best_solution_overall.eval.distance, 0, self.population.best_solution_overall.chromR
    # }}}
