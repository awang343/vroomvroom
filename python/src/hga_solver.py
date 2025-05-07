from hga_local import LocalSearch
from hga_split import Split

import numpy as np
import copy
import random
import heapq
from hga_population import Population
from hga_crossover import HGACrossover
import time

from hga_structures import Individual, AlgoParams

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
        self.offspring = Individual(self)

    # }}}

    # {{{ solve
    def solve(self):
        self.population.generatePopulation()

        num_iter = 0
        num_iter_no_improvment = 1

        # if self.params.verbose:
        #     print("----- STARTING GENETIC ALGORITHM")

        # While we haven't exceeded the iteration limit and we are within the time limit
        start_time = time.time()
        while num_iter_no_improvment <= self.params.num_iter:

            # SELECTION AND CROSSOVER
            parent1 = self.population.getBinaryTournament()
            parent2 = self.population.getBinaryTournament()
            self.ox.crossover_ox(self.offspring, parent1, parent2)

            # LOCAL SEARCH
            self.local_searcher.run(self.offspring, self.penalty_capacity)
            is_new_best = self.population.addIndividual(self.offspring, True)

            # TODO: Check randint
            # Attempt to repair half the infeasible solutions
            if (
                not self.offspring.eval.is_feasible
                and random.random() < 0.5
            ):
                self.local_search.run(self.offspring, self.penalty_capacity * 10)
                if self.offspring.eval.is_feasible:
                    # We do not override isNewBest if the second add is new best
                    isNewBest = (
                        self.population.addIndividual(self.offspring, False)
                        or isNewBest
                    )

            # TRACKING ITERATIONS SINCE LAST IMPROVEMENT
            if isNewBest:
                num_iter_no_improvment = 1
            else:
                num_iter_no_improvment += 1

            # TODO: Check iteration penalty management
            # DIVERSIFICATION, PENALTY MANAGEMENT AND TRACES
            if num_iter % self.params.num_iter == 0:
                self.population.managePenalties()

            # if num_iter % self.params.num_iter_traces == 0:
            #     self.population.printState(num_iter, num_iter_no_improvment)

            # RESET THE ALGORITHM/POPULATION IF NO IMPROVEMENT
            if num_iter_no_improvment == self.params.num_iter:
                self.population.restart()
                num_iter_no_improvment = 1

            nbIter += 1

        # if self.params.verbose:
        #     elapsed = time.time() - startTime
        #     print(f"----- GENETIC ALGORITHM FINISHED AFTER {nbIter} ITERATIONS. "
        #             f"TIME SPENT: {elapsed:.2f} seconds.")

    # }}}
