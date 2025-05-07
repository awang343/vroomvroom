from hga_local import LocalSearch
from hga_split import Split

import numpy as np
import copy
import random
import heapq

from python.src.population import Population
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

        self.local_searcher = LocalSearch(self)
        self.splitter = Split(self)
        
        self.ox = HGACrossover(self)

        # Feasible heaps
        self.feasible_population = []
        self.feasible_elites = []

        # Infeasible heaps
        self.infeasible_population = []
        self.infeasible_elites = []
        
        self.population = Population(self)
        self.offspring = Individual(self.inst)
        
        self.algo_params = AlgoParams()

    
    def solve(self):
        #### filepath: genetic.py

        # INITIAL POPULATION
        self.population.generatePopulation()

        num_iter = 0
        num_iter_no_improvment = 1

        # if self.params.verbose:
        #     print("----- STARTING GENETIC ALGORITHM")

        # While we haven't exceeded the iteration limit and we are within the time limit
        startTime = time.time()
        while (num_iter_no_improvment <= self.algo_params.num_iter):

            # SELECTION AND CROSSOVER
            parent1 = self.population.getBinaryTournament()
            parent2 = self.population.getBinaryTournament()
            self.ox.crossover_ox(self.offspring, parent1, parent2)

            # LOCAL SEARCH
            self.local_searcher.run(self.offspring, self.solver.penalty_capacity)
            isNewBest = self.population.addIndividual(self.offspring, True)

            #TODO: Check randint
            # Attempt to repair half the infeasible solutions
            if (not self.offspring.eval.is_feasible
                and self.params.ran.randint(0, 1) == 0):
                self.local_searcher.run(self.offspring,
                                        self.solver.penalty_capacity * 10.0)
                if self.offspring.eval.is_feasible:
                    # We do not override isNewBest if the second add is new best
                    isNewBest = self.population.addIndividual(self.offspring, False) or isNewBest

            # TRACKING ITERATIONS SINCE LAST IMPROVEMENT
            if isNewBest:
                num_iter_no_improvment = 1
            else:
                num_iter_no_improvment += 1

            #TODO: Check iteration penalty management
            # DIVERSIFICATION, PENALTY MANAGEMENT AND TRACES
            if num_iter % self.algo_params.num_iter == 0:
                self.population.managePenalties()

            # if num_iter % self.algo_params.num_iter_traces == 0:
            #     self.population.printState(num_iter, num_iter_no_improvment)

            # RESET THE ALGORITHM/POPULATION IF NO IMPROVEMENT
            if (num_iter_no_improvment == self.algo_params.num_iter):
                self.population.restart()
                num_iter_no_improvment = 1

            nbIter += 1

        # if self.params.verbose:
        #     elapsed = time.time() - startTime
        #     print(f"----- GENETIC ALGORITHM FINISHED AFTER {nbIter} ITERATIONS. "
        #             f"TIME SPENT: {elapsed:.2f} seconds.")


# class HGASolver:
#     # {{{ __init__
#     def __init__(self, inst, params):
#         self.inst = inst
#         self.params = params
#         self.capacity_penalty = self.inst.default_capacity_penalty

#         self.inst.initNeighbors(params.neighborhood_size)

#         self.local_searcher = LocalSearch(self)
#         self.splitter = Split(self)

#         # Feasible heaps
#         self.feasible_population = []
#         self.feasible_elites = []

#         # Infeasible heaps
#         self.infeasible_population = []
#         self.infeasible_elites = []
#     # }}}

#     # {{{ make_random_solution
#     def make_random_solution(self):
#         """
#         Randomly generate a solution
#         """
#         customers = np.array(self.all_customers)
#         k = self.inst.numVehicles

#         np.random.shuffle(customers)
#         split_points = np.sort(
#             np.random.choice(range(1, len(customers)), size=k - 1, replace=False)
#         )
#         split_points = np.concatenate(([0], split_points, [len(customers)]))
#         routes = [
#             customers[split_points[i] : split_points[i + 1]].tolist() for i in range(k)
#         ]

#         self.insert_solution(routes)

#     # }}}

#     # {{{ make_mating_solution
#     def make_mating_solution(self):
#         p1, p2 = self.select_parents()
#         offspring = self.crossover(p1, p2)
#         self.insert_solution(offspring)
#         self.cull_population()

#     # }}}

#     # {{{ insert_solution
#     def insert_solution(self, dirty_routes):
#         fitness, routes = self.educate(dirty_routes)
#         if all(self.inst.calc_feasible(route) for route in routes):
#             # Feasible
#             heapq.heappush(self.feasible_population, (-fitness, routes))
#         else:
#             # Infeasible
#             heapq.heappush(self.infeasible_population, (-fitness, routes))

#             if random.random() < self.repair_prob:
#                 original_penalty = self.capacity_penalty
#                 for _ in range(2):
#                     self.capacity_penalty *= 10
#                     fitness, routes = self.educate(routes)
#                     if all(self.inst.calc_feasible(route) for route in routes):
#                         heapq.heappush(self.feasible_population, (-fitness, routes))
#                         break
#                 self.capacity_penalty = original_penalty

#         self.cull_population()

#     # }}}

#     # {{{ select_parents
#     def select_parents(self):
#         """
#         Selects two parents using binary tournament selection with elitism bias.
#         """

#         parents = []
#         for parent in range(2):
#             """Run a single binary tournament"""
#             candidates = random.sample(
#                 self.feasible_population + self.infeasible_population, 2
#             )
#             parents.append(max(candidates)[1])

#         return parents

#     # }}}

#     # {{{ crossover
#     def crossover(self, parent1, parent2):
#         """
#         Performs PIX crossover for VRP with a single depot and multiple vehicles.

#         Each customer is visited exactly once. Routes are lists of customer IDs.
#         parent1, parent2: list of routes (list of lists)
#         num_vehicles: number of vehicles (routes per individual)

#         Returns a new child individual (list of routes).
#         """
#         child = [[] for _ in range(self.inst.numVehicles)]
#         visited = set()
#         customers = set(self.all_customers)

#         # Step 0: Split vehicles into Λ1, Λ2, Λmix
#         n1 = random.randint(0, self.inst.numVehicles)
#         n2 = random.randint(0, self.inst.numVehicles)
#         n1, n2 = sorted((n1, n2))

#         idxs = list(range(self.inst.numVehicles))
#         random.shuffle(idxs)

#         lambda1 = idxs[:n1]
#         lambda2 = idxs[n1:n2]
#         lambda_mix = idxs[n2:]

#         # Step 1a: Copy full routes from P1 for Λ1
#         for k in lambda1:
#             for cust in parent1[k]:
#                 if cust not in visited:
#                     child[k].append(cust)
#                     visited.add(cust)

#         # Step 1b: Copy substrings from P1 for Λmix
#         for k in lambda_mix:
#             route = parent1[k]
#             if not route:
#                 continue
#             a, b = (
#                 sorted(random.sample(range(len(route)), 2))
#                 if len(route) >= 2
#                 else (0, len(route))
#             )
#             for cust in route[a:b]:
#                 if cust not in visited:
#                     child[k].append(cust)
#                     visited.add(cust)

#         # Step 2: Fill in from P2 (Λ2 and Λmix)
#         for k in lambda2 + lambda_mix:
#             for cust in parent2[k]:
#                 if cust not in visited:
#                     child[k].append(cust)
#                     visited.add(cust)

#         # Step 3: Repair phase — insert missing customers
#         unvisited = customers - visited
#         for cust in unvisited:
#             # Choose the route with minimal insertion cost (dummy: smallest route)
#             best_route_idx = min(
#                 range(self.inst.numVehicles), key=lambda k: len(child[k])
#             )
#             # Insert at best position (simplified: at end)
#             child[best_route_idx].append(cust)
#             visited.add(cust)

#         return child

#     # }}}

#     # {{{ cull_population
#     def cull_population(self):
#         """
#         Culls population down to mu individuals from mu+lambda
#         """
#         if len(self.feasible_population) >= self.params.population_size + self.params.generation_size:
#             while len(self.feasible_population) > self.params.population_size:
#                 # Pop from feasible population
#                 heapq.heappop(self.feasible_population)

#         if (
#             len(self.infeasible_population)
#             >= self.population_size + self.generation_size
#         ):
#             while len(self.infeasible_population) > self.population_size:
#                 # Pop from infeasible population
#                 heapq.heappop(self.infeasible_population)

#     # }}}

#     # {{{ nuke_population
#     def nuke_population():
#         """
#         Resets all but mu/3 individuals
#         """
#         new_feasible_population = []
#         new_infeasible_population = []

#         for individual in self.feasible_population:
#             if self.feasible_queue_miu_3[0][0] <= individual[0]:
#                 new_feasible_population.append(individual[0])

#         for indvidual in self.infeasible_population:
#             if self.infeasible_queue_miu_3[0][0] <= individual[0]:
#                 new_feasible_population.append(individual[0])

#         # TODO: Repopulate with random individuals
#         return (new_feasible_population, new_infeasible_population)

#     # }}}

#     def solve(self):
#         for it in range(self.params.population_size * 4):
#             print("Initial Generation Iteration:", it)
#             self.make_random_solution()

#             best_cost, best_routes = (
#                 max(self.feasible_population)
#                 if self.feasible_population
#                 else (None, None)
#             )
#             print(best_cost, best_routes)
#         print("Population initialized")

#         _mating_iteration = 0

#         no_improvement = 0
#         while no_improvement < 100:
#             print("Mating Iteration:", _mating_iteration)
#             _mating_iteration += 1

#             self.make_mating_solution()
#             new_cost, new_routes = (
#                 max(self.feasible_population)
#                 if self.feasible_population
#                 else (None, None)
#             )

#             if best_cost is None or new_cost is None or np.isclose(new_cost, best_cost):
#                 no_improvement += 1
#             else:
#                 best_cost, best_routes = new_cost, new_routes
#                 print("\tNew best cost:", new_cost)
#                 no_improvement = 0

#         return -best_cost, 0, best_routes
