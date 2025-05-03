from scipy.spatial import distance
from deap import base, creator, tools

import numpy as np
import copy
import random

creator.create("Fitness_Func", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.Fitness_Func)

"""
We will represent the solution in an n-vehicle problem
as [[route1], [route2], [route3], ..., [routen]]
"""


class HGASolver:

    def __init__(
        self,
        inst,
        population_size=25,
        generation_size=40,
        education_prob=1.0,
        repair_prob=0.5,
        ri_granularity=0.4,
        feasibility_target=0.2,
    ):
        self.tb = base.Toolbox()
        self.inst = inst

        self.population_size = population_size
        self.generation_size = generation_size

        self.education_prob = education_prob
        self.repair_prob = repair_prob
        self.ri_granularity = ri_granularity

        self.feasibility_target = feasiblity_target

        self._register()

    def _register(self):
        self.tb.register("indexes", self.chromo_create)
        self.tb.register(
            "individual", tools.initIterate, creator.Individual, self.tb.indexes
        )
        self.tb.register("population", tools.initRepeat, list, self.tb.individual)

        self.tb.register("evaluate", self.chromo_eval)

        self.tb.register("select", tools.selTournament)
        self.tb.register("mate", self.crossover)
        self.tb.register("mutate", self.mutation)


    def pix_crossover(self, parent1, parent2):
        """
        Performs PIX crossover for VRP with a single depot and multiple vehicles.
        
        Each customer is visited exactly once. Routes are lists of customer IDs.
        parent1, parent2: list of routes (list of lists)
        num_vehicles: number of vehicles (routes per individual)
        
        Returns a new child individual (list of routes).
        """
        assert len(parent1) == num_vehicles
        assert len(parent2) == num_vehicles

        child = [[] for _ in range(num_vehicles)]
        visited = set()
        all_customers = set(c for route in parent1 for c in route)

        # Step 0: Split vehicles into Λ1, Λ2, Λmix
        n1, n2 = sorted(random.sample(range(0, num_vehicles + 1), 2))
        idxs = list(range(num_vehicles))
        random.shuffle(idxs)

        lambda1 = idxs[:n1]
        lambda2 = idxs[n1:n2]
        lambda_mix = idxs[n2:]

        # Step 1a: Copy full routes from P1 for Λ1
        for k in lambda1:
            for cust in parent1[k]:
                if cust not in visited:
                    child[k].append(cust)
                    visited.add(cust)

        # Step 1b: Copy substrings from P1 for Λmix
        for k in lambda_mix:
            route = parent1[k]
            if not route:
                continue
            a, b = sorted(random.sample(range(len(route)), 2)) if len(route) >= 2 else (0, len(route))
            for cust in route[a:b]:
                if cust not in visited:
                    child[k].append(cust)
                    visited.add(cust)

        # Step 2: Fill in from P2 (Λ2 and Λmix)
        for k in lambda2 + lambda_mix:
            for cust in parent2[k]:
                if cust not in visited:
                    child[k].append(cust)
                    visited.add(cust)

        # Step 3: Repair phase — insert missing customers
        unvisited = all_customers - visited
        for cust in unvisited:
            # Choose the route with minimal insertion cost (dummy: smallest route)
            best_route_idx = min(range(num_vehicles), key=lambda k: len(child[k]))
            # Insert at best position (simplified: at end)
            child[best_route_idx].append(cust)
            visited.add(cust)

        return child

    def chromo_create(self):
        # Creates an individual randomly
        schedule = copy.deepcopy(self.customers)
        vehicle = list(np.random.randint(self.inst.numVehicles, size=(len(schedule))))
        np.random.shuffle(schedule)
        chromo = [schedule, vehicle]
        return chromo

    def chromo_eval(self, _chromo):
        route_set = [[] for _ in range(self.inst.numVehicles)]
        for s, v in zip(_chromo[0], _chromo[1]):
            route_set[v].append(s)

        dist = 0
        penalty = 0
        for route in route_set:
            dist += self.calc_route_cost(route)
            penalty += self.capacity_penalty * max(
                0,
                sum(self.inst.demandOfCustomer[i] for i in route)
                - self.inst.vehicleCapacity,
            )

        return dist + penalty, penalty > 0

    def get_route(self, _chromo):
        route_set = [[] for _ in range(self.inst.numVehicles)]
        for s, v in zip(_chromo[0], _chromo[1]):
            route_set[v].append(s)
        return route_set

    def calc_route_cost(self, _route):
        if not _route:
            return 0
        dist = self.inst.distances[_route[-1], 0] + self.inst.distances[0, _route[0]]

        for p in range(len(_route) - 1):
            _i = _route[p]
            _j = _route[p + 1]
            dist += self.inst.distances[_i][_j]
        return dist

    def crossover(self, _chromo1, _chromo2):
        cuts = self.get_chromo_cut()
        self.partial_crossover(_chromo1[0], _chromo2[0], cuts)

        cuts1 = self.get_chromo_cut()
        cuts2 = self.get_chromo_cut(cuts1[2])

        self.swap_genes(_chromo1[1], _chromo2[1], cuts1, cuts2)

    def partial_crossover(self, _chromo1, _chromo2, cuts):
        size = len(_chromo1)
        p1, p2 = [0] * size, [0] * size

        for i in range(size):
            p1[_chromo1[i] - 1] = i
            p2[_chromo2[i] - 1] = i

        for i in range(cuts[0], cuts[1]):
            temp1 = _chromo1[i] - 1
            temp2 = _chromo2[i] - 1

            _chromo1[i], _chromo1[p1[temp2]] = temp2 + 1, temp1 + 1
            _chromo2[i], _chromo2[p2[temp1]] = temp1 + 1, temp2 + 1

            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

    def get_chromo_cut(self, cut_range=None, mutation=False):
        if mutation:
            randrange = self.inst.numCustomers - 1
        else:
            randrange = self.inst.numCustomers

        if cut_range is None:
            cut1 = random.randrange(randrange)
            cut2 = random.randrange(randrange)
            if cut1 > cut2:
                tmp = cut2
                cut2 = cut1
                cut1 = tmp
            cut_range = cut2 - cut1
        else:
            cut1 = random.randrange(self.inst.numCustomers - cut_range)
            cut2 = cut1 + cut_range
        return cut1, cut2, cut_range

    def swap_genes(self, chrom1, chrom2, cuts1, cuts2):
        tmp = chrom1[cuts1[0] : cuts1[1]]
        chrom1[cuts1[0] : cuts1[1]] = chrom2[cuts2[0] : cuts2[1]]
        chrom2[cuts2[0] : cuts2[1]] = tmp

    def mutation(self, _chromo):
        if np.random.rand() < 0.5:
            self.swap_gene(_chromo)
        else:
            self.shuffle_gene(_chromo)

    def swap_gene(self, _chromo):
        cuts = self.get_chromo_cut(mutation=True)

        if np.random.rand() < 0.5:
            _chromo[0][cuts[0]], _chromo[0][cuts[1]] = (
                _chromo[0][cuts[1]],
                _chromo[0][cuts[0]],
            )
        else:
            _chromo[1][cuts[0]], _chromo[1][cuts[1]] = (
                _chromo[1][cuts[1]],
                _chromo[1][cuts[0]],
            )

    def shuffle_gene(self, _chromo):
        cuts = self.get_chromo_cut(mutation=True)

        if np.random.rand() < 0.5:
            to_mix = _chromo[0][cuts[0] : cuts[1]]
            np.random.shuffle(to_mix)
            _chromo[0][cuts[0] : cuts[1]] = to_mix
        else:
            to_mix = _chromo[1][cuts[0] : cuts[1]]
            np.random.shuffle(to_mix)
            _chromo[1][cuts[0] : cuts[1]] = to_mix

    def solve(self):
        population = self.tb.population(n=self.population_size)
        fitness_set = list(self.tb.map(self.tb.evaluate, population))

        for ind, fit in zip(population, fitness_set):
            ind.fitness.values = (fit[0],)

        best_fit_list = []
        best_sol_list = []

        best_fit = np.inf

        for gen in range(0, self.num_generations):
            if gen % 100 == 0:
                print(f"Generation: {gen:4} | Fitness: {best_fit:.2f}")

            offspring = self.tb.select(population, len(population), tournsize=3)
            offspring = list(map(self.tb.clone, offspring))

            for child1, child2 in zip(offspring[0::2], offspring[1::2]):
                if np.random.random() < self.prob_crossover:
                    self.tb.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for chromo in offspring:
                if np.random.random() < self.prob_mutation:
                    self.tb.mutate(chromo)
                    del chromo.fitness.values

            fitness_set = map(self.tb.evaluate, offspring)
            feasible_offspring = []
            for ind, fit in zip(offspring, fitness_set):
                ind.fitness.values = (fit[0],)
                if not fit[1] or True:
                    feasible_offspring.append(ind)

            population[:] = offspring

            if len(feasible_offspring) > 0:
                curr_best_sol = tools.selBest(feasible_offspring, 1)[0]
                curr_best_fit = curr_best_sol.fitness.values[0]

                if curr_best_fit < best_fit:
                    best_sol = curr_best_sol
                    best_fit = curr_best_fit

                best_fit_list.append(best_fit)
                best_sol_list.append(best_sol)

        return curr_best_fit, 0, self.get_route(best_sol)
