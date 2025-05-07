import random
import copy
from typing import List, Dict
from hga_structures import Individual
from collections import deque


class Population:
    def __init__(self, solver):
        self.solver = solver
        self.inst = solver.inst
        self.params = solver.params
        self.splitter = solver.splitter
        self.local_search = solver.local_search

        # Feasible heaps
        self.feasible_population = []
        self.feasible_elites = []

        # Infeasible heaps
        self.infeasible_population = []
        self.infeasible_elites = []

        # Best solution trackers
        self.best_solution_restart = Individual(self.solver)
        self.best_solution_overall = Individual(self.solver)

        # Tracking feasibility history
        self.recent_feasibility = deque(
            [True] * self.params.num_iter_until_penalty,
            maxlen=self.params.num_iter_until_penalty,
        )

    def generatePopulation(self):
        max_iter = 4 * self.params.population_size
        for i in range(max_iter):
            # Create random individual
            indiv = Individual(self.solver)
            # Split into routes
            self.splitter.run(indiv)
            # Local search
            self.local_search.run(indiv, self.solver.capacity_penalty)
            # Add the new individual
            self.addIndividual(indiv, update_feasible=True)
            # Attempt to repair if infeasible
            if not indiv.eval.is_feasible and random.random() < self.params.repair_prob:
                self.local_search.run(indiv, self.solver.capacity_penalty * 10)
                if indiv.eval.is_feasible:
                    self.addIndividual(indiv, update_feasible=False)

    def addIndividual(self, indiv: Individual, update_feasible: bool) -> bool:
        """
        Inserts an individual into the appropriate subpop,
        triggers a survivor selection if needed, and updates best solutions.
        """
        # Update recent feasibility tracker
        if update_feasible:
            self.recent_feasibility.append(indiv.eval.capacity_excess < 1e-3)

        subpop = (
            self.feasible_population
            if indiv.eval.is_feasible
            else self.infeasible_population
        )
        indiv_copy = copy.copy(indiv)

        # Update proximity with existing subpop members
        for other in subpop:
            dist = self.brokenPairsDistance(indiv_copy, other)

            # Check
            indiv_copy.proximity_indivs.add((dist, other))
            other.proximity_indivs.add((dist, indiv_copy))

        # Insert in ascending order of penalizedCost
        place = len(subpop)
        while (
            place > 0
            and subpop[place - 1].eval.penalized_cost
            > indiv_copy.eval.penalized_cost - 1e-3
        ):
            place -= 1
        subpop.insert(place, indiv_copy)

        # Survivor selection if needed
        max_size = self.params.population_size + self.params.generation_size
        while len(subpop) > max_size:
            self.removeWorstBiasedFitness(subpop)

        # Update best solutions if feasible
        if indiv.eval.is_feasible:
            if (
                indiv.eval.penalized_cost
                < self.best_solution_restart.eval.penalized_cost - 1e-3
            ):
                self.best_solution_restart = indiv
                if (
                    indiv.eval.penalized_cost
                    < self.best_solution_overall.eval.penalized_cost - 1e-3
                ):
                    self.best_solution_overall = indiv
                return True
        return False

    def removeWorstBiasedFitness(self, pop: List[Individual]):
        """
        Replicates removeWorstBiasedFitness from Population.cpp.
        Updates biased fitnesses, then removes the worst. Also accounts for clones.
        """
        self.updateBiasedFitnesses(pop)
        if len(pop) <= 1:
            raise Exception("Eliminating the best individual: should not occur in HGS")

        worstInd = None
        worstPos = -1
        worstIsClone = False
        worstBF = -1e30

        for i in range(1, len(pop)):
            isClone = self.averageBrokenPairsDistanceClosest(pop[i], 1) < 1e-3
            if (isClone and not worstIsClone) or (
                isClone == worstIsClone and pop[i].biasedFitness > worstBF
            ):
                worstBF = pop[i].biasedFitness
                worstIsClone = isClone
                worstInd = pop[i]
                worstPos = i

        if worstPos < 0:
            worstPos = len(pop) - 1
            worstInd = pop[worstPos]

        # Remove from pop
        removed = pop.pop(worstPos)

        # Cleanup references
        for indiv2 in pop:
            toRemove = None
            for dval, ref in indiv2.proximity_indivs:
                if ref is removed:
                    toRemove = (dval, ref)
                    break
            if toRemove:
                print(toRemove)
                print(indiv2.proximity_indivs)
                indiv2.proximity_indivs.remove(toRemove)

    def brokenPairsDistance(self, indiv1: Individual, indiv2: Individual) -> float:
        """Calculates the broken pairs distance between two individuals."""
        differences = 0
        for j in range(1, self.inst.num_customers):
            if (
                indiv1.successors[j] != indiv2.successors[j]
                and indiv1.successors[j] != indiv2.predecessors[j]
            ):
                differences += 1
            if (
                indiv1.predecessors[j] == 0
                and indiv2.predecessors[j] != 0
                and indiv2.successors[j] != 0
            ):
                differences += 1
        return differences / self.inst.num_customers

    def averageBrokenPairsDistanceClosest(
        self, indiv: Individual, num_closest: int
    ) -> float:
        """Calculates the average broken pairs distance to the closest neighbors."""
        max_size = min(num_closest, len(indiv.proximity_indivs))
        result = sum(x[0] for x in indiv.proximity_indivs[:max_size])
        return result / max_size if max_size > 0 else 0.0

    def updateBiasedFitnesses(self, pop: List[Individual]):
        """Updates the biased fitness values for a population."""
        # Rank individuals based on diversity contribution (decreasing order of distance)
        ranking = []
        for i, indiv in enumerate(pop):
            diversity = -self.averageBrokenPairsDistanceClosest(
                indiv, self.params.num_close
            )
            ranking.append((diversity, i))
        ranking.sort()  # Sort by diversity (descending)

        # Update biased fitness values
        if len(pop) == 1:
            pop[0].biasedFitness = 0.0
        else:
            for rank, (_, idx) in enumerate(ranking):
                divRank = rank / (len(pop) - 1)  # Diversity rank (0 to 1)
                fitRank = idx / (len(pop) - 1)  # Fitness rank (0 to 1)
                if len(pop) <= self.params.num_elite:
                    pop[idx].biasedFitness = fitRank
                else:
                    pop[idx].biasedFitness = (
                        fitRank + (1.0 - self.params.num_elite / len(pop)) * divRank
                    )

    def getBinaryTournament(self) -> Individual:
        """Selects two individuals and returns the one with better biased fitness."""
        # Combine feasible and infeasible subpopulations
        combined_pop = self.feasible_population + self.infeasible_population
        if len(combined_pop) < 2:
            raise ValueError(
                "Population size must be at least 2 for binary tournament."
            )

        # Randomly select two individuals
        indiv1 = random.choice(combined_pop)
        indiv2 = random.choice(combined_pop)

        # Update biased fitnesses
        self.updateBiasedFitnesses(self.feasible_population)
        self.updateBiasedFitnesses(self.infeasible_population)

        # Return the individual with better (lower) biased fitness
        return indiv1 if indiv1.biasedFitness < indiv2.biasedFitness else indiv2

    def restart(self):
        # if self.params.verbose:
        #     print("----- RESET: CREATING A NEW POPULATION -----")
        # for indiv in self.feasible_population:
        #     # In C++, we delete the pointer. Here, there's no explicit memory free needed.
        #     pass
        # for indiv in self.infeasible_population:
        #     pass

        self.feasible_population.clear()
        self.infeasible_population.clear()
        self.bestSolutionRestart = Individual(self.inst)
        # self.bestSolutionRestart.eval['penalizedCost'] = 1e30
        self.generatePopulation()

    def managePenalties(self):
        """
        Matches Population::managePenalties logic in the code snippet.
        """
        fraction_feasible = sum(self.recent_feasibility) / float(
            len(self.recent_feasibility)
        )

        if (
            fractionFeasibleLoad < self.params.feasibility_target - 0.05
            and self.solver.penalty_capacity < 100000.0
        ):
            self.solver.penalty_capacity = min(
                self.solver.penalty_capacity * self.params.penalty_increase,
                100000.0,
            )
        elif (
            fractionFeasibleLoad > self.params.feasibility_target + 0.05
            and self.solver.penalty_capacity > 0.1
        ):
            self.solver.penalty_capacity = max(
                self.params.penaltyCapacity * self.params.penalty_decrease, 0.1
            )

        # Update penalized cost for infeasible
        for indiv in self.infeasible_population:
            indiv.eval.penalized_cost = (
                indiv.eval.distance
                + self.solver.penalty_capacity * indiv.eval.capacity_excess
            )

        # Reorder via bubble sort for demonstration
        n = len(self.infeasible_population)
        for i in range(n):
            for j in range(n - i - 1):
                if (
                    self.infeasible_population[j].eval.penalized_cost
                    > self.infeasible_population[j + 1].eval.penalized_cost + 1e-3
                ):
                    (
                        self.infeasible_population[j],
                        self.infeasible_population[j + 1],
                    ) = (
                        self.infeasible_population[j + 1],
                        self.infeasible_population[j],
                    )
