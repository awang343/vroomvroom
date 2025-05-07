import random
from typing import List, Dict
from hga_structures import AlgoParams
from hga_structures import Individual
import time
from collections import deque

MY_EPSILON = 0.00001

class Population:
    def __init__(self, solver):
        self.solver = solver
        self.capacity_penalty = solver.capacity_penalty
        self.inst = solver.inst
        self.algo_params = AlgoParams()
        self.splitter = solver.splitter
        self.localSearch = solver.localSearch
        
        self.bestSolutionRestart = Individual(self.inst)
        self.bestSolutionOverall = Individual(self.inst)
        
        # Tracking feasibility
        
        #TODO: Check
        self.listFeasibilityLoad = deque([True] * self.algo_params.num_iter_until_penalty,
                                         maxlen=self.algo_params.num_iter_until_penalty)
        # self.listFeasibilityDuration = deque([True] * self.params.ap['nbIterPenaltyManagement'],
        #                                      maxlen=self.params.ap['nbIterPenaltyManagement'])
    
    
    def generatePopulation(self):
        # if self.params.verbose:
        #     print("----- BUILDING INITIAL POPULATION")
        # Try up to 4*mu attempts
        maxIter = 4 * self.algo_params.population_size
        tStart = time.time()
        for i in range(maxIter):
            # Check time-limit
            # if i > 0 and self.params.ap['timeLimit'] > 0:
            #     if (time.time() - tStart) > self.params.ap['timeLimit']:
            #         break
            # Create random individual
            indiv = Individual(self.inst)
            # Create a random giant tour
            # Typically: a permutation of [1..nbClients]. We'll just fill them for demonstration:
            indiv.chromT = list(range(1, self.inst.num_customers + 1))
            random.shuffle(indiv.chromT)
            # Split into routes
            self.splitter.run(indiv)
            # Local search
            self.localSearch.run(indiv, self.capacity_penalty)
            # Add the new individual
            self.addIndividual(indiv, updateFeasible=True)
            # Attempt to repair if infeasible
            
            #TODO: Check
            if not indiv.eval.is_feasible and self.params.ran.randint(0, 1) == 0:
                self.localSearch.run(indiv, self.capacity_penalty)
                if indiv.eval.is_feasible:
                    self.addIndividual(indiv, updateFeasible=False)
    
    #TODO: Check func
    def addIndividual(self, indiv: Individual, updateFeasible: bool) -> bool:
        """
        Inserts an individual into the appropriate subpop,
        triggers a survivor selection if needed, and updates best solutions.
        """
        # Update feasibility trackers
        if updateFeasible:
            self.listFeasibilityLoad.append(indiv.eval.capacity_excess < MY_EPSILON)
            # self.listFeasibilityDuration.append(indiv.eval['durationExcess'] < MY_EPSILON)

        #TODO: Check feasibleSubpop and infeasibleSubpop
        subpop = self.solver.feasible_population if indiv.eval.is_feasible else self.solver.infeasible_population
        myIndividual = Individual(self.inst)
        myIndividual.copyFrom(indiv)

        # Update proximity with existing subpop members
        for other in subpop:
            dist = self.brokenPairsDistance(myIndividual, other)
            
            #Check
            myIndividual.indivsPerProximity.add((dist, other))
            other.indivsPerProximity.add((dist, myIndividual))

        # Insert in ascending order of penalizedCost
        place = len(subpop)
        while place > 0 and subpop[place - 1].eval.penalized_cost > myIndividual.eval.penalized_cost - MY_EPSILON:
            place -= 1
        subpop.insert(place, myIndividual)

        # Survivor selection if needed
        maxSize = self.algo_params.population_size + self.algo_params.generation_size
        while len(subpop) > maxSize and len(subpop) > self.algo_params.population_size:
            self.removeWorstBiasedFitness(subpop)

        # Update best solutions if feasible
        if myIndividual.eval.is_feasible:
            if myIndividual.eval.penalized_cost < self.bestSolutionRestart.eval.penalized_cost - MY_EPSILON:
                self.bestSolutionRestart.copyFrom(myIndividual)
                if myIndividual.eval.penalized_cost < self.bestSolutionOverall.eval.penalized_cost - MY_EPSILON:
                    self.bestSolutionOverall.copyFrom(myIndividual)
                    self.searchProgress.append((time.time() - self.params.startTime,
                                                self.bestSolutionOverall.eval.penalized_cost))
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
            isClone = (self.averageBrokenPairsDistanceClosest(pop[i], 1) < MY_EPSILON)
            if (isClone and not worstIsClone) or (isClone == worstIsClone and pop[i].biasedFitness > worstBF):
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
            # Erase (dist, removed) from indiv2
            # We locate it by scanning indiv2->indivsPerProximity
            toRemove = None
            for (dval, ref) in indiv2.indivsPerProximity:
                if ref is removed:
                    toRemove = (dval, ref)
                    break
            if toRemove:
                indiv2.indivsPerProximity.remove(toRemove)

    def brokenPairsDistance(self, indiv1: Individual, indiv2: Individual) -> float:
        """Calculates the broken pairs distance between two individuals."""
        differences = 0
        for j in range(1, self.inst.num_customers + 1):
            if indiv1.successors[j] != indiv2.successors[j] and indiv1.successors[j] != indiv2.predecessors[j]:
                differences += 1
            if indiv1.predecessors[j] == 0 and indiv2.predecessors[j] != 0 and indiv2.successors[j] != 0:
                differences += 1
        return differences / self.inst.num_customers

    def averageBrokenPairsDistanceClosest(self, indiv: Individual, nbClosest: int) -> float:
        """Calculates the average broken pairs distance to the closest neighbors."""
        result = 0.0
        maxSize = min(nbClosest, len(indiv.indivsPerProximity))
        sorted_proximity = sorted(indiv.indivsPerProximity.items())  # Sort by distance
        for i in range(maxSize):
            result += sorted_proximity[i][0]  # Add the distance
        return result / maxSize if maxSize > 0 else 0.0

    def updateBiasedFitnesses(self, pop: List[Individual]):
        """Updates the biased fitness values for a population."""
        # Rank individuals based on diversity contribution (decreasing order of distance)
        ranking = []
        for i, indiv in enumerate(pop):
            diversity = -self.averageBrokenPairsDistanceClosest(indiv, self.algo_params.num_close)
            ranking.append((diversity, i))
        ranking.sort()  # Sort by diversity (descending)

        # Update biased fitness values
        if len(pop) == 1:
            pop[0].biasedFitness = 0.0
        else:
            for rank, (_, idx) in enumerate(ranking):
                divRank = rank / (len(pop) - 1)  # Diversity rank (0 to 1)
                fitRank = idx / (len(pop) - 1)  # Fitness rank (0 to 1)
                if len(pop) <= self.algo_params.num_elite:
                    pop[idx].biasedFitness = fitRank
                else:
                    pop[idx].biasedFitness = fitRank + (1.0 - self.algo_params.num_elite / len(pop)) * divRank

    def getBinaryTournament(self) -> Individual:
        """Selects two individuals and returns the one with better biased fitness."""
        # Combine feasible and infeasible subpopulations
        combined_pop = self.solver.feasible_population + self.solver.infeasible_population
        if len(combined_pop) < 2:
            raise ValueError("Population size must be at least 2 for binary tournament.")

        # Randomly select two individuals
        indiv1 = random.choice(combined_pop)
        indiv2 = random.choice(combined_pop)

        # Update biased fitnesses
        self.updateBiasedFitnesses(self.solver.feasible_population)
        self.updateBiasedFitnesses(self.solver.infeasible_population)

        # Return the individual with better (lower) biased fitness
        return indiv1 if indiv1.biasedFitness < indiv2.biasedFitness else indiv2
    
    def restart(self):
        """
        Clears and rebuilds subpopulations. Matches Population::restart in C++.
        """
        # if self.params.verbose:
        #     print("----- RESET: CREATING A NEW POPULATION -----")
        # for indiv in self.solver.feasible_population:
        #     # In C++, we delete the pointer. Here, there's no explicit memory free needed.
        #     pass
        # for indiv in self.solver.infeasible_population:
        #     pass

        self.solver.feasible_population.clear()
        self.solver.infeasible_population.clear()
        self.bestSolutionRestart = Individual(self.inst)
        # self.bestSolutionRestart.eval['penalizedCost'] = 1e30
        self.generatePopulation()
        
    def managePenalties(self):
        """
        Matches Population::managePenalties logic in the code snippet.
        """
        fractionFeasibleLoad = sum(self.listFeasibilityLoad) / float(len(self.listFeasibilityLoad))
        
        #TODO: Check penalty increase/decrease
        if fractionFeasibleLoad < self.algo_params.feasibility_target - 0.05 and self.solver.penalty_capacity < 100000.0:
            self.solver.penalty_capacity = min(self.solver.penalty_capacity * self.params.ap['penaltyIncrease'], 100000.0)
        elif fractionFeasibleLoad > self.algo_params.feasibility_target + 0.05 and self.solver.penalty_capacity > 0.1:
            self.solver.penalty_capacity = max(self.params.penaltyCapacity * self.params.ap['penaltyDecrease'], 0.1)

        # fractionFeasibleDuration = sum(self.listFeasibilityDuration) / float(len(self.listFeasibilityDuration))
        # if fractionFeasibleDuration < self.params.ap['targetFeasible'] - 0.05 and self.params.penaltyDuration < 100000.0:
        #     self.params.penaltyDuration = min(self.params.penaltyDuration * self.params.ap['penaltyIncrease'], 100000.0)
        # elif fractionFeasibleDuration > self.params.ap['targetFeasible'] + 0.05 and self.params.penaltyDuration > 0.1:
        #     self.params.penaltyDuration = max(self.params.penaltyDuration * self.params.ap['penaltyDecrease'], 0.1)

        # Update penalized cost for infeasible
        for indiv in self.solver.infeasible_population:
            indiv.eval.penalized_cost = (
                indiv.eval.distance
                + self.solver.penalty_capacity * indiv.eval.capacity_excess
            )

        # Reorder via bubble sort for demonstration
        n = len(self.solver.infeasible_population)
        for i in range(n):
            for j in range(n - i - 1):
                if self.solver.infeasible_population[j].eval.penalized_cost > self.solver.infeasible_population[j+1].eval.penalized_cost + MY_EPSILON:
                    self.solver.infeasible_population[j], self.solver.infeasible_population[j+1] = self.solver.infeasible_population[j+1], self.solver.infeasible_population[j]