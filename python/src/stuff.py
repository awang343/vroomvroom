import random
from typing import List, Dict

class Population:
    def __init__(self, params):
        self.params = params
        self.feasibleSubpop = []  # List of feasible individuals
        self.infeasibleSubpop = []  # List of infeasible individuals

    def brokenPairsDistance(self, indiv1: Individual, indiv2: Individual) -> float:
        """Calculates the broken pairs distance between two individuals."""
        differences = 0
        for j in range(1, self.params['nbClients'] + 1):
            if indiv1.successors[j] != indiv2.successors[j] and indiv1.successors[j] != indiv2.predecessors[j]:
                differences += 1
            if indiv1.predecessors[j] == 0 and indiv2.predecessors[j] != 0 and indiv2.successors[j] != 0:
                differences += 1
        return differences / self.params['nbClients']

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
            diversity = -self.averageBrokenPairsDistanceClosest(indiv, self.params['nbClose'])
            ranking.append((diversity, i))
        ranking.sort()  # Sort by diversity (descending)

        # Update biased fitness values
        if len(pop) == 1:
            pop[0].biasedFitness = 0.0
        else:
            for rank, (_, idx) in enumerate(ranking):
                divRank = rank / (len(pop) - 1)  # Diversity rank (0 to 1)
                fitRank = idx / (len(pop) - 1)  # Fitness rank (0 to 1)
                if len(pop) <= self.params['nbElite']:
                    pop[idx].biasedFitness = fitRank
                else:
                    pop[idx].biasedFitness = fitRank + (1.0 - self.params['nbElite'] / len(pop)) * divRank

    def getBinaryTournament(self) -> Individual:
        """Selects two individuals and returns the one with better biased fitness."""
        # Combine feasible and infeasible subpopulations
        combined_pop = self.feasibleSubpop + self.infeasibleSubpop
        if len(combined_pop) < 2:
            raise ValueError("Population size must be at least 2 for binary tournament.")

        # Randomly select two individuals
        indiv1 = random.choice(combined_pop)
        indiv2 = random.choice(combined_pop)

        # Update biased fitnesses
        self.updateBiasedFitnesses(self.feasibleSubpop)
        self.updateBiasedFitnesses(self.infeasibleSubpop)

        # Return the individual with better (lower) biased fitness
        return indiv1 if indiv1.biasedFitness < indiv2.biasedFitness else indiv2