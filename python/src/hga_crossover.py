import random
from split import Split


class HGACrossover:
    #### filepath: genetic.py
    #### A Python implementation of the crossoverOX method, mirroring Genetic::crossoverOX.
    def __init__(self, solver):
        self.solver = solver
        self.inst = solver.inst
        self.splitter = self.solver.splitter

    def crossover_ox(self, result, parent1, parent2):
        """
        Perform an OX (Ordered Crossover) on two parent Individuals:
            1) Select a crossover slice [start..end] from parent1 and copy it directly.
            2) Fill remaining positions with the order from parent2.
            3) Finally, use the 'split' method to convert the giant tour into feasible routes.
        """

        # Frequency table for the customers inserted into 'result'
        freq_customer = [False] * self.inst.num_customers

        # Pick the beginning and end of the crossover zone
        start = random.randint(0, self.inst.num_customers - 2)
        end = random.randint(0, self.inst.num_customers - 2)
        while end == start:
            end = random.randint(0, self.inst.num_customers - 1)

        # Copy the crossover slice from parent1 into result
        j = start
        while j % (self.inst.num_customers - 1) != (end + 1) % (
            self.inst.num_customers - 1
        ):
            result.chromT[j % (self.inst.num_customers - 1)] = parent1.chromT[
                j % (self.inst.num_customers - 1)
            ]
            freq_customer[result.chromT[j % (self.inst.num_customers - 1)]] = True
            j += 1

        # Fill remaining slots from parent2
        for i in range(1, self.inst.num_customers):
            c = parent2.chromT[(end + i) % (self.inst.num_customers - 1)]
            if not freq_customer[c]:
                result.chromT[j % (self.inst.num_customers - 1)] = c
                j += 1

        # Convert the giant tour in 'result.chromT' into feasible routes
        self.splitter.run(result)
