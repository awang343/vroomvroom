import random
from split import Split

class HGACrossover:
    #### filepath: genetic.py
#### A Python implementation of the crossoverOX method, mirroring Genetic::crossoverOX.
    def __init__(self, instance):
        self.inst = instance

    def crossover_ox(self, result, parent1, parent2):
        """
        Perform an OX (Ordered Crossover) on two parent Individuals:
            1) Select a crossover slice [start..end] from parent1 and copy it directly.
            2) Fill remaining positions with the order from parent2.
            3) Finally, use the 'split' method to convert the giant tour into feasible routes.

        Assumptions:
            - 'result', 'parent1', 'parent2' are Individual objects with a chromT list.
            - 'params' contains nbClients and random state info.
            - 'split' has a generalSplit method for route splitting.
        """

        # Frequency table for the clients inserted into 'result'
        freq_client = [False] * (self.inst.num_clients + 1)

        # Pick the beginning and end of the crossover zone
        start = random.randint(0, self.inst.num_clients - 1)
        end = random.randint(0, self.inst.num_clients - 1)
        while end == start:
            end = random.randint(0, self.inst.num_clients - 1)

        # Copy the crossover slice from parent1 into result
        j = start
        while j % self.inst.num_clients != (end + 1) % self.inst.num_clients:
            c = parent1.chromT[j % self.inst.num_clients]
            result.chromT[j % self.inst.num_clients] = c
            freq_client[c] = True
            j += 1

        # Fill remaining slots from parent2
        idx_offset = (end + 1) % self.inst.num_clients
        for _ in range(self.inst.num_clients):
            c = parent2.chromT[idx_offset]
            if not freq_client[c]:
                result.chromT[j % self.inst.num_clients] = c
                j += 1
            idx_offset = (idx_offset + 1) % self.inst.num_clients

        # Convert the giant tour in 'result.chromT' into feasible routes
        # Assuming parent1.eval has nbRoutes or a similar attribute
        split = Split(self.inst)
        split.run(result, self.inst.num_vehicles)