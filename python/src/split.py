'''
Splitting for handling the case of cvrp with constraints on 
vehicles and their capacities. Because of this, the maximum
number of split bins cannot exceed the number of vehicles.

However, the number of bins can be less than the number of vehicles
and be more optimal than using all vehicles.

To manage splitting, we use a unique deque structure with 
a front and back index to manage the split bins.
'''

import math

################################################################################
# Support Classes (Analogous to C++ "Params", "Individual", "ClientSplit", etc.)
################################################################################

class ClientSplit:
    """
    Mirrors 'ClientSplit' in C++ (Split.h):
      - demand
      - serviceTime
      - d0_x: cost from depot to client
      - dx_0: cost from client to depot
      - dnext: cost from this client to next giant-tour client
    """
    def __init__(self):
        self.demand = 0.0
        self.serviceTime = 0.0
        self.d0_x = 0.0
        self.dx_0 = 0.0
        self.dnext = 0.0

class TrivialDeque:
    """
    Mirrors the C++ 'Trivial_Deque' structure (Split.h).
    Used to maintain 'best predecessors' in the O(n) approach.
    """
    def __init__(self, num_elements, firstNode):
        # A simple Python list to store indices
        # indexFront and indexBack mark the front/back in C++
        self.myDeque = [firstNode] + [0]*(num_elements - 1)
        self.indexFront = 0
        self.indexBack = 0

    def pop_front(self):
        self.indexFront += 1

    def pop_back(self):
        self.indexBack -= 1

    def push_back(self, i):
        self.indexBack += 1
        self.myDeque[self.indexBack] = i

    def get_front(self):
        return self.myDeque[self.indexFront]

    def get_next_front(self):
        return self.myDeque[self.indexFront + 1] if (self.indexBack - self.indexFront + 1) > 1 else 0

    def get_back(self):
        return self.myDeque[self.indexBack]

    def reset(self, firstNode):
        self.myDeque[0] = firstNode
        self.indexBack = 0
        self.indexFront = 0

    def size(self):
        return self.indexBack - self.indexFront + 1


################################################################################
# Split Class (Mirrors Split.h and Split.cpp, focusing on limited fleet, no durations)
################################################################################

class Split:
    """
    Mirrors the C++ class 'Split' in Split.h/Split.cpp.
    Key fields are:
      - params: problem parameters
      - maxVehicles: computed # of vehicles
      - cliSplit[]: array of ClientSplit
      - sumLoad[], sumService[], sumDistance[]: prefix sums
      - potential[k][i], pred[k][i]: DP cost and predecessor index
    """

    def __init__(self, params):
        self.params = params
        self.maxVehicles = 0
        # +1 so we can index from 1..nbClients
        self.cliSplit = [ClientSplit() for _ in range(params.num_clients + 1)]

        self.sumDistance = [0.0]*(params.num_clients + 1)
        self.sumLoad = [0.0]*(params.num_clients + 1)
        self.sumService = [0.0]*(params.num_clients + 1)

        # potential[k][i], pred[k][i]
        self.potential = [
            [1.e30]*(params.num_clients + 1) for _ in range(params.num_vehicles + 1)
        ]
        self.pred = [
            [0]*(params.num_clients + 1) for _ in range(params.num_vehicles + 1)
        ]

    def split(self, indiv, num_max_vehicles):
        """
        Mirrors Split::generalSplit in the C++ code.
        - Sets up data structures
        - Calls splitSimple first, then falls back to splitLF if needed
        - However, we want limited fleet, no durations, so either skip
          or keep the logic consistent with C++ approach.
        """
        # As in the C++ method:
        trivial_bound = math.ceil(self.params.totalDemand / self.params.vehicleCapacity)
        self.maxVehicles = max(num_max_vehicles, trivial_bound)

        # Initialize cliSplit and prefix sums
        for i in range(1, self.params.num_clients + 1):
            cID = indiv.chromT[i - 1]
            self.cliSplit[i].demand = self.params.cli[cID].demand
            

            if i < self.params.num_clients:
                nxt = indiv.chromT[i]
            else:
                # (Mirroring the C++ sentinel -1.e30)
                self.cliSplit[i].dnext = -1.e30

            # Build prefix sums
            self.sumLoad[i] = self.sumLoad[i - 1] + self.cliSplit[i].demand
            self.sumDistance[i] = self.sumDistance[i - 1] + (0.0 if i == 1 else self.cliSplit[i - 1].dnext)

        # We ignore durations in both. This matches the default code when isDurationConstraint = False.
        
        self.split_lf(indiv)

        # Evaluate solution
        indiv.evaluateCompleteCost(self.params)

    def split_lf(self, indiv):
        """
        Mirrors Split::splitLF in the C++ code (no duration constraints),
        employing the O(n) approach from Vidal (2016) for each route index k.
        """
        # Reinitialize all potential[] values
        self.potential[0][0] = 0.0
        for k in range(self.maxVehicles + 1):
            for i in range(1, self.params.num_clients + 1):
                self.potential[k][i] = 1.e30

       
        # O(n) approach
        for k in range(self.maxVehicles):
            queue = TrivialDeque(self.params.num_clients + 1, k)
            queue.reset(k)  # Start from "client index" = k

            for i in range(k + 1, self.params.num_clients + 1):
                if queue.size() == 0:
                    break
                front_idx = queue.get_front()
                self.potential[k + 1][i] = self._propagate(front_idx, i, k)
                self.pred[k + 1][i] = front_idx

                if i < self.params.num_clients:
                    if not self._dominates(queue.get_back(), i, k):
                        while queue.size() > 0 and self._dominates_right(queue.get_back(), i, k):
                            queue.pop_back()
                        queue.push_back(i)

                    # Possibly remove the front if it's worse for i+1
                    while queue.size() > 1 and \
                            self._propagate(queue.get_front(), i+1, k) > \
                            self._propagate(queue.get_next_front(), i+1, k) - 1e-6:
                        queue.pop_front()

        if self.potential[self.maxVehicles][self.params.num_clients] > 1.e29:
            # "ERROR : no Split solution has been propagated..."
            # In the C++ code, it throws a string. We'll just treat as unsuccessful.
            return 0

        # Could be cheaper with fewer routes
        minCost = self.potential[self.maxVehicles][self.params.num_clients]
        nbRoutes = self.maxVehicles
        for kv in range(1, self.maxVehicles):
            if self.potential[kv][self.params.num_clients] < minCost:
                minCost = self.potential[kv][self.params.num_clients]
                nbRoutes = kv

        # Rebuild solution
        for k in range(self.params.num_vehicles - 1, nbRoutes - 1, -1):
            indiv.chromR[k].clear()

        end = self.params.num_clients
        for route_idx in range(nbRoutes - 1, -1, -1):
            indiv.chromR[route_idx].clear()
            begin = self.pred[route_idx + 1][end]
            for ii in range(begin, end):
                indiv.chromR[route_idx].append(indiv.chromT[ii])
            end = begin

        return 1 if (end == 0) else 0

    ########################################################################
    # Inline-Like Methods (Adapted from Split.h)
    ########################################################################

    def _propagate(self, i, j, k):
        """
        Mirrors the inline 'propagate(int i, int j, int k)' from Split.h.
        i < j in normal usage. This calculates:
         potential[k][i]
         + (sumDistance[j] - sumDistance[i+1])
         + cliSplit[i+1].d0_x + cliSplit[j].dx_0
         + capacity penalty if sumLoad[j] - sumLoad[i] > params.vehicleCapacity
        """
        base_cost = self.potential[k][i]
        travel_distance = self.sumDistance[j] - self.sumDistance[i + 1]
        cost_inbound = self.cliSplit[i + 1].d0_x  # cost from depot to i+1
        cost_outbound = self.cliSplit[j].dx_0     # cost from j back to depot
        load_diff = self.sumLoad[j] - self.sumLoad[i]
        cap_excess = max(load_diff - self.params.vehicle_capacity, 0.0)

        return base_cost + travel_distance + cost_inbound + cost_outbound \
               + self.params.penalty_capacity * cap_excess

    def _dominates(self, i, j, k):
        """
        Mirrors 'dominates(int i, int j, int k)' from Split.h:
         potential[k][j] + cliSplit[j+1].d0_x >
         potential[k][i] + cliSplit[i+1].d0_x
                         + (sumDistance[j+1] - sumDistance[i+1])
                         + penaltyCapacity * (sumLoad[j] - sumLoad[i])
        (We assume i < j.)
        """
        lhs = self.potential[k][j] + self.cliSplit[j + 1].d0_x
        rhs = (self.potential[k][i] + self.cliSplit[i + 1].d0_x
               + (self.sumDistance[j + 1] - self.sumDistance[i + 1])
               + self.params.penalty_capacity * (self.sumLoad[j] - self.sumLoad[i]))
        return lhs > rhs

    def _dominates_right(self, i, j, k):
        """
        Mirrors 'dominatesRight(int i, int j, int k)' from Split.h:
         potential[k][j] + cliSplit[j+1].d0_x <
         potential[k][i] + cliSplit[i+1].d0_x
                         + sumDistance[j+1] - sumDistance[i+1]
                         + MY_EPSILON
        For numerical reasons, we can use 1e-6 as MY_EPSILON.
        (Again, assume i < j.)
        """
        lhs = self.potential[k][j] + self.cliSplit[j + 1].d0_x
        rhs = (self.potential[k][i] + self.cliSplit[i + 1].d0_x
               + (self.sumDistance[j + 1] - self.sumDistance[i + 1]) + 1e-6)
        return lhs < rhs