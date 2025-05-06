from dataclasses import dataclass


@dataclass
class Client:
    x: float
    y: float
    demand: int
    polarangle: int


@dataclass
class Node:
    idx: int = -1  # Node index

    isDepot: bool = False  # Whether the node is a depot
    position: int = -1  # Position in the route
    whenLastTestedRI: int = -1  # Last time RI move was tested
    next: Node = None  # Next node in the route
    prev: Node = None  # Previous node in the route
    route: Node = None  # Reference to the associated route

    cumulatedLoad: float = 0.0  # Cumulative load until this node
    cumulatedDist: float = 0.0  # Cumulative dist until this node
    cumulatedReversalDistance: float = (
        0.0  # Cost change if route reversed up to this node
    )
    deltaRemoval: float = 0.0  # Cost delta of removing the node


@dataclass
class Route:
    idx: int = -1  # Route index

    nbCustomers: int = 0  # Number of customers in the route
    whenLastModified: int = -1  # Last modification time
    whenLastTestedSWAPStar: int = -1  # Last time SWAP* was tested

    depot: Node = None  # Pointer to depot Node
    distance: float = 0.0  # Total route distance
    load: float = 0.0  # Total route load
    reversalDistance: float = 0.0  # Cost change if route reversed
    penalty: float = 0.0  # Load + duration penalties
    polarAngleBarycenter: float = 0.0  # Polar angle of barycenter
    sector = None  # Placeholder for CircleSector object

class Evaluation:
    penalized_cost: float = 1.0e30
    nb_routes: int = 0
    distance: float = 0
    capacity_excess: int = 0 # Sum of excess capacities
    isFeasible: bool = False

class Individual:
    def __init__(self, instance):
        self.inst = instance
        self.eval = Evaluation()

        self.successors = [0] * (instance.numCustomers)
        self.predecessors = [0] * (instance.numCustomers)

        # chromR represents the solution as a list of routes
        self.chromR = [[] for _ in range(instance.numVehicles)]

        # chromT represents the uncut list of customers before route assignment
        self.chromT = [i + 1 for i in range(instance.numCustomers - 1)]

        # Shuffle chromT
        random.shuffle(self.chromT, random=params.ran)

        # Initialize eval attributes

    def evaluateCompleteCost(self, params):
        self.eval = EvalIndiv()

        for r in range(params.nbVehicles):
            if len(self.chromR[r]) <= 0:
                continue

            # Init distance is from depot to first customer
            distance = self.inst.distances[0][self.chromR[r][0]]

            # Initial load is demand of first customer
            load = self.inst.clients[self.chromR[r][0]].demand

            self.predecessors[self.chromR[r][0]] = 0

            for i in range(1, len(self.chromR[r])):
                distance += params.timeCost[self.chromR[r][i - 1]][self.chromR[r][i]]
                load += params.cli[self.chromR[r][i]].demand
                self.predecessors[self.chromR[r][i]] = self.chromR[r][i - 1]
                self.successors[self.chromR[r][i - 1]] = self.chromR[r][i]

            self.successors[self.chromR[r][len(self.chromR[r]) - 1]] = 0
            distance += params.timeCost[self.chromR[r][len(self.chromR[r]) - 1]][0]
            self.eval.distance += distance
            self.eval.nbRoutes += 1

            if load > params.vehicleCapacity:
                self.eval.capacityExcess += load - params.vehicleCapacity
            if distance + service > params.durationLimit:
                self.eval.durationExcess += distance + service - params.durationLimit

        self.eval.penalizedCost = (
            self.eval.distance
            + self.eval.capacityExcess * params.penaltyCapacity
            + self.eval.durationExcess * params.penaltyDuration
        )
        self.eval.isFeasible = (
            self.eval.capacityExcess < MY_EPSILON
            and self.eval.durationExcess < MY_EPSILON
        )
