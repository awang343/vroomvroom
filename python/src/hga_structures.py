from dataclasses import dataclass


@dataclass
class AlgoParams:
    population_size: int = 25
    generation_size: int = 30
    neighborhood_size: int = 20
    num_elite: int = 4
    num_close: int = 5
    education_prob: float = 1.0
    repair_prob: float = 0.5
    feasibility_target: float = 0.2


@dataclass
class Customer:
    x: float
    y: float
    demand: int
    polarangle: int


@dataclass
class Node:
    idx: int = -1  # Node index

    is_depot: bool = False  # Whether the node is a depot
    position: int = -1  # Position in the route
    last_tested_RI: int = -1  # Last time RI move was tested
    next: Node = None  # Next node in the route
    prev: Node = None  # Previous node in the route
    route: Node = None  # Reference to the associated route

    cum_load: float = 0.0  # Cumulative load until this node
    cum_dist: float = 0.0  # Cumulative dist until this node
    cum_reversal_distance: float = 0.0  # Cost change if route reversed up to this node
    delta_removal: float = 0.0  # Cost delta of removing the node


@dataclass
class Route:
    idx: int = -1  # Route index

    num_customers: int = 0  # Number of customers in the route
    last_modified: int = -1  # Last modification time
    last_tested_SWAP: int = -1  # Last time SWAP* was tested

    depot: Node = None  # Pointer to depot Node
    distance: float = 0.0  # Total route distance
    load: float = 0.0  # Total route load
    reversal_distance: float = 0.0  # Cost change if route reversed
    penalty: float = 0.0  # Load + duration penalties
    polar_angle_barycenter: float = 0.0  # Polar angle of barycenter
    sector = None  # Placeholder for CircleSector object


@dataclass
class Evaluation:
    penalized_cost: float = 1.0e30
    num_routes: int = 0
    distance: float = 0
    capacity_excess: int = 0  # Sum of excess capacities
    is_feasible: bool = False


class Individual:
    def __init__(self, instance):
        self.inst = instance
        self.eval = Evaluation()

        self.successors = [0] * (instance.num_customers)
        self.predecessors = [0] * (instance.num_customers)

        # chromR represents the solution as a list of routes
        self.chromR = [[] for _ in range(instance.num_vehicles)]

        # chromT represents the uncut list of customers before route assignment
        self.chromT = [i + 1 for i in range(instance.num_customers - 1)]

        # Shuffle chromT
        random.shuffle(self.chromT, random=params.ran)

    def evaluateCompleteCost(self):
        self.eval = Evaluation()

        for r in range(self.inst.num_vehicles):
            route = self.chromR[r]

            if len(route) == 0:
                continue

            # Initialize distance and load
            distance = (
                self.inst.distances[0][route[0]] + self.inst.distances[route[-1]][0]
            )
            load = self.inst.clients[route[0]].demand

            self.predecessors[route[0]] = 0
            self.successors[route[-1]] = 0

            for i in range(1, len(route)):
                distance += self.inst.distances[route[i - 1]][route[i]]
                load += self.inst.customers[route[i]].demand

                self.predecessors[route[i]] = route[i - 1]
                self.successors[route[i - 1]] = route[i]

            self.eval.distance += distance
            self.eval.num_routes += 1  # Only count nonempty routes
            self.eval.capacity_excess += max(0, load - self.inst.vehicle_capacity)

        self.eval.penalizedCost = (
            self.eval.distance + self.eval.capacity_excess * self.inst.penalty_capacity
        )
        self.eval.is_feasible = self.eval.capacity_excess < 1e-3


class ThreeBestInsert:
    def __init__(self):
        self.last_calculated = None
        self.best_cost = [float("inf")] * 3
        self.best_location = [None] * 3

    def compare_and_add(self, cost_insert, place_insert):
        heap_item = (cost_insert, place_insert)

        if len(self.heap) < 3:
            heapq.heappush(self.heap, heap_item)
        else:
            # If this cost is better than the current worst
            if -self.heap[0][0] > cost_insert:
                heapq.heappushpop(self.heap, heap_item)

    def reset(self):
        self.best = [(float("inf"), None)] * 3
