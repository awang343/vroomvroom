from hga_circle import CircleSector

from dataclasses import dataclass
from typing import Optional
from sortedcontainers import SortedList
import random


@dataclass
class AlgoParams:
    population_size: int = 1
    generation_size: int = 5
    neighborhood_size: int = 20
    num_elite: int = 4
    num_close: int = 5
    education_prob: float = 1.0
    repair_prob: float = 0.5
    feasibility_target: float = 0.2

    num_iter_until_penalty: int = 100
    penalty_decrease: float = 0.85
    penalty_increase: float = 1.2
    num_iter: int = 20000
    num_iter_traces: int = 500


@dataclass
class Customer:
    x: float
    y: float
    demand: int
    polar: int


@dataclass
class Node:
    idx: int = -1  # Node index

    is_depot: bool = False  # Whether the node is a depot
    position: int = -1  # Position in the route
    last_tested_RI: int = -1  # Last time RI move was tested
    next: Optional["Node"] = None  # Next node in the route
    prev: Optional["Node"] = None  # Previous node in the route
    route: Optional["Node"] = None  # Reference to the associated route

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
    sector = CircleSector()  # CircleSector object


@dataclass
class Evaluation:
    penalized_cost: float = 1.0e30
    num_routes: int = 0
    distance: float = 0
    capacity_excess: int = 0  # Sum of excess capacities
    is_feasible: bool = False


@dataclass
class SwapStarElement:
    move_cost: float = 1e30
    U: Optional["Node"] = None
    best_position_U: Optional["Node"] = None
    V: Optional["Node"] = None
    best_position_V: Optional["Node"] = None


class Individual:
    def __init__(self, solver):
        self.solver = solver
        self.inst = solver.inst
        self.eval = Evaluation()

        self.successors = [0] * (self.inst.num_customers)
        self.predecessors = [0] * (self.inst.num_customers)
        self.proximity_indivs = SortedList()

        # chromR represents the solution as a list of routes
        self.chromR = [[] for _ in range(self.inst.num_vehicles)]

        # chromT represents the uncut list of customers before route assignment
        self.chromT = [i + 1 for i in range(self.inst.num_customers - 1)]

        # Shuffle chromT
        random.shuffle(self.chromT)

    def __lt__(self, other):
        return True

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
            load = self.inst.customers[route[0]].demand

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
            self.eval.distance
            + self.eval.capacity_excess * self.solver.capacity_penalty
        )
        self.eval.is_feasible = self.eval.capacity_excess < 1e-3


class ThreeBestInsert:
    def __init__(self):
        self.last_calculated = None
        self.best = [(float("inf"), None)] * 3

    def compare_and_add(self, cost_insert, place_insert):
        heap_item = (cost_insert, place_insert)

        if len(self.best) < 3:
            heapq.heappush(self.heap, heap_item)
        else:
            # If this cost is better than the current worst
            if -self.best[0][0] > cost_insert:
                heapq.heappushpop(self.heap, heap_item)

    def reset(self):
        self.best = [(float("inf"), None)] * 3
