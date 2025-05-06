import math
from dataclasses import dataclass


# {{{ Data Structures
class CustomerSplit:
    demand: float = 0.0
    ddepot: float = 0.0  # Distance from this customer to depot
    dnext: float = 0.0  # Distance from this customer to next customer


class CustomDeque:
    def __init__(self, capacity: int, first_node: int):
        self.queue = [0] * capacity
        self.index_front = 0
        self.index_back = 0
        self.queue[0] = first_node

    def pop_front(self):
        self.index_front += 1

    def pop_back(self):
        self.index_back -= 1

    def push_back(self, i):
        self.index_back += 1
        self.queue[self.index_back] = i

    def get_front(self):
        return self.queue[self.index_front]

    def get_next_front(self):
        return self.queue[self.index_front + 1]

    def get_back(self):
        return self.queue[self.index_back]

    def reset(self, first_node):
        self.index_front = 0
        self.index_back = 0
        self.myDeque[0] = first_node

    def size(self):
        return self.index_back - self.index_front + 1


# }}}


class Split:
    def __init__(self, solver):
        self.solver = solver

        self.inst = solver.inst
        self.params = solver.params

        self.splits = [CustomerSplit() for _ in range(self.inst.num_customers)]

        # Potential includes an index for no customers visited and no vehicles used
        self.potential = [
            [float("inf")] * instance.num_customers
            for _ in range(self.inst.num_vehicles + 1)
        ]
        self.pred = [
            [0] * self.inst.num_customers for _ in range(self.inst.num_vehicles + 1)
        ]
        self.sumDistance = [0.0] * self.inst.num_customers
        self.sumLoad = [0.0] * self.inst.num_customers
        self.sumService = [0.0] * instance.num_customers

    # {{{ Helpers
    def propagate(self, i, j, k):
        return (
            self.potential[k][i]
            + self.sumDistance[j]
            - self.sumDistance[i + 1]
            + self.cliSplit[i + 1].d0_x
            + self.cliSplit[j].dx_0
            + self.solver.capacity_penalty
            * max(
                self.sumLoad[j] - self.sumLoad[i] - self.params["vehicleCapacity"], 0.0
            )
        )

    def dominates(self, i, j, k):
        return self.potential[k][j] + self.cliSplit[j + 1].d0_x > self.potential[k][
            i
        ] + self.cliSplit[i + 1].d0_x + self.sumDistance[j + 1] - self.sumDistance[
            i + 1
        ] + self.params[
            "penaltyCapacity"
        ] * (
            self.sumLoad[j] - self.sumLoad[i]
        )

    def dominates_right(self, i, j, k):
        return (
            self.potential[k][j] + self.cliSplit[j + 1].d0_x
            < self.potential[k][i]
            + self.cliSplit[i + 1].d0_x
            + self.sumDistance[j + 1]
            - self.sumDistance[i + 1]
            + 1e-6
        )

    # }}}

    # {{{ split_simple
    def split_simple(self, indiv):
        self.potential[0][0] = 0  # Minimum cost to hit 0 customers with 0 vehicles
        for i in range(1, self.inst.num_customers):
            self.potential[0][i] = float(
                "inf"
            )  # Minimum cost to hit first i customers with 0 vehicles

        for i in range(self.inst.num_customers - 1):
            load = 0.0
            distance = 0.0
            for j in range(i + 1, self.inst.num_customers):
                load += self.splits[j].demand
                distance += (
                    self.splits[j - 1].dnext if j > i + 1 else self.splits[j].ddepot
                )
                cost = (
                    distance
                    + self.splits[j].dx_0
                    + self.solver.capacity_penalty
                    * max(0.0, load - self.inst.vehicle_capacity)
                )

                if self.potential[0][i] + cost < self.potential[0][j]:
                    self.potential[0][j] = self.potential[0][i] + cost
                    self.pred[0][j] = i

        end = self.inst.num_customers - 1  # Index of the last customer
        for k in range(self.inst.num_vehicles - 1, -1, -1):
            indiv.chromR[k] = []

        for k in range(self.inst.num_vehicles - 1, -1, -1):
            begin = self.pred[0][end]
            for ii in range(begin, end):
                indiv.chromR[k].append(indiv.chromT[ii])
            end = begin

        return end == 0

    # }}}

    # {{{ split_lf
    def split_lf(self, indiv):
        self.potential[0][0] = 0
        for k in range(self.inst.num_vehicles + 1):
            for i in range(1, self.inst.num_customers + 1):
                self.potential[k][i] = float("inf")

        for k in range(self.inst.num_vehicles):
            for i in range(k, self.inst.num_customers - 1):
                load = 0.0
                distance = 0.0
                for j in range(i + 1, self.inst.num_customers):
                    load += self.splits[j].demand
                    distance += (
                        self.splits[j - 1].dnext if j > i + 1 else self.splits[j].d0_x
                    )
                    cost = (
                        distance
                        + self.splits[j].dx_0
                        + self.solver.capacity_penalty
                        * max(0.0, load - self.inst.vehicle_capacity)
                    )

                    if self.potential[k][i] + cost < self.potential[k + 1][j]:
                        self.potential[k + 1][j] = self.potential[k][i] + cost
                        self.pred[k + 1][j] = i

        # Find the num_routes that gives you the minimum cost
        min_cost = self.potential[self.inst.num_vehicles][self.inst.num_customers - 1]
        num_routes = self.inst.num_vehicles

        for k in range(1, self.inst.num_vehicles):
            if self.potential[k][self.inst.num_customers - 1] < min_cost:
                min_cost = self.potential[k][self.inst.num_customers - 1]
                num_routes = k

        # Fill in chromR
        end = self.inst.num_customers - 1
        for k in range(num_routes - 1, -1, -1):
            indiv.chromR[k] = []

        for k in range(num_routes - 1, -1, -1):
            begin = self.pred[k + 1][end]
            for ii in range(begin, end):
                indiv.chromR[k].append(indiv.chromT[ii])
            end = begin

        return end == 0

    # }}}

    def run(self, indiv):
        # Load in all the information from chromT
        for i in range(1, self.inst.num_customers):
            customer = indiv.chromT[i - 1]
            next_customer = indiv.chromT[i]

            self.splits[i].demand = self.inst.customers[customer].demand
            self.splits[i].depot_dist = self.inst.distances[0][customer]
            self.splits[i].dnext = (
                self.inst.distances[customer][next_customer]
                if i < self.inst.num_customers - 1
                else -1e30
            )  # Set distance to next customer to negative infinity at the end

            self.sum_load[i] = self.sum_load[i - 1] + self.inst.customers[i].demand
            self.sum_distance[i] = (
                self.sum_distance[i - 1] + self.inst.customers[i - 1].dnext
            )

        # Perform splitting
        if not self.split_simple(indiv):
            self.split_lf(indiv)

        indiv.evaluate_complete_cost()
