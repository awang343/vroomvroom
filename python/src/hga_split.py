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
            [float("inf")] * self.inst.num_customers
            for _ in range(self.inst.num_vehicles + 1)
        ]
        self.pred = [
            [0] * self.inst.num_customers for _ in range(self.inst.num_vehicles + 1)
        ]
        self.sum_distance = [0.0] * self.inst.num_customers
        self.sum_load = [0.0] * self.inst.num_customers

    # {{{ Helpers
    def propagate(self, i, j, k):
        return (
            self.potential[k][i]
            + self.sum_distance[j]
            - self.sum_distance[i + 1]
            + self.splits[i + 1].ddepot
            + self.splits[j].ddepot
            + self.solver.capacity_penalty
            * max(self.sum_load[j] - self.sum_load[i] - self.inst.vehicle_capacity, 0.0)
        )

    def dominates(self, i, j, k):
        return self.potential[k][j] + self.splits[j + 1].ddepot > self.potential[k][
            i
        ] + self.splits[i + 1].ddepot + self.sum_distance[j + 1] - self.sum_distance[
            i + 1
        ] + self.solver.capacity_penalty * (
            self.sum_load[j] - self.sum_load[i]
        )

    def dominates_right(self, i, j, k):
        return (
            self.potential[k][j] + self.splits[j + 1].ddepot
            < self.potential[k][i]
            + self.splits[i + 1].ddepot
            + self.sum_distance[j + 1]
            - self.sum_distance[i + 1]
            + 1e-6
        )

    # }}}

    # {{{ split_simple
    def split_simple(self, indiv):
        self.potential[0][0] = 0  # Minimum cost to hit 0 customers with 0 vehicles
        for i in range(1, self.inst.num_customers):
            # Minimum cost to hit first i customers with 0 vehicles
            self.potential[0][i] = float("inf")

        queue = CustomDeque(self.inst.num_customers, 0)

        for i in range(1, self.inst.num_customers):
            # The front is the best predecessor for i
            self.potential[0][i] = self.propagate(queue.get_front(), i, 0)
            self.pred[0][i] = queue.get_front()

            if i < self.inst.num_customers - 1:
                # If i is not dominated by the last of the queue
                if not self.dominates(queue.get_back(), i, 0):
                    # Remove from back any elements dominated by i
                    while queue.size() > 0 and self.dominates_right(
                        queue.get_back(), i, 0
                    ):
                        queue.pop_back()
                    queue.push_back(i)

                # Check if front is still the best for i+1
                while (
                    queue.size() > 0
                    and self.propagate(queue.get_front(), i + 1, 0)
                    > self.propagate(queue.get_next_front(), i + 1, 0) - 1e-3
                ):
                    queue.pop_front()

        end = self.inst.num_customers - 1  # Index of the last customer

        for k in range(self.inst.num_vehicles - 1, -1, -1):
            begin = self.pred[0][end]
            indiv.chromR[k] = indiv.chromT[begin:end]
            end = begin

        return end == 0

    # }}}

    # {{{ split_lf
    def split_lf(self, indiv):
        self.potential[0][0] = 0
        for k in range(self.inst.num_vehicles + 1):
            for i in range(1, self.inst.num_customers):
                self.potential[k][i] = float("inf")

        queue = CustomDeque(self.inst.num_customers, k)
        for k in range(self.inst.num_vehicles):
            queue.reset(k)

            for i in range(k + 1, self.inst.num_customers):
                if queue.size() == 0:
                    break

                # The front is the best predecessor for i
                self.potential[k + 1][i] = self.propagate(queue.get_front(), i, k)
                self.pred[k + 1][i] = queue.get_front()

                if i < self.inst.num_customers - 1:
                    # If i is not dominated by the last of the queue
                    if not self.dominates(queue.get_back(), i, k):
                        # Remove from back any elements dominated by i
                        while queue.size() > 0 and self.dominates_right(
                            queue.get_back(), i, k
                        ):
                            queue.pop_back()
                        queue.push_back(i)

                    # Remove front elements while the next is better for i+1
                    while (
                        queue.size() > 1
                        and self.propagate(queue.get_front(), i + 1, k)
                        > self.propagate(queue.get_next_front(), i + 1, k) - 1e-3
                    ):
                        queue.pop_front()

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
            begin = self.pred[0][end]
            indiv.chromR[k] = indiv.chromT[begin:end]
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

            # Set distance to next customer to negative infinity at the end
            self.splits[i].dnext = (
                self.inst.distances[customer][next_customer]
                if i < self.inst.num_customers - 1
                else -1e30
            )

            self.sum_load[i] = self.sum_load[i - 1] + self.splits[i].demand
            self.sum_distance[i] = (
                self.sum_distance[i - 1] + self.inst.splits[i - 1].dnext
            )

        # Perform splitting
        if not self.split_simple(indiv):
            self.split_lf(indiv)

        indiv.evaluate_complete_cost()
