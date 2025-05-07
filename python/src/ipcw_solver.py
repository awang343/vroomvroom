import numpy as np
from dataclasses import dataclass
from docplex.mp.model import Model
import math

class IPCWSolver:
    def __init__(self, inst):
        self.numVehicles = inst.numVehicles
        self.numCustomers = inst.numCustomers - 1
        self.vehicleCapacity = inst.vehicleCapacity
        self.customerDemands = inst.demandOfCustomer[1:]
        self.model = Model()
        self.build_ip()

    def build_ip(self):
        self.vehicle_assignment = [
            self.model.integer_var_list(self.numCustomers, lb=0, ub=1)
            for c in range(self.numVehicles)
        ]

        for v in range(self.numVehicles):
            self.model.add_constraint(
                self.model.scal_prod(
                    terms=self.vehicle_assignment[v], coefs=self.customerDemands
                )
                <= self.vehicleCapacity
            )

        for c in range(self.numCustomers):
            self.model.add_constraint(
                sum(self.vehicle_assignment[v][c] for v in range(self.numVehicles)) == 1
            )

    def clark_wright_savings(self, customers):
        savings = []

        for i, c1 in enumerate(customers):
            for c2 in customers[i + 1 :]:
                savings.append(
                    (
                        self.distances[0][c1]
                        + self.distances[0][c2]
                        - self.distances[c1][c2],
                        int(c1),
                        int(c2),
                    )
                )

        savings.sort()

        # Routes for each customer (initially, each customer is its own route)
        routes = {i: [i] for i in customers}
        route_demands = {
            i: self.demandOfCustomer[i] for i in customers
        }

        # Process savings
        while savings:
            saving, i, j = savings.pop(-1)

            # Check if i and j are still in different routes and the merge respects capacity constraints
            route_i = routes[i]
            route_j = routes[j]

            # Check if we can merge
            if route_i != route_j:
                if route_i[-1] == i and route_j[0] == j:
                    new_route = route_i + route_j
                elif route_i[0] == i and route_j[-1] == j:
                    new_route = route_j + route_i
                else:
                    continue

                new_demand = route_demands[i] + route_demands[j]

                for customer in new_route:
                    routes[customer] = new_route
                    route_demands[customer] = new_demand

        # List of routes formed
        vehicle_routes = set()

        return list(routes.values())[0]

    def solve(self):
        self.model.solve()

        customer_alloc = [
            (np.where(
                np.array(
                    [
                        self.vehicle_assignment[i][j].solution_value
                        for j in range(self.numCustomers)
                    ]
                )
                == 1
            )[0]
            + 1).tolist()
            for i in range(self.numVehicles)
        ]

        routes = [self.clark_wright_savings(route) for route in customer_alloc]

        obj = round(sum(self.calc_route_distance(r) for r in routes), 2)
        optimal = 0

        return obj, optimal, routes
