import numpy as np
import random
from allocator import Allocator


class VRPInstance:
    # {{{ Initializer

    def __init__(self, file_name):
        self.numCustomers = 0
        self.numVehicles = 0
        self.vehicleCapacity = 0
        self.demandOfCustomer = []
        self.xCoordOfCustomer = []
        self.yCoordOfCustomer = []

        try:
            with open(file_name, "r") as file:
                # Reading the first line with basic parameters
                first_line = file.readline().strip().split()
                self.numCustomers = int(first_line[0])
                self.numVehicles = int(first_line[1])
                self.vehicleCapacity = int(first_line[2])

                # print(f"Number of customers: {self.numCustomers}")
                # print(f"Number of vehicles: {self.numVehicles}")
                # print(f"Vehicle capacity: {self.vehicleCapacity}")

                # Reading the customer data
                for i in range(self.numCustomers):
                    line = file.readline().strip().split()
                    demand = int(line[0])
                    xCoord = float(line[1])
                    yCoord = float(line[2])

                    self.demandOfCustomer.append(demand)
                    self.xCoordOfCustomer.append(xCoord)
                    self.yCoordOfCustomer.append(yCoord)

        except FileNotFoundError:
            print(f"Error: in VRPInstance() {file_name} - File not found")
            exit(-1)
        except Exception as e:
            print(f"Error: {e}")
            exit(-1)

    # }}}

    def presolve(self):
        self.calc_distance_matrix()
        self.allocator = Allocator(self)

    def calc_distance_matrix(self):
        x_coords = np.array(self.xCoordOfCustomer)
        y_coords = np.array(self.yCoordOfCustomer)

        dx = x_coords[:, np.newaxis] - x_coords
        dy = y_coords[:, np.newaxis] - y_coords

        self.distances = np.sqrt(dx**2 + dy**2)

    def calc_route_distance(self, route):
        total = 0

        total += self.distances[0][route[0]]
        for i, j in zip(route[:-1], route[1:]):
            total += self.distances[i][j]
        total += self.distances[route[-1]][0]

        return total

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

                if new_demand <= self.vehicleCapacity:
                    # Merge the routes
                    for customer in new_route:
                        routes[customer] = new_route
                        route_demands[customer] = new_demand

        # List of routes formed
        vehicle_routes = set()

        for route in routes.values():
            vehicle_routes.add(tuple(route))
        print(vehicle_routes)

        return list(vehicle_routes)

    def solve(self):
        self.presolve()

        customer_alloc = self.allocator.solve()
        routes = [self.clark_wright_savings(route) for route in customer_alloc]

        # print("Routes:", routes)

        obj = round(sum(self.calc_route_distance(r) for r in routes), 2)
        optimal = 0

        return obj, optimal, routes
