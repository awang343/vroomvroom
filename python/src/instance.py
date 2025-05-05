import numpy as np
import random


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

        self.calc_distance_matrix()

    def calc_distance_matrix(self):
        x_coords = np.array(self.xCoordOfCustomer)
        y_coords = np.array(self.yCoordOfCustomer)

        dx = x_coords[:, np.newaxis] - x_coords
        dy = y_coords[:, np.newaxis] - y_coords

        self.distances = np.sqrt(dx**2 + dy**2)

    # }}}

    def calc_route_distance(self, route):
        """
        Calculates distance of a customer route
        Assumes that input does not include depot
        """
        if not route:
            return 0
        return (
            self.distances[0][route[0]]
            + sum(self.distances[i][j] for i, j in zip(route, route[1:]))
            + self.distances[route[-1]][0]
        )

    def calc_allocation(self, route):
        """
        Calculates the demand allocation to a given vehicle
        """
        return sum(self.demandOfCustomer[c] for c in route)

    def calc_neighbors(self, h):
        """
        Returns a dictionary mapping each customer
        to the hn closest neighbors
        """

        n = self.numCustomers
        k = int(h * n)
        neighbor_dict = {}

        for i in range(1, n):
            # Get the distances from node i to all others
            distances = self.distances[i][1:]

            # Get the indices of the k+1 smallest distances (including self at index i)
            nearest = np.argsort(distances)[: k + 1] + 1
            nearest = [int(j) for j in nearest if j != i][:k]

            neighbor_dict[i] = nearest

        return neighbor_dict

    def calc_feasible(self, route):
        return self.calc_allocation(route) <= self.vehicleCapacity
    
    def _test_cost(self, routes, multiplier):
        print("Real distance:", sum(
            self.calc_route_distance(route) for route in routes
        ))
        print("Real allocation penalty:", multiplier * sum(
            max(0, self.calc_allocation(route) - self.vehicleCapacity)
            for route in routes
        ))

        return sum(
            self.calc_route_distance(route) for route in routes
        ) + multiplier * sum(
            max(0, self.calc_allocation(route) - self.vehicleCapacity)
            for route in routes
        )
