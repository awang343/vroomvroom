from hga_structures import Customer
from hga_circle import CircleSector

import math
import numpy as np
import random
from collections import defaultdict


class VRPInstance:
    def __init__(self, file_name):
        
        #self.num_customers includes depot
        self.num_customers = 0
        self.num_vehicles = 0
        self.vehicle_capacity = 0
        self.customers = []

        with open(file_name, "r") as file:
            first_line = file.readline().strip().split()
            self.num_customers = int(first_line[0])
            self.num_vehicles = int(first_line[1])
            self.vehicle_capacity = int(first_line[2])

            # Reading the customer data
            depot_line = file.readline().strip().split()
            depot_x = float(depot_line[1])
            depot_y = float(depot_line[2])
            self.customers.append(Customer(depot_x, depot_y, 0, 0))

            max_demand = 0
            for i in range(1, self.num_customers):
                line = file.readline().strip().split()
                demand = int(line[0])

                x_coord = float(line[1])
                y_coord = float(line[2])
                polar = CircleSector.positive_mod(
                    int(
                        32768.0
                        * math.atan2(y_coord - depot_y, x_coord - depot_x)
                        / math.pi
                    )
                )

                self.customers.append(Customer(x_coord, y_coord, demand, polar))

        x_coords = np.array([c.x for c in self.customers])
        y_coords = np.array([c.y for c in self.customers])

        dx = x_coords[:, np.newaxis] - x_coords
        dy = y_coords[:, np.newaxis] - y_coords

        self.distances = np.sqrt(dx**2 + dy**2)
        self.default_capacity_penalty = np.max(self.distances) / max(
            c.demand for c in self.customers
        )

        self.neighborhoods = {}

    def initNeighbors(self, num_neighbors):
        for i in range(1, self.num_customers):
            # Get the distances from node i to all others
            distance_vec = self.distances[i][1:]

            # Get the indices of the k+1 smallest distances (including self at index i)
            nearest = np.argsort(distance_vec)[: num_neighbors + 1] + 1
            self.neighborhoods[i] = [int(j) for j in nearest if j != i][:num_neighbors]
