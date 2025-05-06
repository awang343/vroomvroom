from hga_structures import Customer
from hga_circle import CircleSector

import math
import numpy as np
import random
from collections import defaultdict


class VRPInstance:
    def __init__(self, file_name):
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
            depot_x = float(line[1])
            depot_y = float(line[1])
            self.customers.append(depot_x, depot_y, 0, 0)

            max_demand = 0
            for i in range(1, self.num_customers):
                line = file.readline().strip().split()
                demand = int(line[0])
                max_demand = max(max_demand, demand)

                x_coord = float(line[1])
                y_coord = float(line[2])
                polar = positive_mod(
                    32768.0
                    * math.atan2(y_coord - depot_y, x_coord - depot_x)
                    / math.pi
                )

                self.customers.append(Client(x_coord, y_coord, demand, polar))

        x_coords = np.array(c.x for c in self.customers)
        y_coords = np.array(c.y for c in self.customers)

        dx = x_coords[:, np.newaxis] - x_coords
        dy = y_coords[:, np.newaxis] - y_coords

        self.distances = np.sqrt(dx**2 + dy**2)
        max_dist = np.max(self.distances)

        self.penalty_capacity = max_dist / max_demand
        self.neighborhoods = {}

    def init_neighbors(self, num_neighbors):
        for i in range(1, self.num_customers):
            # Get the distances from node i to all others
            distance_vec = self.distances[i][1:]

            # Get the indices of the k+1 smallest distances (including self at index i)
            nearest = np.argsort(distance_vec)[: k + 1] + 1
            self.neighborhoods[i] = [int(j) for j in nearest if j != i][:k]
