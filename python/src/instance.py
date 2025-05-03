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
    # }}}

    def calc_distance_matrix(self):
        x_coords = np.array(self.xCoordOfCustomer)
        y_coords = np.array(self.yCoordOfCustomer)

        dx = x_coords[:, np.newaxis] - x_coords
        dy = y_coords[:, np.newaxis] - y_coords

        self.distances = np.sqrt(dx**2 + dy**2)

    def calc_route_distance(self, route):
        total = 0

        for i, j in zip(route[:-1], route[1:]):
            total += self.distances[i][j]

        return total

