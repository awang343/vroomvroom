import numpy as np
from dataclasses import dataclass
from docplex.mp.model import Model
import math


class Allocator:
    def __init__(self, inst):
        self.numVehicles = inst.numVehicles
        self.numCustomers = inst.numCustomers - 1
        self.vehicleCapacity = inst.vehicleCapacity
        self.customerDemands = inst.demandOfCustomer[1:]
        self.model = Model()

        self.build_constraints()

    def build_constraints(self):
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

    def solve(self):
        self.model.solve()

        return [
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
