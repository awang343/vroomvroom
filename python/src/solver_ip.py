import numpy as np
from docplex.mp.model import Model

class IPSolver:
    def __init__(self, inst):
        self.numVehicles = inst.numVehicles
        self.numCustomers = inst.numCustomers - 1
        self.vehicleCapacity = inst.vehicleCapacity
        self.customerDemands = inst.demandOfCustomer[1:]
        self.distances = inst.distances
        self.model = Model()
        self.build_ip()

    def build_ip(self):
        n = self.numCustomers
        V = self.numVehicles
        N = n + 1  # including depot (0)

        # x[v][i][j] = 1 if vehicle v travels from i to j
        self.x = [
            [
                [
                    self.model.binary_var(name=f"x_{v}_{i}_{j}")
                    for j in range(N)
                ]
                for i in range(N)
            ]
            for v in range(V)
        ]

        # Each customer is visited exactly once (by some vehicle, from some predecessor)
        for c in range(1, N):
            self.model.add_constraint(
                self.model.sum(self.x[v][i][c] for v in range(V) for i in range(N) if i != c) == 1
            )

        # Each customer is left exactly once (by some vehicle, to some successor)
        for c in range(1, N):
            self.model.add_constraint(
                self.model.sum(self.x[v][c][j] for v in range(V) for j in range(N) if j != c) == 1
            )

        # Each vehicle leaves depot once and returns once
        for v in range(V):
            self.model.add_constraint(
                self.model.sum(self.x[v][0][j] for j in range(1, N)) == 1
            )
            self.model.add_constraint(
                self.model.sum(self.x[v][i][0] for i in range(1, N)) == 1
            )

        # Flow conservation for each vehicle
        for v in range(V):
            for h in range(1, N):
                self.model.add_constraint(
                    self.model.sum(self.x[v][i][h] for i in range(N) if i != h) ==
                    self.model.sum(self.x[v][h][j] for j in range(N) if j != h)
                )

        # Capacity constraints (MTZ subtour elimination)
        # u_c: load after visiting customer c (for each vehicle)
        self.u = [
            [self.model.continuous_var(lb=0, ub=self.vehicleCapacity, name=f"u_{v}_{c}") for c in range(N)]
            for v in range(V)
        ]
        for v in range(V):
            self.model.add_constraint(self.u[v][0] == 0)  # depot load is 0
            for i in range(1, N):
                self.model.add_constraint(self.u[v][i] >= self.customerDemands[i - 1] * self.model.sum(self.x[v][j][i] for j in range(N) if j != i))
                self.model.add_constraint(self.u[v][i] <= self.vehicleCapacity)
            for i in range(N):
                for j in range(1, N):
                    if i != j:
                        self.model.add_constraint(
                            self.u[v][j] >= self.u[v][i] + (self.customerDemands[j - 1] if j > 0 else 0) - self.vehicleCapacity * (1 - self.x[v][i][j])
                        )

        # Objective: minimize total distance
        self.model.minimize(
            self.model.sum(
                self.distances[i][j] * self.x[v][i][j]
                for v in range(V)
                for i in range(N)
                for j in range(N)
                if i != j
            )
        )

    def solve(self):
        self.model.solve()
        N = self.numCustomers + 1
        V = self.numVehicles

        # Extract routes for each vehicle
        routes = []
        for v in range(V):
            route = [0]
            visited = set([0])
            current = 0
            while True:
                found = False
                for j in range(N):
                    if j != current and abs(self.x[v][current][j].solution_value) > 0.5:
                        route.append(j)
                        visited.add(j)
                        current = j
                        found = True
                        break
                if not found or current == 0:
                    break
            if len(route) > 1:
                if route[-1] != 0:
                    route.append(0)
                routes.append(route)
        obj = round(sum(self.distances[route[i]][route[i + 1]] for route in routes for i in range(len(route) - 1)), 2)
        optimal = 0
        return obj, optimal, routes