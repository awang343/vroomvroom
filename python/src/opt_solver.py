import hygese as hgs
import numpy as np

class OptSolver:
    def __init__(self, inst):
        self.numVehicles = inst.numVehicles
        self.numCustomers = inst.numCustomers - 1
        self.vehicleCapacity = inst.vehicleCapacity
        self.customerDemands = inst.demandOfCustomer
        self.distances = inst.distances
        
        self.ap = hgs.AlgorithmParameters(timeLimit=3.2)  # seconds
        self.hgs_solver = hgs.Solver(parameters=self.ap, verbose=True)
        
    def solve(self):
        data = dict()
        data['distance_matrix'] = self.distances
        data['num_customers'] = self.numCustomers
        data['vehicle_capacity'] = self.vehicleCapacity
        data['demands'] = self.customerDemands
        result = self.hgs_solver.solve_cvrp(data)
        print(result.cost)
        print(result.routes)
        
        return (result.cost, 0, result.routes)
        

# n = 20
# x = (np.random.rand(n) * 1000)
# y = (np.random.rand(n) * 1000)

# # Solver initialization
# ap = hgs.AlgorithmParameters(timeLimit=3.2)  # seconds
# hgs_solver = hgs.Solver(parameters=ap, verbose=True)

# # data preparation
# data = dict()
# data['x_coordinates'] = x
# data['y_coordinates'] = y

# # You may also supply distance_matrix instead of coordinates, or in addition to coordinates
# # If you supply distance_matrix, it will be used for cost calculation.
# # The additional coordinates will be helpful in speeding up the algorithm.
# # data['distance_matrix'] = dist_mtx

# data['service_times'] = np.zeros(n)
# demands = np.ones(n)
# demands[0] = 0 # depot demand = 0
# data['demands'] = demands
# data['vehicle_capacity'] = np.ceil(n/3).astype(int)
# data['num_vehicles'] = 3
# data['depot'] = 0

# result = hgs_solver.solve_cvrp(data)
# print(result.cost)
# print(result.routes)