from scipy.spatial import distance
from deap import base, creator, tools

import numpy as np
import copy
import random

"""
We will represent the solution in an n-vehicle problem
as [[route1], [route2], [route3], ..., [routen]]
"""


class HGASolver:
    # {{{ __init__
    def __init__(
        self,
        inst,
        population_size=25,
        generation_size=40,
        education_prob=1.0,
        repair_prob=0.5,
        ri_granularity=0.4,
        feasibility_target=0.2,
    ):
        self.inst = inst
        self.neighbors = self.inst.calc_neighbors(ri_granularity)
        self.all_customers = tuple(c for c in range(1, self.inst.numCustomers))

        self.population_size = population_size
        self.generation_size = generation_size

        self.education_prob = education_prob
        self.repair_prob = repair_prob

        self.feasibility_target = feasibility_target
        self.capacity_penalty = self.calc_capacity_penalty()

        self.feasible_population = []
        self.infeasible_population = []

        self.initialize_populations()

    def calc_capacity_penalty(self):
        total_dist = 0
        for i in range(1, self.inst.numCustomers):
            for j in range(i, self.inst.numCustomers):
                total_dist += self.inst.distances[i][j]
        num_pairs = (len(self.all_customers) - 1) * (len(self.all_customers) - 2) / 2

        avg_dist = total_dist / num_pairs
        avg_demand = sum(self.inst.demandOfCustomer) / len(self.all_customers)
        return avg_dist / avg_demand
    
    def initialize_populations(self):
        for it in range(self.population_size * 4):
            self.generate_solution()

    # }}}

    def generate_solution(self):
        """
        Randomly generate a solution
        """
        customers = np.array(self.all_customers)
        np.random.shuffle(customers)

        split_points = np.sort(np.random.randint(0, len(customers) + 1, size=k - 1))

        # Add start and end
        split_points = np.concatenate(([0], split_points, [len(customers)]))

        # Split into sublists
        sublists = [arr[split_points[i]:split_points[i+1]].tolist() for i in range(k)]

        return sublists

    # {{{ evaluate_solution
    def evaluate_solution(self, routes):
        """
        Gives the evaluated fitness of a solution
        """
        feasible = True

        cost = 0
        for route in routes:
            overallocation = self.inst.calc_overallocation(route)
            cost += self.inst.calc_route_distance(route) + overallocation

            if overallocation > 0:
                feasible = False

        return cost, feasible

    # }}}

    # {{{ crossover_solution
    def crossover_solution(self, parent1, parent2):
        """
        Performs PIX crossover for VRP with a single depot and multiple vehicles.

        Each customer is visited exactly once. Routes are lists of customer IDs.
        parent1, parent2: list of routes (list of lists)
        num_vehicles: number of vehicles (routes per individual)

        Returns a new child individual (list of routes).
        """
        # assert len(parent1) == self.inst.numVehicles
        # assert len(parent2) == self.inst.numVehicles

        child = [[] for _ in range(self.inst.numVehicles)]
        visited = set()
        customers = set(self.all_customers)

        # Step 0: Split vehicles into Λ1, Λ2, Λmix
        n1 = random.randint(0, self.inst.numVehicles)
        n2 = random.randint(0, self.inst.numVehicles)
        n1, n2 = sorted((n1, n2))

        idxs = list(range(self.inst.numVehicles))
        random.shuffle(idxs)

        lambda1 = idxs[:n1]
        lambda2 = idxs[n1:n2]
        lambda_mix = idxs[n2:]

        # Step 1a: Copy full routes from P1 for Λ1
        for k in lambda1:
            for cust in parent1[k]:
                if cust not in visited:
                    child[k].append(cust)
                    visited.add(cust)

        # Step 1b: Copy substrings from P1 for Λmix
        for k in lambda_mix:
            route = parent1[k]
            if not route:
                continue
            a, b = (
                sorted(random.sample(range(len(route)), 2))
                if len(route) >= 2
                else (0, len(route))
            )
            for cust in route[a:b]:
                if cust not in visited:
                    child[k].append(cust)
                    visited.add(cust)

        # Step 2: Fill in from P2 (Λ2 and Λmix)
        for k in lambda2 + lambda_mix:
            for cust in parent2[k]:
                if cust not in visited:
                    child[k].append(cust)
                    visited.add(cust)

        # Step 3: Repair phase — insert missing customers
        unvisited = customers - visited
        for cust in unvisited:
            # Choose the route with minimal insertion cost (dummy: smallest route)
            best_route_idx = min(
                range(self.inst.numVehicles), key=lambda k: len(child[k])
            )
            # Insert at best position (simplified: at end)
            child[best_route_idx].append(cust)
            visited.add(cust)

        return child

    # }}}

    # {{{ educate_solution
    def educate_solution(self, routes):
        """
        Perform route improvement using the 9 move operators
        """

        improved_routes = [r.copy() for r in routes]
        improved_cost = self.evaluate_solution(improved_routes)

        improved = True
        while improved:
            improved = False

            # Randomize the order of route and node processing
            route_indices = np.random.permutation(len(improved_routes))

            for r_idx in route_indices:
                route = improved_routes[r_idx]
                if not route:
                    continue

                # Randomize node processing order
                node_indices = np.random.permutation(len(route))

                for u_pos in node_indices:
                    u = route[u_pos]
                    neighbors = self.neighbors[u]

                    # Shuffle neighbors for random processing
                    np.random.shuffle(neighbors)

                    for v in neighbors:
                        # Find which route contains v
                        v_route_idx, v_pos = self.find_node_in_routes(
                            improved_routes, v
                        )
                        v_route = improved_routes[v_route_idx]

                        # Try all 9 moves
                        for move in range(1, 10):
                            new_routes = self.apply_move(
                                improved_routes, r_idx, u_pos, v_route_idx, v_pos, move
                            )

                            if new_routes is None:
                                continue

                            new_cost = self.evaluate_solution(new_routes)

                            if new_cost < improved_cost:
                                improved_routes = new_routes
                                improved_cost = new_cost
                                print(improved_routes, improved_cost)
                                improved = True
                                break

                        if improved:
                            break

                    if improved:
                        break

                if improved:
                    break

        return improved_routes

    def find_node_in_routes(self, routes, node):
        """Find which route contains a node and its position"""
        for r_idx, route in enumerate(routes):
            if node in route:
                return r_idx, route.index(node)
        raise Exception(f"Node {node} not found in routes {routes}")

    # {{{ apply_move
    def apply_move(self, routes, r1_idx, u_pos, r2_idx, v_pos, move_type):
        """
        Apply one of the 9 move operators to the routes.
        """
        new_routes = [r.copy() for r in routes]
        route_u = new_routes[r1_idx]
        route_v = new_routes[r2_idx]
        u = route_u[u_pos]
        v = route_v[v_pos]

        # M1: Remove u and place it after v {{{
        if move_type == 1:
            route_u.pop(u_pos)
            insert_pos = v_pos if r1_idx == r2_idx and u_pos < v_pos else v_pos + 1
            route_v.insert(insert_pos, u)
        # }}}
        # M2/M3: Remove u and x (next node) and place them after v (both orders) {{{
        elif move_type == 2 or move_type == 3:
            if u_pos + 1 >= len(route_u):
                return None

            x = route_u[u_pos + 1]
            route_u.pop(u_pos + 1)
            route_u.pop(u_pos)

            insert_pos = v_pos - 1 if r1_idx == r2_idx and u_pos < v_pos else v_pos + 1

            route_v.insert(insert_pos, x if move_type == 2 else u)
            route_v.insert(insert_pos + 1, u if move_type == 2 else x)
        # }}}
        # M4: Swap u and v{{{
        elif move_type == 4:
            route_u[u_pos], route_v[v_pos] = v, u
        # }}}
        # M5: Swap u and x (next node) with v{{{
        elif move_type == 5:
            if u_pos + 1 >= len(route_u):
                return None

            x = route_u[u_pos + 1]

            if x == v:
                return None

            route_u[u_pos], route_v[v_pos] = v, u
            route_u.pop(u_pos + 1)

            insert_pos = v_pos if r1_idx == r2_idx and u_pos < v_pos else v_pos + 1
            route_v.insert(insert_pos, x)
        # }}}
        # M6: Swap u and x (next node) with v and y (next node){{{
        elif move_type == 6:
            if (
                u_pos + 1 >= len(route_u)
                or v_pos + 1 >= len(route_v)
                or u_pos + 1 == v_pos
                or v_pos + 1 == u_pos
            ):
                return None

            x = route_u[u_pos + 1]
            y = route_v[v_pos + 1]

            route_u[u_pos], route_v[v_pos] = v, u
            route_u[u_pos + 1], route_v[v_pos + 1] = y, x
        # }}}
        # M7: 2-opt intra-route move (replace (u,x) and (v,y) with (u,v) and (x,y)){{{
        elif move_type == 7 and r1_idx == r2_idx:
            if u_pos + 1 >= len(route_u) or v_pos + 1 >= len(route_u):
                return None

            # Ensure u comes before v in the route
            if u_pos >= v_pos:
                return None

            # Perform 2-opt swap
            new_route = (
                route_u[: u_pos + 1] + route_u[v_pos:u_pos:-1] + route_u[v_pos + 1 :]
            )
            new_routes[r1_idx] = new_route
        # }}}
        # {{{ M8: 2-opt inter-route move (replace (u,x) and (v,y) with (u,v) and (x,y))
        elif move_type == 8 and r1_idx != r2_idx:
            if u_pos + 1 >= len(route_u) or v_pos + 1 >= len(route_v):
                return None

            # Split routes after u and v
            part1 = route_u[: u_pos + 1]
            part2 = route_u[u_pos + 1 :]
            part3 = route_v[: v_pos + 1]
            part4 = route_v[v_pos + 1 :]

            # Recombine
            new_route1 = part1 + part3[::-1]
            new_route2 = part2[::-1] + part4

            new_routes[r1_idx] = new_route1
            new_routes[r2_idx] = new_route2
        # }}}
        # M9: 2-opt inter-route move (replace (u,x) and (v,y) with (u,y) and (x,v)){{{
        elif move_type == 9 and r1_idx != r2_idx:
            if u_pos + 1 >= len(route_u) or v_pos + 1 >= len(route_v):
                return None

            # Split routes after u and v
            part1 = route_u[: u_pos + 1]
            part2 = route_u[u_pos + 1 :]
            part3 = route_v[: v_pos + 1]
            part4 = route_v[v_pos + 1 :]

            # Recombine
            new_route1 = part1 + part4
            new_route2 = part2 + part3

            new_routes[r1_idx] = new_route1
            new_routes[r2_idx] = new_route2
        # }}}
        else:
            return None

        return new_routes

    # }}}

    # }}}
    
    import random

    def select_parents(self, distance_matrix, elite_bias=0.7):
        """
        Selects two parents using binary tournament selection with elitism bias.
        
        Args:
            population: List of solutions (each solution is a list of routes)
            distance_matrix: 2D list of distances between nodes
            elite_bias: Probability to select the fitter candidate in a tournament
        
        Returns:
            Two selected parent solutions
        """
        def evaluate(solution):
            """Helper: Calculate total distance of a solution"""
            total = 0
            for route in solution:
                if not route:
                    continue
                total += distance_matrix[0][route[0]]  # Depot to first
                for i in range(len(route)-1):
                    total += distance_matrix[route[i]][route[i+1]]
                total += distance_matrix[route[-1]][0]  # Last to depot
            return total

        def run_tournament():
            """Run a single binary tournament"""
            candidates = random.sample(self.feasible_population.extend(self.infeasible_population), 2)
            cost1, cost2 = evaluate(candidates[0]), evaluate(candidates[1])
            
            # Select fitter candidate with elite_bias probability
            if cost1 < cost2:
                return candidates[0] if random.random() < elite_bias else candidates[1]
            else:
                return candidates[1] if random.random() < elite_bias else candidates[0]

        # Select two parents via independent tournaments
        parent1 = run_tournament()
        parent2 = run_tournament()
        
        return parent1, parent2

    def cull_population(self):
        """
        Culls population down to mu individuals from mu+lambda
        """
        new_feasible_population = []
        new_infeasible_population = []
        
        if len(self.feasible_population) > self.population_size + self.generation_size:
            for individual in self.feasible_population:
                if self.feasible_queue_miu[0][0] <= individual[0]:
                    new_feasible_population.append(individual[0])
        else:
            new_feasible_population = self.feasible_population
                
        if len(self.feasible_population) > self.population_size + self.generation_size:
            for indvidual in self.infeasible_population:
                if self.infeasible_queue_miu[0][0] <= individual[0]:
                    new_feasible_population.append(individual[0])
        else:
            new_infeasible_population = self.infeasible_population
        
        # Remove up to lambda clones with worst biased fitness
        # for ind in subpop:
        #     ind['biased_fitness'] = self.calc_biased_fitness(ind)
        return (new_feasible_population, new_infeasible_population)


    def nuke_population():
        """
        Resets all but mu/3 individuals
        """
        new_feasible_population = []
        new_infeasible_population = []
        
        
        for individual in self.feasible_population:
            if self.feasible_queue_miu_3[0][0] <= individual[0]:
                new_feasible_population.append(individual[0])
        
        
        for indvidual in self.infeasible_population:
            if self.infeasible_queue_miu_3[0][0] <= individual[0]:
                new_feasible_population.append(individual[0])
        
        
        #TODO: Repopulate with random individuals
        return (new_feasible_population, new_infeasible_population)
        
        

    def solve(self):
        # print(self.pix_crossover([[1, 4], [2, 3], [], []], [[1], [2], [3], [4]]))
        self.educate_solution([[1, 4], [2, 3], [], []])
        # print(self.inst.calc_neighbors(0.4))
        return 1, 1, [[1], [2], [3], [4]]
