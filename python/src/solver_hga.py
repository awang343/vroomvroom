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
        population_size=1,
        generation_size=40,
        education_prob=1.0,
        repair_prob=0.5,
        ri_granularity=0.3,
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

        self.generate_solution()

    def calc_capacity_penalty(self):
        total_dist = 0
        for i in range(1, self.inst.numCustomers):
            for j in range(i, self.inst.numCustomers):
                total_dist += self.inst.distances[i][j]
        num_pairs = (len(self.all_customers) - 1) * (len(self.all_customers) - 2) / 2

        avg_dist = total_dist / num_pairs
        avg_demand = sum(self.inst.demandOfCustomer) / len(self.all_customers)
        return avg_dist / avg_demand

    def repopulate(self):
        for it in range(self.population_size * 4):
            solution = self.generate_solution()
            print("Iteration:", it)

            feasible = all(
                self.inst.calc_allocation(route) <= self.inst.vehicleCapacity
                for route in solution
            )

            if feasible:
                self.feasible_population.append(solution)
            else:
                self.infeasible_population.append(solution)

            print(len(self.feasible_population), len(self.infeasible_population))

    # }}}

    # {{{ generate_solution
    def generate_solution(self):
        """
        Randomly generate a solution
        """
        customers = np.array(self.all_customers)
        k = self.inst.numVehicles

        np.random.shuffle(customers)
        split_points = np.sort(np.random.randint(0, len(customers) + 1, size=k - 1))
        split_points = np.concatenate(([0], split_points, [len(customers)]))
        solution = [
            customers[split_points[i] : split_points[i + 1]].tolist() for i in range(k)
        ]

        return self.educate_solution(solution)

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

        best_routes = [r.copy() for r in routes]
        best_dists = [self.inst.calc_route_distance(route) for route in best_routes]
        best_alloc = [self.inst.calc_allocation(route) for route in best_routes]
        best_cost = sum(best_dists) + sum(best_allocations) * self.capacity_penalty

        change = None
        while change < 0 or change is None:
            # Randomize the order of route and node processing
            route_indices = np.random.permutation(len(improved_routes))

            for route_idx in route_indices:
                route = improved_routes[route_idx]
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
                            best_routes, v
                        )
                        v_route = best_routes[v_route_idx]

                        # Try all 9 moves
                        for move in np.random.permutation(10):
                            (
                                new_route,
                                new_v_route,
                                new_alloc,
                                new_v_alloc,
                                cost_change,
                            ) = self.apply_move(
                                route,
                                v_route,
                                alloc,
                                v_alloc,
                                u_pos,
                                v_pos,
                                move,
                            )

                            if cost_change < 0:
                                best_routes[route_idx] = new_route
                                best_routes[v_route_idx] = new_v_route
                                best_alloc[route_idx] = new_alloc
                                best_alloc[v_route_idx] = new_v_alloc
                                best_cost += cost_change
                                break

                        if cost_change < 0:
                            break

                    if cost_change < 0:
                        break

                if cost_change < 0:
                    break

        return best_routes

    def find_node_in_routes(self, routes, node):
        """Find which route contains a node and its position"""
        for r_idx, route in enumerate(routes):
            if node in route:
                return r_idx, route.index(node)
        raise Exception(f"Node {node} not found in routes {routes}")

    # }}}

    # {{{ apply_move
    def apply_move(self, route, v_route, alloc, v_alloc, u_pos, v_pos, move_type):
        """
        Apply one of the 9 move operators to the routes.
        """
        # {{{ Variable Setup
        same_route = route[0] == v_route[0]

        route_u = route.copy()
        route_v = route_u if same_route else v_route.copy()
        new_alloc = alloc
        new_v_alloc = v_alloc
        cost_change = 0

        # -1 in route
        a = route_u[u_pos - 1] if u_pos - 1 >= 0 else 0
        b = route_v[v_pos - 1] if v_pos - 1 >= 0 else 0

        # +0 in route
        u = route_u[u_pos]
        v = route_v[v_pos]
        demand_u = self.inst.demandOfCustomer[u]
        demand_v = self.inst.demandOfCustomer[v]

        # +1 in route
        x = route_u[u_pos + 1] if u_pos + 1 < len(route_u) else 0
        y = route_v[v_pos + 1] if v_pos + 1 < len(route_v) else 0
        demand_x = self.inst.demandOfCustomer[x]
        demand_y = self.inst.demandOfCustomer[y]

        # +2 in route
        w = route_u[u_pos + 2] if u_pos + 2 < len(route_u) else 0
        z = route_v[v_pos + 2] if v_pos + 2 < len(route_v) else 0
        # }}}

        # M1: Remove u and place it after v {{{
        if move_type == 1:
            route_u.pop(u_pos)
            insert_pos = v_pos if same_route and u_pos < v_pos else v_pos + 1
            route_v.insert(insert_pos, u)
            new_alloc = alloc - demand_u
            new_v_alloc = v_alloc + demand_u

            cost_change += (
                self.inst.distances[a][x]
                - self.inst.distances[a][u]
                - self.inst.distances[u][x]
                + self.inst.distances[v][u]
                + self.inst.distances[u][y]
                - self.inst.distances[v][y]
            )

        # }}}
        # M2/M3: Remove u and x and place them after v (both orders) {{{
        elif move_type == 2 or move_type == 3:
            if x == 0:
                return None, None, None, None, 0

            route_u.pop(u_pos + 1)
            route_u.pop(u_pos)
            insert_pos = v_pos - 1 if same_route and u_pos < v_pos else v_pos + 1
            route_v.insert(insert_pos, x if move_type == 2 else u)
            route_v.insert(insert_pos + 1, u if move_type == 2 else x)

            new_alloc = alloc - demand_u - demand_x
            new_v_alloc = v_alloc + demand_u + demand_x

            cost_change += (
                self.inst.distances[a][x]
                - self.inst.distances[a][u]
                - self.inst.distances[x][w]
                + self.inst.distances[v][u if move_type == 2 else x]
                + self.inst.distances[x if move_type == 2 else u][y]
                - self.inst.distances[v][y]
            )
        # }}}
        # M4: Swap u and v{{{
        elif move_type == 4:
            route_u[u_pos], route_v[v_pos] = v, u

            new_alloc = alloc - demand_u + demand_v
            new_v_alloc = v_alloc + demand_u - demand_v

            cost_change += (
                self.inst.distances[a][v]
                + self.inst.distances[v][x]
                - self.inst.distances[a][u]
                - self.inst.distances[u][x]
                + self.inst.distances[b][u]
                + self.inst.distances[u][y]
                - self.inst.distances[b][v]
                - self.inst.distances[v][y]
            )
        # }}}
        # M5: Swap u and x with v{{{
        elif move_type == 5:
            if x == 0 or x == v:
                return None, None, None, None, 0

            route_u[u_pos], route_v[v_pos] = v, u
            route_u.pop(u_pos + 1)

            insert_pos = v_pos if same_route and u_pos < v_pos else v_pos + 1
            route_v.insert(insert_pos, x)

            new_alloc = alloc - demand_u - demand_x + demand_v
            new_v_alloc = v_alloc + demand_u + demand_x - demand_v

            cost_change += (
                self.inst.distances[a][v]
                + self.inst.distances[v][x]
                - self.inst.distances[a][u]
                - self.inst.distances[u][x]
                + self.inst.distances[b][u]
                + self.inst.distances[u][y]
                - self.inst.distances[b][v]
                - self.inst.distances[v][y]
            )
        # }}}
        # M6: Swap u and x with v and y{{{
        elif move_type == 6:
            if x == 0 or y == 0 or x == v or y == u:
                return None, None, None, None, 0

            route_u[u_pos], route_v[v_pos] = v, u
            route_u[u_pos + 1], route_v[v_pos + 1] = y, x

            new_alloc = alloc - demand_u - demand_x + demand_v + demand_y
            new_v_alloc = v_alloc + demand_u + demand_x - demand_v - demand_y

            cost_change += (
                self.inst.distances[a][v]
                + self.inst.distances[y][w]
                - self.inst.distances[a][u]
                - self.inst.distances[x][w]
                + self.inst.distances[b][u]
                + self.inst.distances[x][z]
                - self.inst.distances[b][v]
                - self.inst.distances[y][z]
            )
        # }}}
        # M7: 2-opt intra-route move {{{
        elif move_type == 7:
            if not same_route or x == 0 or y == 0 or u_pos >= v_pos:
                return None, None, None, None, 0

            route_u = (
                route_u[: u_pos + 1] + route_u[v_pos:u_pos:-1] + route_u[v_pos + 1 :]
            )

            return route_u, route_v, alloc, v_alloc, cost_change
        # }}}
        # {{{ M8: 2-opt inter-route move type 1
        elif move_type == 8:
            if x == 0 or y == 0 or same_route:
                return None, None, None, None, 0

            # Split routes after u and v
            part1 = route_u[: u_pos + 1]
            part2 = route_u[u_pos + 1 :]
            part3 = route_v[: v_pos + 1]
            part4 = route_v[v_pos + 1 :]

            # Recombine
            route_u = part1 + part3[::-1]
            route_v = part2[::-1] + part4

            new_alloc = self.inst.calc_allocation(route_u)
            new_v_alloc = self.inst.calc_allocation(route_v)

            return route_u, route_v, new_alloc, new_v_alloc, cost_change
        # }}}
        # M9: 2-opt inter-route move type 2 {{{
        elif move_type == 9:
            if x == 0 or y == 0 or same_route:
                return None, None, None, None, 0

            # Split routes after u and v
            part1 = route_u[: u_pos + 1]
            part2 = route_u[u_pos + 1 :]
            part3 = route_v[: v_pos + 1]
            part4 = route_v[v_pos + 1 :]

            # Recombine
            route_u = part1 + part4
            route_v = part2 + part3

            new_alloc = self.inst.calc_allocation(route_u)
            new_v_alloc = self.inst.calc_allocation(route_v)

            return route_u, route_v, new_alloc, new_v_alloc, cost_change
        # }}}
        else:
            raise Exception("Selected move does not exist")

        cost_change += (
            max(new_alloc, 0)
            - max(alloc, 0)
            + max(new_alloc_v, 0)
            - max(alloc_v, 0)
        ) * self.capacity_penalty

        return route_u, route_v, new_alloc, new_v_alloc, cost_change
    # }}}

# {{{ select_parents
    def select_parents(population, distance_matrix, elite_bias=0.7):
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
            candidates = random.sample(population, 2)
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
# }}}

# {{{ cull_population, nuke_population
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
        # }}}

    def solve(self):
        # print(self.pix_crossover([[1, 4], [2, 3], [], []], [[1], [2], [3], [4]]))
        # self.educate_solution([[1, 4], [2, 3], [], []])
        # print(self.inst.calc_neighbors(0.4))
        return 1, 1, [[1], [2], [3], [4]]
