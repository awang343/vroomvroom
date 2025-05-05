class LocalSearch:
    def __init__(inst):
        print()

    # {{{ educate
    def educate(self, routes):
        """
        Perform route improvement using the 9 move operators
        """

        best_routes = [r.copy() for r in routes]
        best_dists = [self.inst.calc_route_distance(route) for route in best_routes]
        best_alloc = [self.inst.calc_allocation(route) for route in best_routes]
        best_cost = (
            sum(best_dists)
            + sum(max(0, b - self.inst.vehicleCapacity) for b in best_alloc)
            * self.capacity_penalty
        )

        cost_change = None
        while cost_change is None or cost_change < 0:
            # Randomize the order of route and node processing
            route_indices = np.random.permutation(len(best_routes))

            for route_idx in route_indices:
                route = best_routes[route_idx]
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
                        v_route_idx, v_pos = self.find_node_in_routes(best_routes, v)
                        v_route = best_routes[v_route_idx]

                        # Try all 9 moves
                        for move in np.random.permutation(9) + 1:
                            (
                                new_route,
                                new_v_route,
                                new_alloc,
                                new_v_alloc,
                                cost_change,
                            ) = self.apply_move(
                                route,
                                v_route,
                                best_alloc[route_idx],
                                best_alloc[v_route_idx],
                                u_pos,
                                v_pos,
                                move,
                            )

                            if cost_change < -1e-2:
                                best_routes[route_idx] = new_route
                                best_routes[v_route_idx] = new_v_route
                                best_alloc[route_idx] = new_alloc
                                best_alloc[v_route_idx] = new_v_alloc

                                best_cost += cost_change

                                break

                        if cost_change < -1e-2:
                            break

                    if cost_change < -1e-2:
                        break

                if cost_change < -1e-2:
                    break

        return best_cost, best_routes

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
            if same_route and u_pos - v_pos == 1:
                return None, None, None, None, 0

            route_u.pop(u_pos)
            insert_pos = v_pos if same_route and u_pos < v_pos else v_pos + 1
            route_v.insert(insert_pos, u)

            if not same_route:
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
            if x == 0 or (same_route and abs(u_pos - v_pos) == 1):
                return None, None, None, None, 0

            route_u.pop(u_pos + 1)
            route_u.pop(u_pos)
            insert_pos = v_pos - 1 if same_route and u_pos < v_pos else v_pos + 1
            route_v.insert(insert_pos, u if move_type == 2 else x)
            route_v.insert(insert_pos + 1, x if move_type == 2 else u)

            if not same_route:
                new_alloc = alloc - demand_u - demand_x
                new_v_alloc = v_alloc + demand_u + demand_x

            cost_change += (
                self.inst.distances[a][w]
                - self.inst.distances[a][u]
                - self.inst.distances[x][w]
                + self.inst.distances[v][u if move_type == 2 else x]
                + self.inst.distances[x if move_type == 2 else u][y]
                - self.inst.distances[v][y]
            )

        # }}}
        # M4: Swap u and v{{{
        elif move_type == 4:
            if same_route and abs(u_pos - v_pos) == 1:
                return None, None, None, None, 0
            route_u[u_pos], route_v[v_pos] = v, u

            if not same_route:
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
            if x == 0 or (same_route and abs(v_pos - u_pos) <= 2):
                return None, None, None, None, 0

            route_u[u_pos], route_v[v_pos] = v, u
            route_u.pop(u_pos + 1)

            insert_pos = v_pos if same_route and u_pos < v_pos else v_pos + 1
            route_v.insert(insert_pos, x)

            if not same_route:
                new_alloc = alloc - demand_u - demand_x + demand_v
                new_v_alloc = v_alloc + demand_u + demand_x - demand_v

            cost_change += (
                self.inst.distances[a][v]
                + self.inst.distances[v][w]
                + self.inst.distances[b][u]
                + self.inst.distances[x][y]
                - self.inst.distances[a][u]
                - self.inst.distances[x][w]
                - self.inst.distances[b][v]
                - self.inst.distances[v][y]
            )

        # }}}
        # M6: Swap u and x with v and y{{{
        elif move_type == 6:
            if x == 0 or y == 0 or (same_route and abs(u_pos - v_pos) <= 2):
                return None, None, None, None, 0

            route_u[u_pos], route_v[v_pos] = v, u
            route_u[u_pos + 1], route_v[v_pos + 1] = y, x

            if not same_route:
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
            route_v = route_u

            cost_change += (
                self.inst.distances[u][v]
                + self.inst.distances[x][y]
                - self.inst.distances[u][x]
                - self.inst.distances[v][y]
            )

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

            cost_change += (
                self.inst.distances[u][v]
                + self.inst.distances[x][y]
                - self.inst.distances[u][x]
                - self.inst.distances[v][y]
            )

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
            route_v = part3 + part2

            new_alloc = self.inst.calc_allocation(route_u)
            new_v_alloc = self.inst.calc_allocation(route_v)

            cost_change += (
                self.inst.distances[u][y]
                + self.inst.distances[v][x]
                - self.inst.distances[u][x]
                - self.inst.distances[v][y]
            )
        # }}}
        else:
            raise Exception("Selected move does not exist")

        cost_change += (
            max(new_alloc, self.inst.vehicleCapacity)
            - max(alloc, self.inst.vehicleCapacity)
            + max(new_v_alloc, self.inst.vehicleCapacity)
            - max(v_alloc, self.inst.vehicleCapacity)
        ) * self.capacity_penalty

        return route_u, route_v, new_alloc, new_v_alloc, cost_change

    # }}}
