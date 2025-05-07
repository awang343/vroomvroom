from hga_structures import Node, Route, ThreeBestInsert


class LocalSearch:
    # {{{ __init__
    def __init__(self, solver):
        self.inst = solver.inst
        self.params = solver.params
        self.capacity_penalty_ls = 0 # To be set on each run

        self.customers = [Node() for _ in range(self.inst.num_customers)]
        self.routes = [Route() for _ in range(self.inst.num_vehicles)]
        self.depots = [Node() for _ in range(self.inst.num_vehicles)]
        self.end_depots = [Node() for _ in range(self.inst.num_vehicles)]

        # Keep track of the three best insertion positions
        # for each customer in each route
        self.best_inserts = [
            [ThreeBestInsert() for _ in range(self.inst.num_customers)]
            for _ in range(self.inst.num_vehicles)
        ]

        # Initialize client nodes with index and depot
        for i in range(self.inst.num_customers):
            self.customers[i].idx = i
            self.customers[i].is_depot = False

        # Initialize routes and depots
        for i in range(self.inst.num_vehicles):
            self.routes[i].idx = i
            self.routes[i].depot = self.depots[i]

            self.depots[i].idx = 0
            self.depots[i].is_depot = True
            self.depots[i].route = self.routes[i]

            self.end_depots[i].idx = 0
            self.end_depots[i].is_depot = True
            self.end_depots[i].route = self.routes[i]

        # Order vectors for heuristic use
        self.order_nodes = list(range(1, self.inst.num_customers))
        self.order_routes = list(range(self.inst.num_vehicles))

        self.empty_routes = set()

    # }}}

    # {{{ run
    def run(self, indiv, capacity_penalty):
        self.capacity_penalty = capacity_penalty
        self.loadIndividual(indiv)  # Set up LS for specific individual

        # Shuffling the order of the nodes explored by the LS to allow for more diversity in the search
        random.shuffle(self.order_nodes)
        random.shuffle(self.order_routes)
        for i in range(1, self.inst.num_customers):
            if random.randint(0, self.params.neighborhood_size - 1) == 0:
                random.shuffle(self.inst.neighborhoods[i])

        self.search_completed = False
        self.loop_id = 0

        while not self.search_completed:
            if self.loop_id > 1:
                # Force at least 2 loops to make sure all moves are checked
                self.search_completed = True

            # CLASSICAL ROUTE IMPROVEMENT (RI) MOVES IN NEIGHBORHOOD
            for pos_u in range(self.inst.num_customers):
                self.node_u = self.clients[self.order_nodes[pos_u]]
                last_test_RI_node_u = self.node_u.last_tested_RI
                self.node_u.last_tested_RI = self.num_moves

                for pos_v in range(len(self.inst.neighborhood[self.node_u.idx])):
                    self.node_v = self.customers[
                        self.inst.neighborhood[self.node_u.idx][pos_v]
                    ]

                    if (
                        self.loop_id == 0
                        or max(
                            self.node_u.route.last_modified,
                            self.node_v.route.last_modified,
                        )
                        > last_test_RI_node_u
                    ):
                        # Only evaluate moves involving routes that have
                        # been modified since last move evaluations for node_u
                        self.setLocalVariables()

                        if self.move1():
                            continue  # RELOCATE
                        if self.move2():
                            continue  # RELOCATE
                        if self.move3():
                            continue  # RELOCATE
                        if self.node_u.idx <= self.node_v.idx and self.move4():
                            continue  # SWAP
                        if self.move5():
                            continue  # SWAP
                        if self.node_u.idx <= self.node_v.idx and self.move6():
                            continue  # SWAP
                        if self.intraroute and self.move7():
                            continue  # 2-OPT
                        if not self.intraroute and self.move8():
                            continue  # 2-OPT*
                        if not self.intraroute and self.move9():
                            continue  # 2-OPT*

                        # Trying moves that insert node_u directly after the depot
                        if self.node_v.prev.is_depot:
                            self.node_v = self.node_v.prev
                            self.setLocalVariables()
                            if self.move1():
                                continue  # RELOCATE
                            if self.move2():
                                continue  # RELOCATE
                            if self.move3():
                                continue  # RELOCATE
                            if not self.intraroute and self.move8():
                                continue  # 2-OPT*
                            if not self.intraroute and self.move9():
                                continue  # 2-OPT*

                # MOVES INVOLVING AN EMPTY ROUTE
                if self.loop_id > 0 and self.empty_routes:
                    self.node_v = self.routes[self.empty_routes.pop(0)].depot
                    self.setLocalVariables()
                    if self.move1():
                        continue  # RELOCATE
                    if self.move2():
                        continue  # RELOCATE
                    if self.move3():
                        continue  # RELOCATE
                    if self.move9():
                        continue  # 2-OPT*

            # (SWAP*) MOVES LIMITED TO ROUTE PAIRS WHOSE CIRCLE SECTORS OVERLAP
            for rU in range(self.inst.num_vehicles):
                self.route_u = self.routes[self.order_routes[rU]]
                last_test_SWAP_route_u = self.route_u.last_tested_SWAP
                self.route_u.last_tested_SWAP = self.num_moves

                for rV in range(self.inst.num_vehicles):
                    self.route_v = self.routes[self.order_routes[rV]]
                    if (
                        self.route_u.nbCustomers > 0
                        and self.route_v.nbCustomers > 0
                        and self.route_u.idx < self.route_v.idx
                    ):
                        if (
                            self.loop_id == 0
                            or max(
                                self.route_u.last_modified,
                                self.route_v.last_modified,
                            )
                            > last_test_SWAP_route_u
                        ):
                            if CircleSector.overlap(
                                self.route_u.sector, self.route_v.sector
                            ):
                                self.swapStar()

            self.loop_id += 1

        # Register the solution produced by the LS in the individual
        self.exportIndividual(indiv)

    # }}}

    # {{{ loadIndividual
    def loadIndividual(self, indiv):
        self.empty_routes.clear()
        self.num_moves = 0

        for r in range(self.inst.num_vehicles):
            start_depot_ls = self.depots[r]
            end_depot_ls = self.end_depots[r]
            route_ls = self.routes[r]

            # Connect the depot start and end
            start_depot_ls.prev = end_depot
            end_depot_ls.next = start_depot

            route_indiv = indiv.chromR[r]
            if route_indiv:  # Route in individual is not empty
                # Transfer customer info to LS
                customer_ls = self.customers_ls[route_indiv[0]]
                customer_ls.route = route
                customer_ls.prev = start_depot_ls
                start_depot_ls.next = customer_ls

                for idx in route_indiv[1:]:
                    prev_customer_ls = customer_ls
                    customer_ls = self.customers[idx]
                    customer_ls.prev = prev_customer_ls
                    prev_customer_ls.next = customer_ls
                    customer_ls.route = route_ls

                customer_ls.next = end_depot_ls
                end_depot_ls.prev = customer_ls
            else:
                # Route is empty, just link depot to depotEnd
                start_depot_ls.next = end_depot_ls
                end_depot_ls.prev = start_depot_ls

            self.updateRouteData(route_ls)
            route_ls.last_tested_SWAP = -1

            # Reset insertion memory for this route
            for i in range(1, self.inst.num_customers + 1):
                self.best_insert_client[r][i].last_calculated = -1

        # Reset RI test memory for all clients
        for i in range(1, self.inst.num_customers + 1):
            self.customers[i].last_tested_RI = -1

    # }}}

    # {{{ Variable Aliasing
    def setLocalVariables(self):
        self.route_u = self.node_u.route
        self.route_v = self.node_v.route
        self.load_u = self.inst.customers[self.node_u.idx].demand
        self.load_x = self.inst.customers[self.node_x.idx].demand
        self.load_v = self.inst.customers[self.node_v.idx].demand
        self.load_y = self.inst.customers[self.node_y.idx].demand
        self.intraroute = self.route_u == self.route_v
    # }}}

    # {{{ exportIndividual
    def exportIndividual(self, indiv):
        # Create a list of (polar angle, route index) tuples
        routePolarAngles = [
            (self.routes[r].polarAngleBarycenter, r)
            for r in range(self.inst.num_vehicles)
        ]

        # Sort routes by polar angle (empty routes have 1e30, so they go to the end)
        routePolarAngles.sort()

        pos = 0
        for _, route_index in routePolarAngles:
            indiv.chromR[route_index].clear()
            node = self.depots[route_index].next
            while not node.is_depot:
                indiv.chromT[pos] = node.idx
                indiv.chromR[route_index].append(node.idx)
                node = node.next
                pos += 1

        # Evaluate total cost using problem parameters
        indiv.evaluateCompleteCost()

    # }}}

    # {{{ updateRouteData
    def updateRouteData(self, route_ls):
        """
        Recompute route metrics and metadata after making a move
        """

        place = 0
        load = 0
        time = 0
        reversal_distance = 0
        cum_x = 0
        cum_y = 0

        node = route_ls.depot
        node.position = 0
        node.cum_load = 0
        node.cum_time = 0
        node.cum_reversal_distance = 0

        do = True
        while do or (not node.is_depot):
            # Loop through each node in myRoute
            node = node.next
            place += 1
            node.position = place

            customer = self.inst.customers[node.idx]
            load += customer.demand

            prev_idx = node.prev.idx
            time += self.inst.distances[prev_idx][node.idx]

            reversal_distance += (
                self.inst.distances[node.idx][prev_idx]
                - self.inst.distances[prev_idx][node.idx]
            )

            mynode.cum_load = load
            mynode.cum_time = time
            mynode.cum_reversal_distance = reversal_distance

            if not node.is_depot:
                cum_x += customer.coordX
                cum_y += customer.coordY
                if do:
                    route_ls.sector.initialize(customer.polar)
                else:
                    route_ls.sector.extend(customer.polar)

            do = False

        route_ls.load = load
        route_ls.penalty = self.penaltyExcessLoad(load)
        route_ls.num_customers = myplace - 1
        route_ls.reversal_distance = reversal_distance
        route_ls.last_modified = self.num_moves

        if route_ls.num_customers == 0:
            route_ls.polar_angle_barycenter = 1e30
            self.empty_routes.add(route_ls.idx)
        else:
            avg_x = cum_x / route_ls.num_customers
            avg_y = cum_y / route_ls.num_customers
            depot_x = self.inst.customers[0].x
            depot_y = self.inst.customers[0].y
            route_ls.polar_angle_barycenter = math.atan2(
                avg_y - depot_y, avg_x - depot_x
            )
            self.empty_routes.discard(route_ls.idx)

    # }}}

    # {{{ Route Operations (insertNode and swapNode)
    def insertNode(self, U, V):
        # Remove U from its current position
        U.prev.next = U.next
        U.next.prev = U.prev

        # Insert U after V
        V.next.prev = U
        U.prev = V
        U.next = V.next
        V.next = U

        # Update route
        U.route = V.route

    def swapNode(self, U, V):
        myVPred = V.prev
        myVSuiv = V.next
        myUPred = U.prev
        myUSuiv = U.next
        myRouteU = U.route
        myRouteV = V.route

        # Reconnect neighbors
        myUPred.next = V
        myUSuiv.prev = V
        myVPred.next = U
        myVSuiv.prev = U

        # Swap U and V pointers
        U.prev = myVPred
        U.next = myVSuiv
        V.prev = myUPred
        V.next = myUSuiv

        # Swap routes
        U.route = myRouteV
        V.route = myRouteU

    # }}}

    # {{{ preprocessInsertions
    def preprocessInsertions(self, R1, R2):
        U = R1.depot.next
        while not U.is_depot:
            # Compute delta removal cost
            U.deltaRemoval = (
                self.inst.distances[U.prev.idx][U.next.idx]
                - self.inst.distances[U.prev.idx][U.idx]
                - self.inst.distances[U.idx][U.next.idx]
            )

            if R2.last_modified > self.best_inserts[R2.idx][U.idx].last_calculated:
                self.best_inserts[R2.idx][U.idx].reset()
                self.best_inserts[R2.idx][U.idx].last_calculated = self.num_moves

                initial_cost = (
                    self.inst.distances[0][U.idx]
                    + self.inst.distances[U.idx][R2.depot.next.idx]
                    - self.inst.distances[0][R2.depot.next.idx]
                )
                self.best_inserts[R2.idx][U.idx].bestCost[0] = initial_cost
                self.best_inserts[R2.idx][U.idx].bestLocation[0] = R2.depot

                V = R2.depot.next
                while not V.is_depot:
                    deltaCost = (
                        self.inst.distances[V.idx][U.idx]
                        + self.inst.distances[U.idx][V.next.idx]
                        - self.inst.distances[V.idx][V.next.idx]
                    )
                    self.best_inserts[R2.idx][U.idx].compareAndAdd(deltaCost, V)
                    V = V.next

            U = U.next

    # }}}

    # {{{ getCheapestInsertSimultRemoval
    def getCheapestInsertSimultRemoval(self, U, V):
        myBestInsert = self.best_inserts[V.route.idx][U.idx]
        found = False
        bestPosition = myBestInsert.bestLocation[0]
        bestCost = myBestInsert.bestCost[0]

        # Check if the best position isn't adjacent to V
        if bestPosition != V and bestPosition.next != V:
            found = True
        elif myBestInsert.bestLocation[1] is not None:
            bestPosition = myBestInsert.bestLocation[1]
            bestCost = myBestInsert.bestCost[1]
            if bestPosition != V and bestPosition.next != V:
                found = True
            elif myBestInsert.bestLocation[2] is not None:
                bestPosition = myBestInsert.bestLocation[2]
                bestCost = myBestInsert.bestCost[2]
                found = True

        # Evaluate inserting in place of V
        deltaCost = (
            self.inst.distances[V.prev.idx][U.idx]
            + self.inst.distances[U.idx][V.next.idx]
            - self.inst.distances[V.prev.idx][V.next.idx]
        )

        if not found or deltaCost < bestCost:
            bestPosition = V.prev
            bestCost = deltaCost

        return bestCost, bestPosition

    # }}}

    # {{{ swapStar
    def swapStar(self):
        myBestSwapStar = SwapStarElement()

        # Preprocess insertion costs
        self.preprocessInsertions(self.route_u, self.route_v)
        self.preprocessInsertions(self.route_v, self.route_u)

        # Try all combinations for SWAP*
        node_u = self.route_u.depot.next
        while not node_u.is_depot:
            node_v = self.route_v.depot.next
            while not node_v.is_depot:
                deltaPenRouteU = (
                    penaltyExcessLoad(
                        self.route_u.load
                        + self.inst.customers[node_v.idx].demand
                        - self.inst.customers[node_u.idx].demand
                    )
                    - self.route_u.penalty
                )
                deltaPenRouteV = (
                    penaltyExcessLoad(
                        self.route_v.load
                        + self.inst.customers[node_u.idx].demand
                        - self.inst.customers[node_v.idx].demand
                    )
                    - self.route_v.penalty
                )

                if (
                    deltaPenRouteU
                    + node_u.deltaRemoval
                    + deltaPenRouteV
                    + node_v.deltaRemoval
                    <= 0
                ):
                    mySwapStar = SwapStarElement()
                    mySwapStar.U = node_u
                    mySwapStar.V = node_v

                    extraV, mySwapStar.bestPositionU = (
                        self.getCheapestInsertSimultRemoval(node_u, node_v)
                    )
                    extraU, mySwapStar.bestPositionV = (
                        self.getCheapestInsertSimultRemoval(node_v, node_u)
                    )

                    mySwapStar.moveCost = (
                        deltaPenRouteU
                        + node_u.deltaRemoval
                        + extraU
                        + deltaPenRouteV
                        + node_v.deltaRemoval
                        + extraV
                    )

                    if mySwapStar.moveCost < myBestSwapStar.moveCost:
                        myBestSwapStar = mySwapStar

                node_v = node_v.next
            node_u = node_u.next

        # Try RELOCATE from routeU to routeV
        node_u = self.route_u.depot.next
        while not node_u.is_depot:
            mySwapStar = SwapStarElement()
            mySwapStar.U = node_u
            mySwapStar.bestPositionU = self.best_inserts[self.route_v.idx][
                node_u.idx
            ].bestLocation[0]
            deltaDistRouteU = (
                self.inst.distances[node_u.prev.idx][node_u.next.idx]
                - self.inst.distances[node_u.prev.idx][node_u.idx]
                - self.inst.distances[node_u.idx][node_u.next.idx]
            )
            deltaDistRouteV = self.best_inserts[self.route_v.idx][node_u.idx].bestCost[
                0
            ]

            mySwapStar.moveCost = (
                deltaDistRouteU
                + deltaDistRouteV
                + penaltyExcessLoad(
                    self.route_u.load - self.inst.customers[node_u.idx].demand
                )
                - self.route_u.penalty
                + penaltyExcessLoad(
                    self.route_v.load + self.inst.customers[node_u.idx].demand
                )
                - self.route_v.penalty
            )

            if mySwapStar.moveCost < myBestSwapStar.moveCost:
                myBestSwapStar = mySwapStar

            node_u = node_u.next

        # Try RELOCATE from routeV to routeU
        node_v = self.route_v.depot.next
        while not node_v.is_depot:
            mySwapStar = SwapStarElement()
            mySwapStar.V = node_v
            mySwapStar.bestPositionV = self.best_inserts[self.route_u.idx][
                node_v.idx
            ].bestLocation[0]
            deltaDistRouteU = self.best_inserts[self.route_u.idx][node_v.idx].bestCost[
                0
            ]
            deltaDistRouteV = (
                self.inst.distances[node_v.prev.idx][node_v.next.idx]
                - self.inst.distances[node_v.prev.idx][node_v.idx]
                - self.inst.distances[node_v.idx][node_v.next.idx]
            )

            mySwapStar.moveCost = (
                deltaDistRouteU
                + deltaDistRouteV
                + penaltyExcessLoad(
                    self.route_u.load + self.inst.customers[node_v.idx].demand
                )
                - self.route_u.penalty
                + penaltyExcessLoad(
                    self.route_v.load - self.inst.customers[node_v.idx].demand
                )
                - self.route_v.penalty
            )

            if mySwapStar.moveCost < myBestSwapStar.moveCost:
                myBestSwapStar = mySwapStar

            node_v = node_v.next

        if myBestSwapStar.moveCost > -1e-3:
            return False

        if myBestSwapStar.bestPositionU is not None:
            self.insertNode(myBestSwapStar.U, myBestSwapStar.bestPositionU)
        if myBestSwapStar.bestPositionV is not None:
            self.insertNode(myBestSwapStar.V, myBestSwapStar.bestPositionV)

        self.num_moves += 1
        self.search_completed = False
        self.updateRouteData(self.route_u)
        self.updateRouteData(self.route_v)

        return True

    # }}}

    # {{{ M1
    def move1(self):
        costSuppU = (
            self.inst.distances[self.node_u.prev.idx][self.node_x.idx]
            - self.inst.distances[self.node_u.prev.idx][self.node_u.idx]
            - self.inst.distances[self.node_u.idx][self.node_x.idx]
        )

        costSuppV = (
            self.inst.distances[self.node_v.idx][self.node_u.idx]
            + self.inst.distances[self.node_u.idx][self.node_y.idx]
            - self.inst.distances[self.node_v.idx][self.node_y.idx]
        )

        if not self.intraroute:
            # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional load constraints
            if costSuppU + costSuppV >= self.route_u.penalty + self.route_v.penalty:
                return False

            costSuppU += (
                +self.penaltyExcessLoad(self.route_u.load - self.loud_u)
                - self.route_u.penalty
            )

            costSuppV += (
                +self.penaltyExcessLoad(self.route_v.load + self.loud_u)
                - self.route_v.penalty
            )

        if costSuppU + costSuppV > -1e-3:
            return False

        if self.node_u.idx == self.node_y.idx:
            return False

        self.insertNode(self.node_u, self.node_v)
        self.num_moves += 1  # Increment move counter before updating route data
        self.search_completed = False
        self.updateRouteData(self.route_u)

        if not self.intraroute:
            self.updateRouteData(self.route_v)

        return True

    # }}}

    # {{{ M2
    def move2(self):
        costSuppU = (
            self.inst.distances[self.node_u.prev.idx][self.node_x.next.idx]
            - self.inst.distances[self.node_u.prev.idx][self.node_u.idx]
            - self.inst.distances[self.node_x.idx][self.node_x.next.idx]
        )

        costSuppV = (
            self.inst.distances[self.node_v.idx][self.node_u.idx]
            + self.inst.distances[self.node_x.idx][self.node_y.idx]
            - self.inst.distances[self.node_v.idx][self.node_y.idx]
        )

        if not self.intraroute:
            # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load) constraints
            if costSuppU + costSuppV >= self.route_u.penalty + self.route_v.penalty:
                return False

            costSuppU += (
                self.penaltyExcessLoad(self.route_u.load - self.loud_u - self.load_x)
                - self.route_u.penalty
            )

            costSuppV += (
                self.penaltyExcessLoad(self.route_v.load + self.loud_u + self.load_x)
                - self.route_v.penalty
            )

        if costSuppU + costSuppV > -1e-3:
            return False

        if (
            self.node_u == self.node_y
            or self.node_v == self.node_x
            or self.node_x.is_depot
        ):
            return False

        self.insertNode(self.node_u, self.node_v)
        self.insertNode(self.node_x, self.node_u)

        self.num_moves += 1  # Increment move counter before updating route data
        self.search_completed = False
        self.updateRouteData(self.route_u)

        if not self.intraroute:
            self.updateRouteData(self.route_v)

        return True

    # }}}

    # {{{ M3
    def move3(self):
        costSuppU = (
            self.inst.distances[self.node_u.prev.idx][self.node_x.next.idx]
            - self.inst.distances[self.node_u.prev.idx][self.node_u.idx]
            - self.inst.distances[self.node_u.idx][self.node_x.idx]
            - self.inst.distances[self.node_x.idx][self.node_x.next.idx]
        )

        costSuppV = (
            self.inst.distances[self.node_v.idx][self.node_x.idx]
            + self.inst.distances[self.node_x.idx][self.node_u.idx]
            + self.inst.distances[self.node_u.idx][self.node_y.idx]
            - self.inst.distances[self.node_v.idx][self.node_y.idx]
        )

        if not self.intraroute:
            # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking load constraints
            if costSuppU + costSuppV >= self.route_u.penalty + self.route_v.penalty:
                return False

            costSuppU += (
                self.penaltyExcessLoad(self.route_u.load - self.loud_u - self.load_x)
                - self.route_u.penalty
            )

            costSuppV += (
                self.penaltyExcessLoad(self.route_v.load + self.loud_u + self.load_x)
                - self.route_v.penalty
            )

        if costSuppU + costSuppV > -1e-3:
            return False

        if (
            self.node_u == self.node_y
            or self.node_x == self.node_v
            or self.node_x.is_depot
        ):
            return False

        self.insertNode(self.node_x, self.node_v)
        self.insertNode(self.node_u, self.node_x)

        self.num_moves += 1  # Increment move counter before updating route data
        self.search_completed = False
        self.updateRouteData(self.route_u)

        if not self.intraroute:
            self.updateRouteData(self.route_v)

        return True

    # }}}

    # {{{ M4
    def move4(self):
        costSuppU = (
            self.inst.distances[self.node_u.prev.idx][self.node_v.idx]
            + self.inst.distances[self.node_v.idx][self.node_x.idx]
            - self.inst.distances[self.node_u.prev.idx][self.node_u.idx]
            - self.inst.distances[self.node_u.idx][self.node_x.idx]
        )

        costSuppV = (
            self.inst.distances[self.node_v.prev.idx][self.node_u.idx]
            + self.inst.distances[self.node_u.idx][self.node_y.idx]
            - self.inst.distances[self.node_v.prev.idx][self.node_v.idx]
            - self.inst.distances[self.node_v.idx][self.node_y.idx]
        )

        if not self.intraroute:
            # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking load constraints
            if costSuppU + costSuppV >= self.route_u.penalty + self.route_v.penalty:
                return False

            costSuppU += (
                self.penaltyExcessLoad(self.route_u.load + self.load_v - self.loud_u)
                - self.route_u.penalty
            )

            costSuppV += (
                self.penaltyExcessLoad(self.route_v.load + self.loud_u - self.load_v)
                - self.route_v.penalty
            )

        if costSuppU + costSuppV > -1e-3:
            return False

        if (
            self.node_u.idx == self.node_v.prev.idx
            or self.node_u.idx == self.node_y.idx
        ):
            return False

        self.swapNode(self.node_u, self.node_v)

        self.num_moves += 1  # Increment move counter before updating route data
        self.search_completed = False
        self.updateRouteData(self.route_u)

        if not self.intraroute:
            self.updateRouteData(self.route_v)

        return True

    # }}}

    # {{{ M5
    def move5(self):
        costSuppU = (
            self.inst.distances[self.node_u.prev.idx][self.node_v.idx]
            + self.inst.distances[self.node_v.idx][self.node_x.next.idx]
            - self.inst.distances[self.node_u.prev.idx][self.node_u.idx]
            - self.inst.distances[self.node_x.idx][self.node_x.next.idx]
        )

        costSuppV = (
            self.inst.distances[self.node_v.prev.idx][self.node_u.idx]
            + self.inst.distances[self.node_x.idx][self.node_y.idx]
            - self.inst.distances[self.node_v.prev.idx][self.node_v.idx]
            - self.inst.distances[self.node_v.idx][self.node_y.idx]
        )

        if not self.intraroute:
            # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking load constraints
            if costSuppU + costSuppV >= self.route_u.penalty + self.route_v.penalty:
                return False

            costSuppU += (
                self.penaltyExcessLoad(
                    self.route_u.load + self.load_v - self.loud_u - self.load_x
                )
                - self.route_u.penalty
            )

            costSuppV += (
                self.penaltyExcessLoad(
                    self.route_v.load + self.loud_u + self.load_x - self.load_v
                )
                - self.route_v.penalty
            )

        if costSuppU + costSuppV > -1e-3:
            return False

        if (
            self.node_u == self.node_v.prev
            or self.node_x == self.node_v.prev
            or self.node_u == self.node_y
            or self.node_x.is_depot
        ):
            return False

        self.swapNode(self.node_u, self.node_v)
        self.insertNode(self.node_x, self.node_u)

        self.num_moves += 1  # Increment move counter before updating route data
        self.search_completed = False
        self.updateRouteData(self.route_u)

        if not self.intraroute:
            self.updateRouteData(self.route_v)

        return True

    # }}}

    # {{{ M6
    def move6(self):
        costSuppU = (
            self.inst.distances[self.node_u.prev.idx][self.node_v.idx]
            + self.inst.distances[self.node_y.idx][self.node_x.next.idx]
            - self.inst.distances[self.node_u.prev.idx][self.node_u.idx]
            - self.inst.distances[self.node_x.idx][self.node_x.next.idx]
        )

        costSuppV = (
            self.inst.distances[self.node_v.prev.idx][self.node_u.idx]
            + self.inst.distances[self.node_x.idx][self.node_y.next.idx]
            - self.inst.distances[self.node_v.prev.idx][self.node_v.idx]
            - self.inst.distances[self.node_y.idx][self.node_y.next.idx]
        )

        if not self.intraroute:
            # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional load constraint
            if costSuppU + costSuppV >= self.route_u.penalty + self.route_v.penalty:
                return False

            costSuppU += (
                self.penaltyExcessLoad(
                    self.route_u.load
                    + self.load_v
                    + self.load_y
                    - self.load_u
                    - self.load_x
                )
                - self.route_u.penalty
            )

            costSuppV += (
                self.penaltyExcessLoad(
                    self.route_v.load
                    + self.loud_u
                    + self.load_x
                    - self.load_v
                    - self.loadY
                )
                - self.route_v.penalty
            )

        if costSuppU + costSuppV > -1e-3:
            return False

        if (
            self.node_x.is_depot
            or self.node_y.is_depot
            or self.node_y == self.node_u.prev
            or self.node_u == self.node_y
            or self.node_x == self.node_v
            or self.node_v == self.node_x.next
        ):
            return False

        self.swapNode(self.node_u, self.node_v)
        self.swapNode(self.node_x, self.node_y)

        self.num_moves += 1  # Increment move counter before updating route data
        self.search_completed = False
        self.updateRouteData(self.route_u)

        if not self.intraroute:
            self.updateRouteData(self.route_v)

        return True

    # }}}

    # {{{ M7
    def move7(self):
        if self.node_u.position > self.node_v.position:
            return False

        cost = (
            self.inst.distances[self.node_u.idx][self.node_v.idx]
            + self.inst.distances[self.node_x.idx][self.node_y.idx]
            - self.inst.distances[self.node_u.idx][self.node_x.idx]
            - self.inst.distances[self.node_v.idx][self.node_y.idx]
            + self.node_v.cumulatedReversalDistance
            - self.node_x.cumulatedReversalDistance
        )

        if cost > -1e-3:
            return False

        if self.node_u.next == self.node_v:
            return False

        nodeNum = self.node_x.next
        self.node_x.prev = nodeNum
        self.node_x.next = self.node_y

        while nodeNum != self.node_v:
            temp = nodeNum.next
            nodeNum.next = nodeNum.prev
            nodeNum.prev = temp
            nodeNum = temp

        self.node_v.next = self.node_v.prev
        self.node_v.prev = self.node_u
        self.node_u.next = self.node_v
        self.node_y.prev = self.node_x

        self.num_moves += 1  # Increment move counter before updating route data
        self.search_completed = False
        self.updateRouteData(self.route_u)
        return True

    # }}}

    # {{{ M8
    def move8(self):
        cost = (
            self.inst.distances[self.node_u.idx][self.node_v.idx]
            + self.inst.distances[self.node_x.idx][self.node_y.idx]
            - self.inst.distances[self.node_u.idx][self.node_x.idx]
            - self.inst.distances[self.node_v.idx][self.node_y.idx]
            + self.node_v.cumulatedReversalDistance
            + self.route_u.reversalDistance
            - self.node_x.cumulatedReversalDistance
            - self.route_u.penalty
            - self.route_v.penalty
        )

        # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking load constraints
        if cost >= 0:
            return False

        cost += self.penaltyExcessLoad(
            self.node_u.cum_load + self.node_v.cum_load
        ) + self.penaltyExcessLoad(
            self.route_u.load
            + self.route_v.load
            - self.node_u.cum_load
            - self.node_v.cum_load
        )

        if cost > -1e-3:
            return False

        depot_u = self.route_u.depot
        depot_v = self.route_v.depot
        end_depot_u = self.route_u.depot.prev
        end_depot_v = self.route_v.depot.prev
        depot_vSuiv = depot_v.next

        # Reverse node_x and set its route to routeV
        xx = self.node_x
        while not xx.is_depot:
            temp = xx.next
            xx.next = xx.prev
            xx.prev = temp
            xx.route = self.route_v
            xx = temp

        # Reverse node_v and set its route to routeU
        vv = self.node_v
        while not vv.is_depot:
            temp = vv.prev
            vv.prev = vv.next
            vv.next = temp
            vv.route = self.route_u
            vv = temp

        # Adjust node links
        self.node_u.next = self.node_v
        self.node_v.prev = self.node_u
        self.node_x.next = self.node_y
        self.node_y.prev = self.node_x

        if self.node_x.is_depot:
            end_depot_u.next = depot_u
            end_depot_u.prev = depot_vSuiv
            end_depot_u.prev.next = end_depot_u
            depot_v.next = self.node_y
            self.node_y.prev = depot_v
        elif self.node_v.is_depot:
            depot_v.next = end_depot_u.prev
            depot_v.next.prev = depot_v
            depot_v.prev = end_depot_v
            end_depot_u.prev = self.node_u
            self.node_u.next = end_depot_u
        else:
            depot_v.next = end_depot_u.prev
            depot_v.next.prev = depot_v
            end_depot_u.prev = depot_vSuiv
            end_depot_u.prev.next = end_depot_u

        self.num_moves += 1  # Increment move counter before updating route data
        self.search_completed = False
        self.updateRouteData(self.route_u)
        self.updateRouteData(self.route_v)
        return True

    # }}}

    # {{{ M9
    def move9(self):
        cost = (
            self.inst.distances[self.node_u.idx][self.node_y.idx]
            + self.inst.distances[self.node_v.idx][self.node_x.idx]
            - self.inst.distances[self.node_u.idx][self.node_x.idx]
            - self.inst.distances[self.node_v.idx][self.node_y.idx]
            - self.route_u.penalty
            - self.route_v.penalty
        )

        # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking load constraints
        if cost >= 0:
            return False

        cost += self.penaltyExcessLoad(
            self.node_u.cum_load + self.route_v.load - self.node_v.cum_load
        ) + self.penaltyExcessLoad(
            self.node_v.cum_load + self.route_u.load - self.node_u.cum_load
        )

        if cost > -1e-3:
            return False

        depot_u = self.route_u.depot
        depot_v = self.route_v.depot
        end_depot_u = depot_u.prev
        end_depot_v = depot_v.prev
        depot_u_pred = end_depot_u.prev

        # Reassign route of node_y to routeU
        count = self.node_y
        while not count.is_depot:
            count.route = self.route_u
            count = count.next

        # Reassign route of node_x to routeV
        count = self.node_x
        while not count.is_depot:
            count.route = self.route_v
            count = count.next

        # Adjust node links
        self.node_u.next = self.node_y
        self.node_y.prev = self.node_u
        self.node_v.next = self.node_x
        self.node_x.prev = self.node_v

        if self.node_x.is_depot:
            end_depot_u.prev = end_depot_v.prev
            end_depot_u.prev.next = end_depot_u
            self.node_v.next = end_depot_v
            end_depot_v.prev = self.node_v
        else:
            end_depot_u.prev = end_depot_v.prev
            end_depot_u.prev.next = end_depot_u
            end_depot_v.prev = depot_u_pred
            end_depot_v.prev.next = end_depot_v

        self.num_moves += 1  # Increment move counter before updating route data
        self.search_completed = False
        self.updateRouteData(self.route_u)
        self.updateRouteData(self.route_v)
        return True

    # }}}

    def penalty_excess_load(load):
        return max(0, load - self.inst.vehicle_capacity) * self.capacity_penalty
