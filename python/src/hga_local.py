from hga_structures import Node, Route


class LocalSearch:
    # {{{ __init__
    def __init__(self, instance):
        self.inst = instance

        self.clients = [Node() for _ in range(instance.numCustomers)]
        self.routes = [Route() for _ in range(instance.numVehicles)]
        self.depots = [Node() for _ in range(instance.numVehicles)]
        self.depotsEnd = [Node() for _ in range(instance.numVehicles)]

        # Keep track of the three best insertion positions
        # for each customer in each route
        self.bestInsertClient = [
            [ThreeBestInsert() for _ in range(instance.numCustomers)]
            for _ in range(instance.numVehicles)
        ]

        # Initialize client nodes with index and depot
        for i in range(instance.numCustomers):
            self.clients[i].idx = i
            self.clients[i].isDepot = False

        # Initialize routes and depots
        for i in range(instance.numVehicles):
            self.routes[i].idx = i
            self.routes[i].depot = self.depots[i]

            self.depots[i].idx = 0
            self.depots[i].isDepot = True
            self.depots[i].route = self.routes[i]

            self.depotsEnd[i].idx = 0
            self.depotsEnd[i].isDepot = True
            self.depotsEnd[i].route = self.routes[i]

        # Order vectors for heuristic use
        self.orderNodes = list(range(1, instance.numCustomers))
        self.orderRoutes = list(range(instance.numVehicles))

    # }}}

    # {{{ run
    def run(self, indiv, capacity_penalty):
        self.capacity_penalty = capacity_penalty
        self.loadIndividual(indiv)  # Load in the individual into the LocalSearch class

        # Shuffling the order of the nodes explored by the LS to allow for more diversity in the search
        random.shuffle(self.orderNodes)
        random.shuffle(self.orderRoutes)
        for i in range(1, self.self.inst.numCustomers + 1):
            if self.params.ran() % self.params.ap.nbGranular == 0:
                random.shuffle(self.params.correlatedVertices[i])

        self.searchCompleted = False
        self.loopID = 0

        while not self.searchCompleted:
            if (
                self.loopID > 1
            ):  # Allows at least two loops since some moves involving empty routes are not checked at the first loop
                self.searchCompleted = True

            # CLASSICAL ROUTE IMPROVEMENT (RI) MOVES SUBJECT TO A PROXIMITY RESTRICTION
            for posU in range(self.self.inst.numCustomers):
                self.nodeU = self.clients[self.orderNodes[posU]]
                lastTestRINodeU = self.nodeU.whenLastTestedRI
                self.nodeU.whenLastTestedRI = self.numMoves

                for posV in range(len(self.params.correlatedVertices[self.nodeU.idx])):
                    self.nodeV = self.clients[
                        self.params.correlatedVertices[self.nodeU.idx][posV]
                    ]

                    if (
                        self.loopID == 0
                        or max(
                            self.nodeU.route.whenLastModified,
                            self.nodeV.route.whenLastModified,
                        )
                        > lastTestRINodeU
                    ):
                        # Only evaluate moves involving routes that have been modified since last move evaluations for nodeU
                        self.setLocalVariablesRouteU()
                        self.setLocalVariablesRouteV()

                        if self.move1():
                            continue  # RELOCATE
                        if self.move2():
                            continue  # RELOCATE
                        if self.move3():
                            continue  # RELOCATE
                        if self.nodeUIndex <= self.nodeVIndex and self.move4():
                            continue  # SWAP
                        if self.move5():
                            continue  # SWAP
                        if self.nodeUIndex <= self.nodeVIndex and self.move6():
                            continue  # SWAP
                        if self.intraRouteMove and self.move7():
                            continue  # 2-OPT
                        if not self.intraRouteMove and self.move8():
                            continue  # 2-OPT*
                        if not self.intraRouteMove and self.move9():
                            continue  # 2-OPT*

                        # Trying moves that insert nodeU directly after the depot
                        if self.nodeV.prev.isDepot:
                            self.nodeV = self.nodeV.prev
                            self.setLocalVariablesRouteV()
                            if self.move1():
                                continue  # RELOCATE
                            if self.move2():
                                continue  # RELOCATE
                            if self.move3():
                                continue  # RELOCATE
                            if not self.intraRouteMove and self.move8():
                                continue  # 2-OPT*
                            if not self.intraRouteMove and self.move9():
                                continue  # 2-OPT*

                # MOVES INVOLVING AN EMPTY ROUTE -- NOT TESTED IN THE FIRST LOOP TO AVOID INCREASING TOO MUCH THE FLEET SIZE
                if self.loopID > 0 and self.emptyRoutes:
                    self.nodeV = self.routes[self.emptyRoutes.pop(0)].depot
                    self.setLocalVariablesRouteU()
                    self.setLocalVariablesRouteV()
                    if self.move1():
                        continue  # RELOCATE
                    if self.move2():
                        continue  # RELOCATE
                    if self.move3():
                        continue  # RELOCATE
                    if self.move9():
                        continue  # 2-OPT*

            if self.params.ap.useSwapStar == 1 and self.params.areCoordinatesProvided:
                # (SWAP*) MOVES LIMITED TO ROUTE PAIRS WHOSE CIRCLE SECTORS OVERLAP
                for rU in range(self.self.inst.numVehicles):
                    self.routeU = self.routes[self.orderRoutes[rU]]
                    lastTestSWAPStarRouteU = self.routeU.whenLastTestedSWAPStar
                    self.routeU.whenLastTestedSWAPStar = self.numMoves
                    for rV in range(self.self.inst.numVehicles):
                        self.routeV = self.routes[self.orderRoutes[rV]]
                        if (
                            self.routeU.nbCustomers > 0
                            and self.routeV.nbCustomers > 0
                            and self.routeU.idx < self.routeV.idx
                        ):
                            if (
                                self.loopID == 0
                                or max(
                                    self.routeU.whenLastModified,
                                    self.routeV.whenLastModified,
                                )
                                > lastTestSWAPStarRouteU
                            ):
                                if CircleSector.overlap(
                                    self.routeU.sector, self.routeV.sector
                                ):
                                    self.swapStar()

            self.loopID += 1

        # Register the solution produced by the LS in the individual
        self.exportIndividual(indiv)

    # }}}

    # {{{ variable aliasing
    def setLocalVariablesRouteU(self):
        self.routeU = self.nodeU.route
        self.nodeX = self.nodeU.next
        self.nodeXNextIndex = self.nodeX.next.idx
        self.nodeUIndex = self.nodeU.idx
        self.nodeUPrevIndex = self.nodeU.prev.idx
        self.nodeXIndex = self.nodeX.idx
        self.loadU = self.params.cli[self.nodeUIndex].demand
        self.serviceU = self.params.cli[self.nodeUIndex].serviceDuration
        self.loadX = self.params.cli[self.nodeXIndex].demand
        self.serviceX = self.params.cli[self.nodeXIndex].serviceDuration

    def setLocalVariablesRouteV(self):
        self.routeV = self.nodeV.route
        self.nodeY = self.nodeV.next
        self.nodeYNextIndex = self.nodeY.next.idx
        self.nodeVIndex = self.nodeV.idx
        self.nodeVPrevIndex = self.nodeV.prev.idx
        self.nodeYIndex = self.nodeY.idx
        self.loadV = self.params.cli[self.nodeVIndex].demand
        self.serviceV = self.params.cli[self.nodeVIndex].serviceDuration
        self.loadY = self.params.cli[self.nodeYIndex].demand
        self.serviceY = self.params.cli[self.nodeYIndex].serviceDuration
        self.intraRouteMove = self.routeU == self.routeV

    # }}}

    # {{{ exportIndividual
    def exportIndividual(self, indiv):
        # Create a list of (polar angle, route index) tuples
        routePolarAngles = [
            (self.routes[r].polarAngleBarycenter, r)
            for r in range(self.self.inst.numVehicles)
        ]

        # Sort routes by polar angle (empty routes have 1e30, so they go to the end)
        routePolarAngles.sort()

        pos = 0
        for _, route_index in routePolarAngles:
            indiv.chromR[route_index].clear()
            node = self.depots[route_index].next
            while not node.isDepot:
                indiv.chromT[pos] = node.idx
                indiv.chromR[route_index].append(node.idx)
                node = node.next
                pos += 1

        # Evaluate total cost using problem parameters
        indiv.evaluateCompleteCost(self.params)

    # }}}

    # {{{ loadIndividual
    def loadIndividual(self, indiv):
        self.emptyRoutes.clear()
        self.numMoves = 0

        for r in range(self.self.inst.numVehicles):
            myDepot = self.depots[r]
            myDepotFin = self.depotsEnd[r]
            myRoute = self.routes[r]

            # Connect the depot start and end
            myDepot.prev = myDepotFin
            myDepotFin.next = myDepot

            if indiv.chromR[r]:  # Route is not empty
                myClient = self.clients[indiv.chromR[r][0]]
                myClient.route = myRoute
                myClient.prev = myDepot
                myDepot.next = myClient

                for idx in indiv.chromR[r][1:]:
                    myClientPred = myClient
                    myClient = self.clients[idx]
                    myClient.prev = myClientPred
                    myClientPred.next = myClient
                    myClient.route = myRoute

                myClient.next = myDepotFin
                myDepotFin.prev = myClient
            else:
                # Route is empty, just link depot to depotEnd
                myDepot.next = myDepotFin
                myDepotFin.prev = myDepot

            self.updateRouteData(myRoute)
            myRoute.whenLastTestedSWAPStar = -1

            # Reset insertion memory for this route
            for i in range(1, self.self.inst.numCustomers + 1):
                self.bestInsertClient[r][i].whenLastCalculated = -1

        # Reset RI test memory for all clients
        for i in range(1, self.self.inst.numCustomers + 1):
            self.clients[i].whenLastTestedRI = -1

    # }}}

    # {{{ updateRouteData
    def updateRouteData(self, myRoute):
        """
        Recompute route metrics and metadata after making a move
        """

        myplace = 0
        myload = 0
        mytime = 0
        myReversalDistance = 0
        cumulatedX = 0
        cumulatedY = 0

        mynode = myRoute.depot
        mynode.position = 0
        mynode.cumulatedLoad = 0
        mynode.cumulatedTime = 0
        mynode.cumulatedReversalDistance = 0

        do = True
        while do or (not mynode.isDepot):
            # Loop through each node in myRoute
            mynode = mynode.next
            myplace += 1
            mynode.position = myplace

            cli = self.params.cli[mynode.idx]
            myload += cli.demand
            prevIdx = mynode.prev.idx
            mytime += self.params.timeCost[prevIdx][mynode.idx] + cli.serviceDuration
            myReversalDistance += (
                self.params.timeCost[mynode.idx][prevIdx]
                - self.params.timeCost[prevIdx][mynode.idx]
            )

            mynode.cumulatedLoad = myload
            mynode.cumulatedTime = mytime
            mynode.cumulatedReversalDistance = myReversalDistance

            if not mynode.isDepot:
                cumulatedX += cli.coordX
                cumulatedY += cli.coordY
                if firstIt:
                    myRoute.sector.initialize(cli.polarAngle)
                else:
                    myRoute.sector.extend(cli.polarAngle)

            firstIt = False

        myRoute.duration = mytime
        myRoute.load = myload
        myRoute.penalty = self.penaltyExcessDuration(mytime) + self.penaltyExcessLoad(
            myload
        )
        myRoute.nbCustomers = myplace - 1
        myRoute.reversalDistance = myReversalDistance
        myRoute.whenLastModified = self.numMoves

        if myRoute.nbCustomers == 0:
            myRoute.polarAngleBarycenter = 1e30
            self.emptyRoutes.add(myRoute.idx)
        else:
            avgX = cumulatedX / myRoute.nbCustomers
            avgY = cumulatedY / myRoute.nbCustomers
            depotX = self.params.cli[0].coordX
            depotY = self.params.cli[0].coordY
            myRoute.polarAngleBarycenter = math.atan2(avgY - depotY, avgX - depotX)
            self.emptyRoutes.discard(myRoute.idx)

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
        while not U.isDepot:
            # Compute delta removal cost
            U.deltaRemoval = (
                self.params.timeCost[U.prev.idx][U.next.idx]
                - self.params.timeCost[U.prev.idx][U.idx]
                - self.params.timeCost[U.idx][U.next.idx]
            )

            if (
                R2.whenLastModified
                > self.bestInsertClient[R2.idx][U.idx].whenLastCalculated
            ):
                self.bestInsertClient[R2.idx][U.idx].reset()
                self.bestInsertClient[R2.idx][U.idx].whenLastCalculated = self.numMoves

                initial_cost = (
                    self.params.timeCost[0][U.idx]
                    + self.params.timeCost[U.idx][R2.depot.next.idx]
                    - self.params.timeCost[0][R2.depot.next.idx]
                )
                self.bestInsertClient[R2.idx][U.idx].bestCost[0] = initial_cost
                self.bestInsertClient[R2.idx][U.idx].bestLocation[0] = R2.depot

                V = R2.depot.next
                while not V.isDepot:
                    deltaCost = (
                        self.params.timeCost[V.idx][U.idx]
                        + self.params.timeCost[U.idx][V.next.idx]
                        - self.params.timeCost[V.idx][V.next.idx]
                    )
                    self.bestInsertClient[R2.idx][U.idx].compareAndAdd(deltaCost, V)
                    V = V.next

            U = U.next

    # }}}

    # {{{ getCheapestInsertSimultRemoval
    def getCheapestInsertSimultRemoval(self, U, V):
        myBestInsert = self.bestInsertClient[V.route.idx][U.idx]
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
            self.params.timeCost[V.prev.idx][U.idx]
            + self.params.timeCost[U.idx][V.next.idx]
            - self.params.timeCost[V.prev.idx][V.next.idx]
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
        self.preprocessInsertions(self.routeU, self.routeV)
        self.preprocessInsertions(self.routeV, self.routeU)

        # Try all combinations for SWAP*
        nodeU = self.routeU.depot.next
        while not nodeU.isDepot:
            nodeV = self.routeV.depot.next
            while not nodeV.isDepot:
                deltaPenRouteU = (
                    penaltyExcessLoad(
                        self.routeU.load
                        + self.params.cli[nodeV.idx].demand
                        - self.params.cli[nodeU.idx].demand
                    )
                    - self.routeU.penalty
                )
                deltaPenRouteV = (
                    penaltyExcessLoad(
                        self.routeV.load
                        + self.params.cli[nodeU.idx].demand
                        - self.params.cli[nodeV.idx].demand
                    )
                    - self.routeV.penalty
                )

                if (
                    deltaPenRouteU
                    + nodeU.deltaRemoval
                    + deltaPenRouteV
                    + nodeV.deltaRemoval
                    <= 0
                ):
                    mySwapStar = SwapStarElement()
                    mySwapStar.U = nodeU
                    mySwapStar.V = nodeV

                    extraV, mySwapStar.bestPositionU = (
                        self.getCheapestInsertSimultRemoval(nodeU, nodeV)
                    )
                    extraU, mySwapStar.bestPositionV = (
                        self.getCheapestInsertSimultRemoval(nodeV, nodeU)
                    )

                    mySwapStar.moveCost = (
                        deltaPenRouteU
                        + nodeU.deltaRemoval
                        + extraU
                        + deltaPenRouteV
                        + nodeV.deltaRemoval
                        + extraV
                        + penaltyExcessDuration(
                            self.routeU.duration
                            + nodeU.deltaRemoval
                            + extraU
                            + self.params.cli[nodeV.idx].serviceDuration
                            - self.params.cli[nodeU.idx].serviceDuration
                        )
                        + penaltyExcessDuration(
                            self.routeV.duration
                            + nodeV.deltaRemoval
                            + extraV
                            - self.params.cli[nodeV.idx].serviceDuration
                            + self.params.cli[nodeU.idx].serviceDuration
                        )
                    )

                    if mySwapStar.moveCost < myBestSwapStar.moveCost:
                        myBestSwapStar = mySwapStar

                nodeV = nodeV.next
            nodeU = nodeU.next

        # Try RELOCATE from routeU to routeV
        nodeU = self.routeU.depot.next
        while not nodeU.isDepot:
            mySwapStar = SwapStarElement()
            mySwapStar.U = nodeU
            mySwapStar.bestPositionU = self.bestInsertClient[self.routeV.idx][
                nodeU.idx
            ].bestLocation[0]
            deltaDistRouteU = (
                self.params.timeCost[nodeU.prev.idx][nodeU.next.idx]
                - self.params.timeCost[nodeU.prev.idx][nodeU.idx]
                - self.params.timeCost[nodeU.idx][nodeU.next.idx]
            )
            deltaDistRouteV = self.bestInsertClient[self.routeV.idx][
                nodeU.idx
            ].bestCost[0]

            mySwapStar.moveCost = (
                deltaDistRouteU
                + deltaDistRouteV
                + penaltyExcessLoad(
                    self.routeU.load - self.params.cli[nodeU.idx].demand
                )
                - self.routeU.penalty
                + penaltyExcessLoad(
                    self.routeV.load + self.params.cli[nodeU.idx].demand
                )
                - self.routeV.penalty
                + penaltyExcessDuration(
                    self.routeU.duration
                    + deltaDistRouteU
                    - self.params.cli[nodeU.idx].serviceDuration
                )
                + penaltyExcessDuration(
                    self.routeV.duration
                    + deltaDistRouteV
                    + self.params.cli[nodeU.idx].serviceDuration
                )
            )

            if mySwapStar.moveCost < myBestSwapStar.moveCost:
                myBestSwapStar = mySwapStar

            nodeU = nodeU.next

        # Try RELOCATE from routeV to routeU
        nodeV = self.routeV.depot.next
        while not nodeV.isDepot:
            mySwapStar = SwapStarElement()
            mySwapStar.V = nodeV
            mySwapStar.bestPositionV = self.bestInsertClient[self.routeU.idx][
                nodeV.idx
            ].bestLocation[0]
            deltaDistRouteU = self.bestInsertClient[self.routeU.idx][
                nodeV.idx
            ].bestCost[0]
            deltaDistRouteV = (
                self.params.timeCost[nodeV.prev.idx][nodeV.next.idx]
                - self.params.timeCost[nodeV.prev.idx][nodeV.idx]
                - self.params.timeCost[nodeV.idx][nodeV.next.idx]
            )

            mySwapStar.moveCost = (
                deltaDistRouteU
                + deltaDistRouteV
                + penaltyExcessLoad(
                    self.routeU.load + self.params.cli[nodeV.idx].demand
                )
                - self.routeU.penalty
                + penaltyExcessLoad(
                    self.routeV.load - self.params.cli[nodeV.idx].demand
                )
                - self.routeV.penalty
                + penaltyExcessDuration(
                    self.routeU.duration
                    + deltaDistRouteU
                    + self.params.cli[nodeV.idx].serviceDuration
                )
                + penaltyExcessDuration(
                    self.routeV.duration
                    + deltaDistRouteV
                    - self.params.cli[nodeV.idx].serviceDuration
                )
            )

            if mySwapStar.moveCost < myBestSwapStar.moveCost:
                myBestSwapStar = mySwapStar

            nodeV = nodeV.next

        if myBestSwapStar.moveCost > -MY_EPSILON:
            return False

        if myBestSwapStar.bestPositionU is not None:
            self.insertNode(myBestSwapStar.U, myBestSwapStar.bestPositionU)
        if myBestSwapStar.bestPositionV is not None:
            self.insertNode(myBestSwapStar.V, myBestSwapStar.bestPositionV)

        self.numMoves += 1
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        self.updateRouteData(self.routeV)

        return True

    # }}}

    # {{{ M1
    def move1(self):
        costSuppU = (
            self.params.timeCost[self.nodeUPrevIndex][self.nodeXIndex]
            - self.params.timeCost[self.nodeUPrevIndex][self.nodeUIndex]
            - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
        )

        costSuppV = (
            self.params.timeCost[self.nodeVIndex][self.nodeUIndex]
            + self.params.timeCost[self.nodeUIndex][self.nodeYIndex]
            - self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
        )

        if not self.intraRouteMove:
            # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
            if costSuppU + costSuppV >= self.routeU.penalty + self.routeV.penalty:
                return False

            costSuppU += (
                self.penaltyExcessDuration(
                    self.routeU.duration + costSuppU - self.serviceU
                )
                + self.penaltyExcessLoad(self.routeU.load - self.loadU)
                - self.routeU.penalty
            )

            costSuppV += (
                self.penaltyExcessDuration(
                    self.routeV.duration + costSuppV + self.serviceU
                )
                + self.penaltyExcessLoad(self.routeV.load + self.loadU)
                - self.routeV.penalty
            )

        if costSuppU + costSuppV > -self.MY_EPSILON:
            return False

        if self.nodeUIndex == self.nodeYIndex:
            return False

        self.insertNode(self.nodeU, self.nodeV)
        self.numMoves += 1  # Increment move counter before updating route data
        self.searchCompleted = False
        self.updateRouteData(self.routeU)

        if not self.intraRouteMove:
            self.updateRouteData(self.routeV)

        return True

    # }}}

    # {{{ M2
    def move2(self):
        costSuppU = (
            self.params.timeCost[self.nodeUPrevIndex][self.nodeXNextIndex]
            - self.params.timeCost[self.nodeUPrevIndex][self.nodeUIndex]
            - self.params.timeCost[self.nodeXIndex][self.nodeXNextIndex]
        )

        costSuppV = (
            self.params.timeCost[self.nodeVIndex][self.nodeUIndex]
            + self.params.timeCost[self.nodeXIndex][self.nodeYIndex]
            - self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
        )

        if not self.intraRouteMove:
            # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
            if costSuppU + costSuppV >= self.routeU.penalty + self.routeV.penalty:
                return False

            costSuppU += (
                self.penaltyExcessDuration(
                    self.routeU.duration
                    + costSuppU
                    - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
                    - self.serviceU
                    - self.serviceX
                )
                + self.penaltyExcessLoad(self.routeU.load - self.loadU - self.loadX)
                - self.routeU.penalty
            )

            costSuppV += (
                self.penaltyExcessDuration(
                    self.routeV.duration
                    + costSuppV
                    + self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
                    + self.serviceU
                    + self.serviceX
                )
                + self.penaltyExcessLoad(self.routeV.load + self.loadU + self.loadX)
                - self.routeV.penalty
            )

        if costSuppU + costSuppV > -self.MY_EPSILON:
            return False

        if self.nodeU == self.nodeY or self.nodeV == self.nodeX or self.nodeX.isDepot:
            return False

        self.insertNode(self.nodeU, self.nodeV)
        self.insertNode(self.nodeX, self.nodeU)

        self.numMoves += 1  # Increment move counter before updating route data
        self.searchCompleted = False
        self.updateRouteData(self.routeU)

        if not self.intraRouteMove:
            self.updateRouteData(self.routeV)

        return True

    # }}}

    # {{{ M3
    def move3(self):
        costSuppU = (
            self.params.timeCost[self.nodeUPrevIndex][self.nodeXNextIndex]
            - self.params.timeCost[self.nodeUPrevIndex][self.nodeUIndex]
            - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
            - self.params.timeCost[self.nodeXIndex][self.nodeXNextIndex]
        )

        costSuppV = (
            self.params.timeCost[self.nodeVIndex][self.nodeXIndex]
            + self.params.timeCost[self.nodeXIndex][self.nodeUIndex]
            + self.params.timeCost[self.nodeUIndex][self.nodeYIndex]
            - self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
        )

        if not self.intraRouteMove:
            # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
            if costSuppU + costSuppV >= self.routeU.penalty + self.routeV.penalty:
                return False

            costSuppU += (
                self.penaltyExcessDuration(
                    self.routeU.duration + costSuppU - self.serviceU - self.serviceX
                )
                + self.penaltyExcessLoad(self.routeU.load - self.loadU - self.loadX)
                - self.routeU.penalty
            )

            costSuppV += (
                self.penaltyExcessDuration(
                    self.routeV.duration + costSuppV + self.serviceU + self.serviceX
                )
                + self.penaltyExcessLoad(self.routeV.load + self.loadU + self.loadX)
                - self.routeV.penalty
            )

        if costSuppU + costSuppV > -self.MY_EPSILON:
            return False

        if self.nodeU == self.nodeY or self.nodeX == self.nodeV or self.nodeX.isDepot:
            return False

        self.insertNode(self.nodeX, self.nodeV)
        self.insertNode(self.nodeU, self.nodeX)

        self.numMoves += 1  # Increment move counter before updating route data
        self.searchCompleted = False
        self.updateRouteData(self.routeU)

        if not self.intraRouteMove:
            self.updateRouteData(self.routeV)

        return True

    # }}}

    # {{{ M4
    def move4(self):
        costSuppU = (
            self.params.timeCost[self.nodeUPrevIndex][self.nodeVIndex]
            + self.params.timeCost[self.nodeVIndex][self.nodeXIndex]
            - self.params.timeCost[self.nodeUPrevIndex][self.nodeUIndex]
            - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
        )

        costSuppV = (
            self.params.timeCost[self.nodeVPrevIndex][self.nodeUIndex]
            + self.params.timeCost[self.nodeUIndex][self.nodeYIndex]
            - self.params.timeCost[self.nodeVPrevIndex][self.nodeVIndex]
            - self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
        )

        if not self.intraRouteMove:
            # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
            if costSuppU + costSuppV >= self.routeU.penalty + self.routeV.penalty:
                return False

            costSuppU += (
                self.penaltyExcessDuration(
                    self.routeU.duration + costSuppU + self.serviceV - self.serviceU
                )
                + self.penaltyExcessLoad(self.routeU.load + self.loadV - self.loadU)
                - self.routeU.penalty
            )

            costSuppV += (
                self.penaltyExcessDuration(
                    self.routeV.duration + costSuppV - self.serviceV + self.serviceU
                )
                + self.penaltyExcessLoad(self.routeV.load + self.loadU - self.loadV)
                - self.routeV.penalty
            )

        if costSuppU + costSuppV > -self.MY_EPSILON:
            return False

        if self.nodeUIndex == self.nodeVPrevIndex or self.nodeUIndex == self.nodeYIndex:
            return False

        self.swapNode(self.nodeU, self.nodeV)

        self.numMoves += 1  # Increment move counter before updating route data
        self.searchCompleted = False
        self.updateRouteData(self.routeU)

        if not self.intraRouteMove:
            self.updateRouteData(self.routeV)

        return True

    # }}}

    # {{{ M5
    def move5(self):
        costSuppU = (
            self.params.timeCost[self.nodeUPrevIndex][self.nodeVIndex]
            + self.params.timeCost[self.nodeVIndex][self.nodeXNextIndex]
            - self.params.timeCost[self.nodeUPrevIndex][self.nodeUIndex]
            - self.params.timeCost[self.nodeXIndex][self.nodeXNextIndex]
        )

        costSuppV = (
            self.params.timeCost[self.nodeVPrevIndex][self.nodeUIndex]
            + self.params.timeCost[self.nodeXIndex][self.nodeYIndex]
            - self.params.timeCost[self.nodeVPrevIndex][self.nodeVIndex]
            - self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
        )

        if not self.intraRouteMove:
            # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
            if costSuppU + costSuppV >= self.routeU.penalty + self.routeV.penalty:
                return False

            costSuppU += (
                self.penaltyExcessDuration(
                    self.routeU.duration
                    + costSuppU
                    - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
                    + self.serviceV
                    - self.serviceU
                    - self.serviceX
                )
                + self.penaltyExcessLoad(
                    self.routeU.load + self.loadV - self.loadU - self.loadX
                )
                - self.routeU.penalty
            )

            costSuppV += (
                self.penaltyExcessDuration(
                    self.routeV.duration
                    + costSuppV
                    + self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
                    - self.serviceV
                    + self.serviceU
                    + self.serviceX
                )
                + self.penaltyExcessLoad(
                    self.routeV.load + self.loadU + self.loadX - self.loadV
                )
                - self.routeV.penalty
            )

        if costSuppU + costSuppV > -self.MY_EPSILON:
            return False

        if (
            self.nodeU == self.nodeV.prev
            or self.nodeX == self.nodeV.prev
            or self.nodeU == self.nodeY
            or self.nodeX.isDepot
        ):
            return False

        self.swapNode(self.nodeU, self.nodeV)
        self.insertNode(self.nodeX, self.nodeU)

        self.numMoves += 1  # Increment move counter before updating route data
        self.searchCompleted = False
        self.updateRouteData(self.routeU)

        if not self.intraRouteMove:
            self.updateRouteData(self.routeV)

        return True

    # }}}

    # {{{ M6
    def move6(self):
        costSuppU = (
            self.params.timeCost[self.nodeUPrevIndex][self.nodeVIndex]
            + self.params.timeCost[self.nodeYIndex][self.nodeXNextIndex]
            - self.params.timeCost[self.nodeUPrevIndex][self.nodeUIndex]
            - self.params.timeCost[self.nodeXIndex][self.nodeXNextIndex]
        )

        costSuppV = (
            self.params.timeCost[self.nodeVPrevIndex][self.nodeUIndex]
            + self.params.timeCost[self.nodeXIndex][self.nodeYNextIndex]
            - self.params.timeCost[self.nodeVPrevIndex][self.nodeVIndex]
            - self.params.timeCost[self.nodeYIndex][self.nodeYNextIndex]
        )

        if not self.intraRouteMove:
            # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
            if costSuppU + costSuppV >= self.routeU.penalty + self.routeV.penalty:
                return False

            costSuppU += (
                self.penaltyExcessDuration(
                    self.routeU.duration
                    + costSuppU
                    - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
                    + self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
                    + self.serviceV
                    + self.serviceY
                    - self.serviceU
                    - self.serviceX
                )
                + self.penaltyExcessLoad(
                    self.routeU.load + self.loadV + self.loadY - self.loadU - self.loadX
                )
                - self.routeU.penalty
            )

            costSuppV += (
                self.penaltyExcessDuration(
                    self.routeV.duration
                    + costSuppV
                    + self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
                    - self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
                    - self.serviceV
                    - self.serviceY
                    + self.serviceU
                    + self.serviceX
                )
                + self.penaltyExcessLoad(
                    self.routeV.load + self.loadU + self.loadX - self.loadV - self.loadY
                )
                - self.routeV.penalty
            )

        if costSuppU + costSuppV > -self.MY_EPSILON:
            return False

        if (
            self.nodeX.isDepot
            or self.nodeY.isDepot
            or self.nodeY == self.nodeU.prev
            or self.nodeU == self.nodeY
            or self.nodeX == self.nodeV
            or self.nodeV == self.nodeX.next
        ):
            return False

        self.swapNode(self.nodeU, self.nodeV)
        self.swapNode(self.nodeX, self.nodeY)

        self.numMoves += 1  # Increment move counter before updating route data
        self.searchCompleted = False
        self.updateRouteData(self.routeU)

        if not self.intraRouteMove:
            self.updateRouteData(self.routeV)

        return True

    # }}}

    # {{{ M7
    def move7(self):
        if self.nodeU.position > self.nodeV.position:
            return False

        cost = (
            self.params.timeCost[self.nodeUIndex][self.nodeVIndex]
            + self.params.timeCost[self.nodeXIndex][self.nodeYIndex]
            - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
            - self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
            + self.nodeV.cumulatedReversalDistance
            - self.nodeX.cumulatedReversalDistance
        )

        if cost > -self.MY_EPSILON:
            return False

        if self.nodeU.next == self.nodeV:
            return False

        nodeNum = self.nodeX.next
        self.nodeX.prev = nodeNum
        self.nodeX.next = self.nodeY

        while nodeNum != self.nodeV:
            temp = nodeNum.next
            nodeNum.next = nodeNum.prev
            nodeNum.prev = temp
            nodeNum = temp

        self.nodeV.next = self.nodeV.prev
        self.nodeV.prev = self.nodeU
        self.nodeU.next = self.nodeV
        self.nodeY.prev = self.nodeX

        self.numMoves += 1  # Increment move counter before updating route data
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        return True

    # }}}

    # {{{ M8
    def move8(self):
        cost = (
            self.params.timeCost[self.nodeUIndex][self.nodeVIndex]
            + self.params.timeCost[self.nodeXIndex][self.nodeYIndex]
            - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
            - self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
            + self.nodeV.cumulatedReversalDistance
            + self.routeU.reversalDistance
            - self.nodeX.cumulatedReversalDistance
            - self.routeU.penalty
            - self.routeV.penalty
        )

        # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
        if cost >= 0:
            return False

        cost += (
            self.penaltyExcessDuration(
                self.nodeU.cumulatedTime
                + self.nodeV.cumulatedTime
                + self.nodeV.cumulatedReversalDistance
                + self.params.timeCost[self.nodeUIndex][self.nodeVIndex]
            )
            + self.penaltyExcessDuration(
                self.routeU.duration
                - self.nodeU.cumulatedTime
                - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
                + self.routeU.reversalDistance
                - self.nodeX.cumulatedReversalDistance
                + self.routeV.duration
                - self.nodeV.cumulatedTime
                - self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
                + self.params.timeCost[self.nodeXIndex][self.nodeYIndex]
            )
            + self.penaltyExcessLoad(
                self.nodeU.cumulatedLoad + self.nodeV.cumulatedLoad
            )
            + self.penaltyExcessLoad(
                self.routeU.load
                + self.routeV.load
                - self.nodeU.cumulatedLoad
                - self.nodeV.cumulatedLoad
            )
        )

        if cost > -self.MY_EPSILON:
            return False

        depotU = self.routeU.depot
        depotV = self.routeV.depot
        depotUFin = self.routeU.depot.prev
        depotVFin = self.routeV.depot.prev
        depotVSuiv = depotV.next

        # Reverse nodeX and set its route to routeV
        xx = self.nodeX
        while not xx.isDepot:
            temp = xx.next
            xx.next = xx.prev
            xx.prev = temp
            xx.route = self.routeV
            xx = temp

        # Reverse nodeV and set its route to routeU
        vv = self.nodeV
        while not vv.isDepot:
            temp = vv.prev
            vv.prev = vv.next
            vv.next = temp
            vv.route = self.routeU
            vv = temp

        # Adjust node links
        self.nodeU.next = self.nodeV
        self.nodeV.prev = self.nodeU
        self.nodeX.next = self.nodeY
        self.nodeY.prev = self.nodeX

        if self.nodeX.isDepot:
            depotUFin.next = depotU
            depotUFin.prev = depotVSuiv
            depotUFin.prev.next = depotUFin
            depotV.next = self.nodeY
            self.nodeY.prev = depotV
        elif self.nodeV.isDepot:
            depotV.next = depotUFin.prev
            depotV.next.prev = depotV
            depotV.prev = depotVFin
            depotUFin.prev = self.nodeU
            self.nodeU.next = depotUFin
        else:
            depotV.next = depotUFin.prev
            depotV.next.prev = depotV
            depotUFin.prev = depotVSuiv
            depotUFin.prev.next = depotUFin

        self.numMoves += 1  # Increment move counter before updating route data
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        self.updateRouteData(self.routeV)
        return True

    # }}}

    # {{{ M9
    def move9(self):
        cost = (
            self.params.timeCost[self.nodeUIndex][self.nodeYIndex]
            + self.params.timeCost[self.nodeVIndex][self.nodeXIndex]
            - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
            - self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
            - self.routeU.penalty
            - self.routeV.penalty
        )

        # Early move pruning to save CPU time. Guarantees that this move cannot improve without checking additional (load, duration...) constraints
        if cost >= 0:
            return False

        cost += (
            self.penaltyExcessDuration(
                self.nodeU.cumulatedTime
                + self.routeV.duration
                - self.nodeV.cumulatedTime
                + self.params.timeCost[self.nodeVIndex][self.nodeYIndex]
                - self.params.timeCost[self.nodeUIndex][self.nodeYIndex]
            )
            + self.penaltyExcessDuration(
                self.routeU.duration
                - self.nodeU.cumulatedTime
                - self.params.timeCost[self.nodeUIndex][self.nodeXIndex]
                + self.nodeV.cumulatedTime
                + self.params.timeCost[self.nodeVIndex][self.nodeXIndex]
            )
            + self.penaltyExcessLoad(
                self.nodeU.cumulatedLoad + self.routeV.load - self.nodeV.cumulatedLoad
            )
            + self.penaltyExcessLoad(
                self.nodeV.cumulatedLoad + self.routeU.load - self.nodeU.cumulatedLoad
            )
        )

        if cost > -self.MY_EPSILON:
            return False

        depotU = self.routeU.depot
        depotV = self.routeV.depot
        depotUFin = depotU.prev
        depotVFin = depotV.prev
        depotUpred = depotUFin.prev

        # Reassign route of nodeY to routeU
        count = self.nodeY
        while not count.isDepot:
            count.route = self.routeU
            count = count.next

        # Reassign route of nodeX to routeV
        count = self.nodeX
        while not count.isDepot:
            count.route = self.routeV
            count = count.next

        # Adjust node links
        self.nodeU.next = self.nodeY
        self.nodeY.prev = self.nodeU
        self.nodeV.next = self.nodeX
        self.nodeX.prev = self.nodeV

        if self.nodeX.isDepot:
            depotUFin.prev = depotVFin.prev
            depotUFin.prev.next = depotUFin
            self.nodeV.next = depotVFin
            depotVFin.prev = self.nodeV
        else:
            depotUFin.prev = depotVFin.prev
            depotUFin.prev.next = depotUFin
            depotVFin.prev = depotUpred
            depotVFin.prev.next = depotVFin

        self.numMoves += 1  # Increment move counter before updating route data
        self.searchCompleted = False
        self.updateRouteData(self.routeU)
        self.updateRouteData(self.routeV)
        return True

    # }}}
