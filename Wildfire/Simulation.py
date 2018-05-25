# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:32:32 2017

@author: davey
"""

import numpy
import math
import copy
import pulp
from Fire import Fire


class Simulation():
    # Class for defining a simulation run
    simulations = 0

    def __init__(self):
        # Constructs an instance
        self.fireSeverity = numpy.empty([0, 0])
        self.dangerIndex = numpy.empty([0, 0])
        self.rain = numpy.empty([0, 0])
        self.humidity = numpy.empty([0, 0])
        self.wind = numpy.empty([0, 0])
        self.temperature = numpy.empty([0, 0])
        self.experimentalScenario = None
        self.controls = []
        self.model = None
        self.id = Simulation.simulations
        Simulation.simulations = Simulation.simulations + 1

    def getModel(self):
        return self.model

    def setModel(self, m):
        self.model = m

    def getFireSeverity(self):
        return self.fireSeverity

    def setFireSeverity(self, fs):
        self.fireSeverity(fs)

    def getDangerIndex(self):
        return self.dangerIndex

    def setDangerIndex(self, di):
        self.dangerIndex = di

    def getRain(self):
        return self.rain

    def setRain(self, rain):
        self.rain = rain

    def getHumidity(self):
        return self.humidity

    def setHumidity(self, h):
        self.humidity = h

    def getWind(self):
        return self.wind

    def setWind(self, w):
        self.wind = w

    def getTemperature(self):
        return self.temperature

    def setTemperature(self, t):
        self.temperature = t

    def getExperimentalScenario(self):
        return self.experimentalScenario

    def setExperimentalScenario(self, es):
        self.experimentalScenario = es

    def getControls(self):
        return self.controls

    def setControls(self, c):
        self.controls = c

    def simulate(self):
        # Generate exogenous forward paths for weather and fire starts and save
        # The forward paths are just the Danger Indices at each location
        exogenousPaths = self.forwardPaths()

        # Generate random control paths and store
        randCont = self.randomControls()

        # Generate endogenous fire growth paths given the above and store
        [endogenousPaths, damageMaps] = self.endogenousPaths(exogenousPaths,
                                                             randCont)

        # Use the Monte Carlo paths to compute the policy maps
        rovMaps = self.rov(exogenousPaths,
                           randCont,
                           endogenousPaths,
                           damageMaps)

    def forwardPaths(self):
        # We don't need to store the precipitation, wind, and temperature
        # matrices over time. We only need the resulting danger index

        paths = []

        for path in range(self.model.getROVPaths()):
            paths.append(self.initialForwardPath())

        return paths

    def rov(self, exogenousPaths, randCont, endogenousPaths):
        pass

    def randomControls(self):
        randControls = (numpy.random.choice(
                range(len(self.model.getControls())),
                self.model.getROVPaths()*self.model.getTotalSteps())
            .reshape(self.model.getROVPaths(),
                     self.model.getTotalSteps()))

        return randControls

    def endogenousPaths(self, ep, rc):
        # We store the actual fires and their sizes

        paths = []

        [initialFires, initialAss, initialLocs, cumHours, resourceTypes] = (
            self.initialiseAssignments())

        for path in range(self.model.getROVPaths()):
            [fires, assignments, currLocs, cumHours, cumDamage] = (
                self.initialEndogenousPath(ep[path], rc[path], initialFires,
                                           initialAss, initialLocs, cumHours,
                                           resourceTypes))
            paths.append([fires, assignments, currLocs, cumHours,
                          cumDamage])

        return paths

    def initialForwardPath(self):
        region = self.model.getRegion()
        regionSize = region.getX().size
        timeSteps = self.model.getTotalSteps()

        rain = numpy.empty([timeSteps+1, regionSize])
        rain[0] = region.getRain()
        precipitation = numpy.empty([timeSteps+1, regionSize])
        precipitation[0] = region.getHumidity()
        temperatureMin = numpy.empty([timeSteps+1, regionSize])
        temperatureMin[0] = region.getTemperatureMin()
        temperatureMax = numpy.empty([timeSteps+1, regionSize])
        temperatureMax[0] = region.getTemperatureMax()
        windNS = numpy.empty([timeSteps+1, regionSize])
        windNS[0] = region.getWindN()
        windEW = numpy.empty([timeSteps+1, regionSize])
        windEW[0] = region.getWindE()
        FFDI = numpy.empty([timeSteps+1, regionSize])
        FFDI[0] = region.getDangerIndex()
        windRegimes = numpy.empty([timeSteps+1])
        windRegimes[0] = region.getWindRegime()

        wg = region.getWeatherGenerator()

        # Simulate the path forward from time zero to the end
        for ii in range(timeSteps):
            # Compute weather
            wg.computeWeather(rain, precipitation, temperatureMin,
                              temperatureMax, windRegimes, windNS, windEW,
                              FFDI, ii)
            pass

        return FFDI

    def initialEndogenousPath(self, ep, rc, initialFires, initialAss,
                              initialLocs, cumHours0, resourceTypes):
        timeSteps = self.model.getTotalSteps()
        lookahead = self.model.getLookahead()

        # Keep a whole map of fire in the region. We do not keep track of all
        # the randomly-generated numbers used for fire growth and success (or
        # even first starts for that matter) as it is expected that the fires
        # will have different successes and may even be different fires in the
        # first place. This will save a lot of memory.
        # The assignments list is 1 element longer than the other values. This
        # is because the first element is where the aircraft were assigned in
        # the time step leading up to the start of the simulation.
        assignments = [None]*(timeSteps+1)
        currLocs = [None]*(timeSteps+1)
        cumHours = [None]*(timeSteps+1)
        cumDamage = numpy.empty(timeSteps+1)
        # Initial cumulative damage is zero (we don't care what happened prior
        # to our study period).
        cumDamage[0] = 0.0
        # fireSeverityMap[0] = initialMap
        assignments[0] = initialAss
        currLocs[0] = initialLocs
        cumHours[0] = cumHours0

        fires = [None]*(self.model.getTimeSteps()+1)
        fires[0] = initialFires

        for ii in range(timeSteps):
            control = rc[ii]
            locationProgram = self.model.getNestedOptMethod()

            # NESTED OPTIMISATION #############################################
            # Optimise aircraft locations given selected control and state
            assignments[ii+1] = self.optimalLocations(
                control,
                fires[ii],
                assignments[ii],
                currLocs[ii],
                cumHours[ii],
                resourceTypes,
                ep[ii:(ii + lookahead - 1)],
                locationProgram)

            # Given the locations found for this control, update the fire
            # severities for the next time period. We use the probabilities. We
            # also compute the new damage incurred and the hours used.
            [cumHours[ii+1], fires[ii+1], currLocs[ii+1], damage] = (
                self.fireSeverity(fires[ii], assignments[ii+1], currLocs[ii],
                                  cumHours[ii], resourceTypes, ep[ii]))
            cumDamage[ii+1] = cumDamage[ii] + damage

        return [fires, assignments, currLocs, cumHours, cumDamage]

    def initialiseAssignments(self):
        # Find where all of the fires currently are
        initialFires = self.model.getRegion().getFires()

        # Create a matrix of all existing aircraft (tankers then helicopters)
        totalTankers = 0
        totalHelis = 0

        airStripIdx = 0
        aircraftPositions = []
        initialLocations = []
        resourceTypes = []
        cumHours = []
        resources = self.model.getRegion().getResourceTypes()

        for airStrip in self.model.getRegion().getStations()[0]:
            currentTankers = airStrip.getAirTankers()
            currentHelis = airStrip.getHelicopters()

            totalTankers = totalTankers + len(currentTankers)
            totalHelis = totalHelis + len(currentHelis)

            for tanker in range(len(currentTankers)):
                aircraftPositions.append(airStripIdx)
                resourceTypes.append(0)
                initialLocations.append([airStrip.getLocation()[0],
                                         airStrip.getLocation()[1]])
                cumHours.append(resources[0].getMaxDailyHours())

            for heli in range(len(currentHelis)):
                aircraftPositions.append(airStripIdx)
                resourceTypes.append(1)
                initialLocations.append([airStrip.getLocation()[0],
                                         airStrip.getLocation()[1]])
                cumHours.append(resources[1].getMaxDailyHours())

            airStripIdx = airStripIdx + 1

        initialBaseAss = numpy.array(aircraftPositions)
        resourceTypes = numpy.array(resourceTypes)
        initialLocations = numpy.array(initialLocations)

        # First column of the assignment is the base assignment. The second is
        # the assignment to fires. Initially, no aircraft are assigned to fires
        # until the location program is run.
        # A fire assignment of 0 implies that the aircraft is not assigned to
        # any fire
        initialFireAss = numpy.zeros(initialBaseAss.size)
        initialAss = numpy.array([initialBaseAss, initialFireAss]).transpose()

        return [initialFires, initialAss, initialLocations, cumHours,
                resourceTypes]

    def comparator(self, ffdi, time):
        comparators = numpy.empty(self.region.getX().size(), 1)
        # Serial
        for ii in range(self.region.getX().size):
            # Linear interpolation. Assume ffdis evenly spaced
            veg = self.region.getVegetation()[ii]
            ffdiRange = self.region.getVegetations[veg].getFFDIRange()
            occurrenceProbs = self.region.getVegetations[veg].getOccurrence()
            ffdis = ffdiRange.size
            ffdiMinIdx = math.floor((ffdi[time][ii] - ffdiRange[0])*(ffdis-1) /
                                    (ffdiRange[ffdis] - ffdiRange[0]))
            ffdiMaxIdx = ffdiMinIdx + 1

            if ffdiMinIdx < 0:
                ffdiMinIdx = 0
                ffdiMaxIdx = 1
            elif ffdiMaxIdx >= ffdis:
                ffdiMinIdx = ffdis - 2
                ffdiMaxIdx = ffdis - 1

            xd = ((ffdi[time][ii] - ffdiRange[ffdiMinIdx]) /
                  (ffdiRange[ffdiMaxIdx] - ffdiRange[ffdiMinIdx]))

            comparators[ii] = (xd * occurrenceProbs[ffdiMinIdx] +
                               (1 - xd) * occurrenceProbs[ffdiMaxIdx])

        return comparators

    def optimalLocations(self, randCont, fires, assignmentsCurr, currLocs,
                         cumHoursCurr, resourceTypes, ffdi,
                         locationProgram):
        switch = {
            0: self.maxCover,
            1: self.pCenter,
            2: self.assignmentOne,
            3: self.assignmentTwo
        }

        prog = switch.get(locationProgram)
        assignments = prog(randCont, fires, assignmentsCurr, currLocs,
                           cumHoursCurr, resourceTypes, ffdi)

        return assignments

    def maxCover(self, randCont, fires, assignmentsCurr, currLocs,
                 cumHoursCurr, resourceTypes, ffdi):
        # We only consider the maximum cover of 1 tanker and 1 helicopter for
        # now
        maxCoverDists = numpy.empty(2)
        lookahead = self.model.getLookahead()
        patches = self.model.getRegion().getPatches()
        bases = self.model.getRegion().getStations()
        speed = numpy.empty(2)

        for aircraft in range(2):
            speed[aircraft] = (self.model.getResourceTypes()[aircraft]
                               .getSpeed())
            maxTime = self.model.getCoverTime()/60
            maxCoverDists[aircraft] = speed[aircraft]*maxTime

        # We only get air bases for now
        baseNodeSufficient = [None]*2
        baseFiresSufficient = [None]*2
        baseTPrev = [sum(assignmentsCurr[resourceTypes == 0, 0] == jj)
                     for jj in range(len(bases))]
        baseHPrev = [sum(assignmentsCurr[resourceTypes == 1, 0] == jj)
                     for jj in range(len(bases))]
        basesX = numpy.array([
                self.model.getRegion().getStations()[ii].getLocation()[0]
                for ii in range(len(self.model.getRegion().getPatches()))])
        basesY = numpy.array([
                self.model.getRegion().getStations()[ii].getLocation()[1]
                for ii in range(len(patches))])
        firesX = numpy.array([
                fires[ii].getLocation()[0]
                for ii in range(len(fires))]).reshape(len(fires), 1)
        firesY = numpy.array([
                fires[ii].getLocation()[1]
                for ii in range(len(fires))]).reshape(len(fires), 1)
        baseFireDistances = numpy.sqrt(
                numpy.power(numpy.tile(basesX, (len(fires), 1)) -
                            numpy.tile(firesX, (1, len(basesX))), 2) +
                numpy.power(numpy.tile(basesY, (len(fires), 1)) -
                            numpy.tile(firesY, (1, len(basesY))), 2))

        expectedNewFiresPatches = numpy.zeros([lookahead, len(patches)])

        # Expected fires over the lookahead period
        for time in range(lookahead):
            for patch in range(len(patches)):
                veg = self.model.getRegion().getVegetation()[patch]
                occurrenceProbsRange = (
                    self.model.getRegion().getVegetation()[veg]
                        .getOccurrence())
                ffdiRange = (
                        self.model.getRegion().getVegetations[veg]
                        .getFFDIRange())
                ffdis = ffdiRange.size
                ffdiMinIdx = math.floor(
                    (ffdi[time][patch] - ffdiRange[0])*(ffdis-1) /
                    (ffdiRange[ffdis-1] - ffdiRange[0]))
                ffdiMaxIdx = ffdiMinIdx + 1

                if ffdiMinIdx < 0:
                    ffdiMinIdx = 0
                    ffdiMaxIdx = 1
                elif ffdiMaxIdx >= ffdis:
                    ffdiMinIdx = ffdis - 2
                    ffdiMaxIdx = ffdis - 1

                xd = ((ffdi[time][patch] - ffdiRange[ffdiMinIdx]) /
                      (ffdiRange[ffdiMaxIdx] - ffdiRange[ffdiMinIdx]))

                expectedNewFiresPatches[time, patch] = (
                        expectedNewFiresPatches[time, patch] +
                        xd*occurrenceProbsRange[ffdiMinIdx] +
                        (1 - xd)*occurrenceProbsRange[ffdiMaxIdx])

        for aircraft in range(2):
            baseNodeSufficient[aircraft] = (
                    self.model.getRegion().getStationPatchDistances()[0] <=
                    maxCoverDists[aircraft])
            baseFiresSufficient[aircraft] = (
                    baseFireDistances <= maxCoverDists[aircraft])

        # Expected number of fires that can be reached by aircraft stationed at
        # each base
        # Potential
        accessibleFiresBaseP = [None]*2
        accessibleFiresBaseE = [None]*2

        for ii in range(2):
            accessibleFiresBaseP[ii] = numpy.matmul(baseNodeSufficient[ii],
                                                    expectedNewFiresPatches)
            accessibleFiresBaseE[ii] = numpy.sum(baseFiresSufficient[ii],
                                                 axis=0)

        # SET UP THE LINEAR PROGRAM (using PuLP for now)
        relocate = pulp.LpProblem("Fire Resource Relocation", pulp.LpMaximize)

        # INSTANTIATE THE SOLVER (GUROBI FOR NOW)
        solver = pulp.GUROBI()
        relocate.setSolver(solver)
        relocate.solver.buildSolverModel(relocate)

        # Decision Variables ##################################################
        # Patch cover variables
        X1_T_i_Vars = ["X1_T_Patch_" + str(ii+1) for ii in range(len(patches))]

        # Fire cover variables
        X1_T_l_Vars = ["X1_T_Fire_" + str(ll+1) for ll in range(len(fires))]

        # Base assignment variables - Tankers
        X2_Tank_T_j_Vars = ["X2_Tank_T_" + str(jj+1)
                            for jj in range(len(bases))]

        # Base assignment variables - Helicopters
        X2_Heli_T_j_Vars = ["X2_Heli_T_" + str(jj+1)
                            for jj in range(len(bases))]

        # Relocation between bases - Tankers
        Y_Tank_T_j1j2_Vars = [["Y_Tank_T_" + str(j1+1) + "_" + str(j2+1)
                               for j1 in range(len(bases))]
                              for j2 in range(len(bases))]

        # Relocation between bases - Helicopters
        Y_Heli_T_j1j2_Vars = [["Y_Heli_T_" + str(j1+1) + "_" + str(j2+1)
                               for j1 in range(len(bases))]
                              for j2 in range(len(bases))]

        # Demand of tankers at each base
        D_Tank_j_Vars = ["D_Tank_" + str(jj+1) for jj in range(len(bases))]

        # Demand of helicopters at each base
        D_Heli_j_Vars = ["D_Heli" + str(jj+1) for jj in range(len(bases))]

        # Supply of tankers from each base
        S_Tank_j_Vars = ["S_Tank_" + str(jj+1) for jj in range(len(bases))]

        # Supply of helicopters from each base
        S_Heli_j_Vars = ["S_Heli_" + str(jj+1) for jj in range(len(bases))]

        # Create the LP variables for use in PULP #############################
        X1_T_i = [pulp.LpVariable(X1_T_i_Vars[ii],
                                  cat="Binary")
                  for ii in range(len(patches))]

        X1_T_l = [pulp.LpVariable(X1_T_l_Vars[ll],
                                  cat="Binary")
                  for ll in range(len(fires))]

        X2_Tank_T_j = [pulp.LpVariable(X2_Tank_T_j_Vars[jj],
                                       lowBound=0,
                                       upBound=bases[jj].getMaxTankers())
                       for jj in range(len(bases))]

        X2_Heli_T_j = [pulp.LpVariable(X2_Heli_T_j_Vars[jj],
                                       lowBound=0,
                                       upBound=bases[jj].getMaxHelicopters())
                       for jj in range(len(bases))]

        Y_Tank_T_j1j2 = [[pulp.LpVariable(Y_Tank_T_j1j2_Vars[j1][j2],
                                          lowBound=0,
                                          upBound=totalTankers,
                                          cat="Integer")
                          for j1 in range(len(bases))]
                         for j2 in range(len(bases))]

        Y_Heli_T_j1j2 = [[pulp.LpVariable(Y_Heli_T_j1j2_Vars[j1][j2],
                                          lowBound=0,
                                          upBound=totalHelis,
                                          cat="Integer")
                          for j1 in range(len(bases))]
                         for j2 in range(len(bases))]

        D_Tank_j = [pulp.LpVariable(D_Tank_j_Vars[jj],
                                    lowBound=0,
                                    upBound=totalTankers,
                                    cat="Integer")
                    for jj in range(len(bases))]

        D_Heli_j = [pulp.LpVariable(D_Heli_j_Vars[jj],
                                    lowBound=0,
                                    upBound=totalHelis,
                                    cat="Integer")
                    for jj in range(len(bases))]

        S_Tank_j = [pulp.LpVariable(S_Tank_j_Vars[jj],
                                    lowBound=0,
                                    upBound=totalTankers,
                                    cat="Integer")
                    for jj in range(len(bases))]

        S_Heli_j = [pulp.LpVariable(S_Heli_j_Vars[jj],
                                    lowBound=0,
                                    upBound=totalHelis,
                                    cat="Integer")
                    for jj in range(len(bases))]

        # The LP uses an aggregation of the weather generator data (FFDI and
        # FireSeverityMap)
        [ffdiAgg, fireSeverityAgg] = self.aggregateWeatherData(ffdi, fires)

        # Objective ###########################################################
        # Due to the aggregated nature of this program, we cannot track
        # individual aircraft. Instead, the objective assumes that for
        # relocation, aircraft are initially at their respective bases (even
        # though this may not be the case in reality).
        relocate += ((pulp.lpSum([[lambda2 * (1 - lambda1) *
                                   ffdiAgg[t][ii] * X1_T_i[ii]
                                   for ii in range(len(patches))]
                                  for t in range(lookahead)]) +
                      pulp.lpSum([lambda2 * lambda1 *
                                  fireSeverityAgg[ii] *
                                  X1_T_l[ii]
                                  for ii in range(len(patches))]) -
                      pulp.lpSum([[(1 - lambda2) *
                                   (baseDistances[j1, j2] *
                                    Y_Tank_T_j1j2[j1, j2] +
                                    baseDistances[j1, j2] *
                                    Y_Heli_T_j1j2[j1, j2])
                                   for j1 in range(len(bases))]
                                  for j2 in range(len(bases))])),
                     "Total location and relocation cost over horizon")

        # Constraints #########################################################
        relocate += (pulp.lpSum([X2_Tank_T_j[jj]
                                for jj in range(len(bases))]) == totalTankers,
                     "Sum of tankers is total")

        relocate += (pulp.lpSum([X2_Heli_T_j[jj]
                                 for jj in range(len(bases))]) == totalHelis,
                     "Sum of helicopters is total")

        for ii in range(len(patches)):
            relocate += ((X1_T_i[ii] -
                          pulp.lpSum([X2_Tank_T_j[jj] *
                                     baseNodeSufficient[0][ii, jj] /
                                     (sum([accessibleFiresBaseP[0][t, jj]
                                           for t in range(lookahead)]) +
                                      accessibleFiresBaseE[jj])
                                     for jj in range(len(bases))])) <= 0,
                         "Patch " + str(ii) +
                         " covered by at least one tanker")

        for ii in range(len(patches)):
            relocate += ((X1_T_i[ii] -
                          pulp.lpSum([X2_Heli_T_j[jj] *
                                     baseNodeSufficient[1][ii, jj] /
                                     (sum([accessibleFiresBaseP[1][t, jj]
                                           for t in range(lookahead)]) +
                                      accessibleFiresBaseE[jj])
                                     for jj in range(len(bases))])) <= 0,
                         "Patch " + str(ii) +
                         " covered by at least one helicopter")

        for ii in range(len(fires)):
            relocate += ((X1_T_i[ii] -
                          pulp.lpSum([X2_Tank_T_j[jj] *
                                     baseFiresSufficient[0][ii, jj] /
                                     (sum([accessibleFiresBaseP[0][t, jj]
                                           for t in range(lookahead)]) +
                                      accessibleFiresBaseE[jj])
                                     for jj in range(len(bases))])) <= 0,
                         "Fire " + str(ii) +
                         " covered by at least one tanker")

        for ii in range(len(fires)):
            relocate += ((X1_T_i[ii] -
                          pulp.lpSum([X2_Heli_T_j[jj] *
                                     baseFiresSufficient[1][ii, jj] /
                                     (sum([accessibleFiresBaseP[1][t, jj]
                                           for t in range(lookahead)]) +
                                      accessibleFiresBaseE[jj])
                                     for jj in range(len(bases))])) <= 0,
                         "Fire " + str(ii) +
                         " covered by at least one helicopter")

        # The signs in Chow and Regan (2011), INFOR appear to be the wrong way
        # around. Corrected here.
        for jj in range(len(bases)):
            relocate += ((-S_Tank_j[jj] - X2_Tank_T_j[jj] + baseTPrev[jj]) ==
                         0,
                         "Supply of tankers from base " + str(jj))

            relocate += ((-S_Heli_j[jj] - X2_Heli_T_j[jj] + baseHPrev[jj]) ==
                         0,
                         "Supply of helicopters from base " + str(jj))

            relocate += ((-D_Tank_j[jj] + X2_Tank_T_j[jj] - baseTPrev[jj]) ==
                         0,
                         "Demand of tankers by base " + str(jj))

            relocate += ((-D_Heli_j[jj] + X2_Heli_T_j[jj] - baseHPrev[jj]) ==
                         0,
                         "Demand of helicopters by base " + str(jj))

        for jj in range(len(bases)):
            relocate += (pulp.lpSum([Y_Tank_T_j1j2[jj][j2]
                                     for j2 in range(len(bases))]) ==
                         S_Tank_j[jj],
                         "Supply flow conservation for tankers for base " +
                         str(jj))

            relocate += (pulp.lpSum([Y_Heli_T_j1j2[jj][j2]
                                     for j2 in range(len(bases))]) ==
                         S_Heli_j[jj],
                         "Supply flow conservation for helicopters for base " +
                         str(jj))

            relocate += (pulp.lpSum([Y_Tank_T_j1j2[j1][jj]
                                     for j1 in range(len(bases))]) ==
                         D_Tank_j[jj],
                         "Demand flow conservation for tankers for base " +
                         str(jj))

            relocate += (pulp.lpSum([Y_Heli_T_j1j2[j1][jj]
                                     for j1 in range(len(bases))]) ==
                         D_Heli_j[jj],
                         "Demand flow conservation for helicopters for base " +
                         str(jj))

        relocate.writeLP("Relocation.lp")
        relocate.solve()

        # SEND THE ASSIGNMENT OUTPUT TO THE LINEAR PROGRAM
#        print("Status: ", LpStatus[relocate.status])
#        print("Value: ", value(relocate.objective))

        # Extract the optimal values for the variables
        varsdict = {}
        for var in relocate.variables():
            varsdict[var.name] = var.varValue

        # Base assignments
        baseAss = numpy.empty([bases, 2])
        for base in range(len(bases)):
            # Tankers
            baseAss[base, 0] = varsdict['X2_Tank_T_' + str(base+1)]
            # Helicopters
            baseAss[base, 1] = varsdict['X2_Heli_T_' + str(base+1)]

        # Relocations
        relocs = []
        relocs.append(numpy.empty([len(bases), len(bases)]))
        relocs.append(numpy.empty([len(bases), len(bases)]))

        for base1 in range(len(bases)):
            for base2 in range(len(bases)):
                relocs[0][base1][base2] = varsdict["Y_Tank_T_" + str(base1) +
                                                   "_" + str(base2)]
                relocs[1][base1][base2] = varsdict["Y_Heli_T_" + str(base1) +
                                                   "_" + str(base2)]

        assignmentsNew = self.assignmentsHeuristic(assignmentsCurr, fires,
                                                   cumHoursCurr, currLocs,
                                                   resourceTypes, baseAss,
                                                   relocs, lambda2)

        return assignmentsNew

    def pCenter(self, randCont, fires, assignmentsCurr, currLocs,
                cumHoursCurr, resourceTypes, ffdi):
        # This is the second type of relocation program. Like the Max Cover
        # formulation, we only compute aggregate values, not assignments of
        # individual aircraft.
        lookahead = self.model.getLookahead()

        # We only get air bases for now
        bases = self.model.getRegion().getStations
        baseTPrev = [sum(assignmentsCurr[resourceTypes == 0, 0] == jj)
                     for jj in range(len(bases))]
        baseHPrev = [sum(assignmentsCurr[resourceTypes == 1, 0] == jj)
                     for jj in range(len(bases))]

        # Decision variables
        # First, names for parameters and variables
        patches = self.model.getRegion().getPatches()
        totalTankers = numpy.sum(resourceTypes == 0)
        totalHelis = numpy.sum(resourceTypes == 1)
        lambda1 = self.model.getControls()[randCont].getLambda1()
        lambda2 = self.model.getControls()[randCont].getLambda2()
        baseDistances = self.model.getRegion().getStationDistances()[0]
        basePatchDistances = (self.model.getRegion()
                              .getStationPatchDistances()[0])

        baseTPrev = [sum(assignmentsCurr[resourceTypes == 0, 0] == jj)
                     for jj in range(len(bases))]
        baseHPrev = [sum(assignmentsCurr[resourceTypes == 1, 0] == jj)
                     for jj in range(len(bases))]
        basesX = numpy.array([
                self.model.getRegion().getStations()[ii].getLocation()[0]
                for ii in range(len(patches))])
        basesY = numpy.array([
                self.model.getRegion().getStations()[ii].getLocation()[1]
                for ii in range(len(patches))])
        firesX = numpy.array([
                fires[ii].getLocation()[0]
                for ii in range(len(fires))]).reshape(len(fires), 1)
        firesY = numpy.array([
                fires[ii].getLocation()[1]
                for ii in range(len(fires))]).reshape(len(fires), 1)
        baseFireDistances = numpy.sqrt(
                numpy.power(numpy.tile(basesX, (len(fires), 1)) -
                            numpy.tile(firesX, (1, len(basesX))), 2) +
                numpy.power(numpy.tile(basesY, (len(fires), 1)) -
                            numpy.tile(firesY, (1, len(basesY))), 2))

        # SET UP THE LINEAR PROGRAM
        relocate = pulp.LpProblem("Fire Resource Relocation", pulp.LpMinimize)

        # INSTANTIATE THE SOLVER (GUROBI FOR NOW)
        solver = pulp.GUROBI()
        relocate.setSolver(solver)
        relocate.solver.buildSolverModel(relocate)

        # Decision Variables ##################################################
        # Base assignment variables - Tankers
        X2_Tank_T_j_Vars = ["X2_Tank_T_" + str(jj+1)
                            for jj in range(len(bases))]

        # Base assignment variables - Helicopters
        X2_Heli_T_j_Vars = ["X2_Heli_T_" + str(jj+1)
                            for jj in range(len(bases))]

        # Relocation between bases - Tankers
        Y_Tank_T_j1j2_Vars = [["Y_Tank_T_" + str(j1+1) + "_" + str(j2+1)
                               for j1 in range(len(bases))]
                              for j2 in range(len(bases))]

        # Relocation between bases - Helicopters
        Y_Heli_T_j1j2_Vars = [["Y_Heli_T_" + str(j1+1) + "_" + str(j2+1)
                               for j1 in range(len(bases))]
                              for j2 in range(len(bases))]

        # Demand of tankers at each base
        D_Tank_j_Vars = ["D_Tank_" + str(jj+1) for jj in range(len(bases))]

        # Demand of helicopters at each base
        D_Heli_j_Vars = ["D_Heli" + str(jj+1) for jj in range(len(bases))]

        # Supply of tankers from each base
        S_Tank_j_Vars = ["S_Tank_" + str(jj+1) for jj in range(len(bases))]

        # Supply of helicopters from each base
        S_Heli_j_Vars = ["S_Heli_" + str(jj+1) for jj in range(len(bases))]

        # Create the LP variables for use in PULP #############################
        X2_Tank_T_j = [pulp.LpVariable(X2_Tank_T_j_Vars[jj],
                                       lowBound=0,
                                       upBound=bases[jj].getMaxTankers())
                       for jj in range(len(bases))]

        X2_Heli_T_j = [pulp.LpVariable(X2_Heli_T_j_Vars[jj],
                                       lowBound=0,
                                       upBound=bases[jj].getMaxHelicopters())
                       for jj in range(len(bases))]

        Y_Tank_T_j1j2 = [[pulp.LpVariable(Y_Tank_T_j1j2_Vars[j1][j2],
                                          lowBound=0,
                                          upBound=totalTankers,
                                          cat="Integer")
                          for j1 in range(len(bases))]
                         for j2 in range(len(bases))]

        Y_Heli_T_j1j2 = [[pulp.LpVariable(Y_Heli_T_j1j2_Vars[j1][j2],
                                          lowBound=0,
                                          upBound=totalHelis,
                                          cat="Integer")
                          for j1 in range(len(bases))]
                         for j2 in range(len(bases))]

        D_Tank_j = [pulp.LpVariable(D_Tank_j_Vars[jj],
                                    lowBound=0,
                                    upBound=totalTankers,
                                    cat="Integer")
                    for jj in range(len(bases))]

        D_Heli_j = [pulp.LpVariable(D_Heli_j_Vars[jj],
                                    lowBound=0,
                                    upBound=totalHelis,
                                    cat="Integer")
                    for jj in range(len(bases))]

        S_Tank_j = [pulp.LpVariable(S_Tank_j_Vars[jj],
                                    lowBound=0,
                                    upBound=totalTankers,
                                    cat="Integer")
                    for jj in range(len(bases))]

        S_Heli_j = [pulp.LpVariable(S_Heli_j_Vars[jj],
                                    lowBound=0,
                                    upBound=totalHelis,
                                    cat="Integer")
                    for jj in range(len(bases))]

        # The LP uses an aggregation of the weather generator data (FFDI and
        # FireSeverityMap)
        [ffdiAgg, fireSeverityAgg] = self.aggregateWeatherData(ffdi, fires)

        # Objective ###########################################################
        # Due to the aggregated nature of this program, we cannot track
        # individual aircraft. Instead, the objective assumes that for
        # relocation, aircraft are initially at their respective bases (even
        # though this may not be the case in reality).
        relocate += ((pulp.lpSum([[[lambda2 * (1 - lambda1) * ffdiAgg[t][ii] *
                                    basePatchDistances[ii, jj] *
                                    X2_Tank_T_j[jj] +
                                    lambda2 * (1 - lambda1) * ffdiAgg[t][ii] *
                                    basePatchDistances[ii, jj] *
                                    X2_Heli_T_j[jj]
                                    for jj in range(len(bases))]
                                   for ii in range(len(patches))]
                                  for t in range(lookahead)]) +
                      pulp.lpSum([[lambda2 * lambda1 * fireSeverityAgg[ll] *
                                   baseFireDistances[ll, jj] *
                                   X2_Tank_T_j[jj] +
                                   lambda2 * lambda1 * fireSeverityAgg[ll] *
                                   baseFireDistances[ll, jj]
                                   for jj in range(len(bases))]
                                  for ll in range(fires)]) +
                      pulp.lpSum([[(1 - lambda2) *
                                   (baseDistances[j1, j2] *
                                    Y_Tank_T_j1j2[j1, j2] +
                                    baseDistances[j1, j2] *
                                    Y_Heli_T_j1j2[j1, j2])
                                   for j1 in range(len(bases))]
                                  for j2 in range(len(bases))])),
                     "Total location and relocation cost over horizon")

        # Constraints #########################################################
        relocate += (pulp.lpSum([X2_Tank_T_j[jj]
                                for jj in range(len(bases))]) == totalTankers,
                     "Sum of tankers is total")

        relocate += (pulp.lpSum([X2_Heli_T_j[jj]
                                 for jj in range(len(bases))]) == totalHelis,
                     "Sum of helicopters is total")

        # The signs in Chow and Regan (2011), INFOR appear to be the wrong way
        # around. Corrected here.
        for jj in range(len(bases)):
            relocate += ((-S_Tank_j[jj] - X2_Tank_T_j[jj] + baseTPrev[jj]) ==
                         0,
                         "Supply of tankers from base " + str(jj))

            relocate += ((-S_Heli_j[jj] - X2_Heli_T_j[jj] + baseHPrev[jj]) ==
                         0,
                         "Supply of helicopters from base " + str(jj))

            relocate += ((-D_Tank_j[jj] + X2_Tank_T_j[jj] - baseTPrev[jj]) ==
                         0,
                         "Demand of tankers by base " + str(jj))

            relocate += ((-D_Heli_j[jj] + X2_Heli_T_j[jj] - baseHPrev[jj]) ==
                         0,
                         "Demand of helicopters by base " + str(jj))

        for jj in range(len(bases)):
            relocate += (pulp.lpSum([Y_Tank_T_j1j2[jj][j2]
                                     for j2 in range(len(bases))]) ==
                         S_Tank_j[jj],
                         "Supply flow conservation for tankers for base " +
                         str(jj))

            relocate += (pulp.lpSum([Y_Heli_T_j1j2[jj][j2]
                                     for j2 in range(len(bases))]) ==
                         S_Heli_j[jj],
                         "Supply flow conservation for helicopters for base " +
                         str(jj))

            relocate += (pulp.lpSum([Y_Tank_T_j1j2[j1][jj]
                                     for j1 in range(len(bases))]) ==
                         D_Tank_j[jj],
                         "Demand flow conservation for tankers for base " +
                         str(jj))

            relocate += (pulp.lpSum([Y_Heli_T_j1j2[j1][jj]
                                     for j1 in range(len(bases))]) ==
                         D_Heli_j[jj],
                         "Demand flow conservation for helicopters for base " +
                         str(jj))

        relocate.writeLP("Relocation.lp")
        relocate.solve()

        # Extract the optimal vlaues for the variables
        varsdict = {}
        for var in relocate.variables():
            varsdict[var.name] = var.varValue

        # Base assignments
        baseAss = numpy.empty([bases, 2])
        for base in range(len(bases)):
            # Tankers
            baseAss[base, 0] = varsdict['X2_Tank_T_' + str(base)]
            # Helicopters
            baseAss[base, 1] = varsdict['X2_Heli_T_' + str(base)]

        # Relocations
        relocs = []
        relocs.append(numpy.empty([len(bases), len(bases)]))
        relocs.append(numpy.empty([len(bases), len(bases)]))

        for base1 in range(len(bases)):
            for base2 in range(len(bases)):
                relocs[0][base1][base2] = varsdict["Y_Tank_T_" + str(base1) +
                                                   "_" + str(base2)]
                relocs[1][base1][base2] = varsdict["Y_Heli_T_" + str(base1) +
                                                   "_" + str(base2)]

        assignmentsNew = self.assignmentsHeuristic(assignmentsCurr, fires,
                                                   cumHoursCurr, currLocs,
                                                   resourceTypes, baseAss,
                                                   relocs, lambda2)

        return assignmentsNew

    def assignmentOne(self, randCont, fires, assignmentsCurr, currLocs,
                      cumHoursCurr, resourceTypes, ffdi):

        lookahead = self.model.getLookahead()
        stepSize = self.model.getStepSize()
        patches = self.model.getRegion().getPatches()
        bases = self.model.getRegion().getStations()
        totalTankers = numpy.sum(resourceTypes == 0)
        totalHelis = numpy.sum(resourceTypes == 1)
        lambda1 = self.model.getControls()[randCont].getLambda1()
        lambda2 = self.model.getControls()[randCont].getLambda2()

        # We need threshold for early and late attacks
        maxCoverDists = numpy.empty(2)
        speed = numpy.zero(2)
        maxDistLookahead = numpy.zero(2)

        for aircraft in range(2):
            speed[aircraft] = (self.model.getResourceTypes()[aircraft]
                               .getSpeed())
            maxTime = self.model.getCoverTime()/60
            maxCoverDists[aircraft] = speed[aircraft]*maxTime
            maxDistLookahead[aircraft] =

        # We only get air bases for now
        baseNodeSufficient = [None]*2
        baseFiresSufficient = [None]*2
        baseAircraftSufficient = [None]*2
        aircraftFiresSufficient = [None]*2
        baseTPrev = [sum(assignmentsCurr[resourceTypes == 0, 0] == jj)
                     for jj in range(len(bases))]
        baseHPrev = [sum(assignmentsCurr[resourceTypes == 1, 0] == jj)
                     for jj in range(len(bases))]
        basesX = numpy.array([
                self.model.getRegion().getStations()[ii].getLocation()[0]
                for ii in range(len(patches))])
        basesY = numpy.array([
                self.model.getRegion().getStations()[ii].getLocation()[1]
                for ii in range(len(patches))])
        firesX = numpy.array([
                fires[ii].getLocation()[0]
                for ii in range(len(fires))]).reshape(len(fires), 1)
        firesY = numpy.array([
                fires[ii].getLocation()[1]
                for ii in range(len(fires))]).reshape(len(fires), 1)
        tankersX = numpy.array([
                currLocs[ii, 0]
                for ii in range(len(resourceTypes))
                if resourceTypes[ii] == 0])
        tankersY = numpy.array([
                currLocs[ii, 1]
                for ii in range(len(resourceTypes))
                if resourceTypes[ii] == 0])
        helisX = numpy.array([
                currLocs[ii, 0]
                for ii in range(len(resourceTypes))
                if resourceTypes[ii] == 1])
        helisY = numpy.array([
                currLocs[ii, 1]
                for ii in range(len(resourceTypes))
                if resourceTypes[ii] == 1])
        baseFireDistances = numpy.sqrt(
                numpy.power(numpy.tile(basesX, (len(fires), 1)) -
                            numpy.tile(firesX, (1, len(basesX))), 2) +
                numpy.power(numpy.tile(basesY, (len(fires), 1)) -
                            numpy.tile(firesY, (1, len(basesY))), 2))
        baseAircraftDistances = [None]*2
        baseAircraftDistances[0] = numpy.sqrt(
                numpy.power(numpy.tile(basesX, (len(totalTankers), 1)) -
                            numpy.tile(tankersX, (1, len(basesX))), 2) +
                numpy.power(numpy.tile(basesY, (len(totalTankers), 1)) -
                            numpy.tile(tankersY, (1, len(basesX))), 2))
        baseAircraftDistances[1] = numpy.sqrt(
                numpy.power(numpy.tile(basesX, (len(totalHelis), 1)) -
                            numpy.tile(helisX, (1, len(basesX))), 2) +
                numpy.power(numpy.tile(basesY, (len(totalHelis), 1)) -
                            numpy.tile(helisY, (1, len(basesX))), 2))
        aircraftFireDistances = [None]*2
        aircraftFireDistances[0] = numpy.sqrt(
                numpy.power(numpy.tile(firesX, (len(totalTankers), 1)) -
                            numpy.tile(tankersX, (1, len(firesX))), 2) +
                numpy.power(numpy.tile(firesY, (len(totalTankers), 1)) -
                            numpy.tile(tankersY, (1, len(firesY))), 2))
        aircraftFireDistances[1] = numpy.sqrt(
                numpy.power(numpy.tile(firesX, (len(totalHelis), 1)) -
                            numpy.tile(helisX, (1, len(firesX))), 2) +
                numpy.power(numpy.tile(firesY, (len(totalHelis), 1)) -
                            numpy.tile(helisY, (1, len(firesY))), 2))

        for aircraft in range(2):
            baseNodeSufficient[aircraft] = (
                    self.model.getRegion().getStationPatchDistances()[0] <=
                    maxCoverDists[aircraft])
            baseFiresSufficient[aircraft] = (
                    baseFireDistances <= maxCoverDists[aircraft])

            aircraftFiresSufficient[aircraft] = (
                    aircraftFireDistances[aircraft] <=
                    maxCoverDists[aircraft])

            baseAircraftSufficient[aircraft] = (
                    baseAircraftDistances[aircraft] <=
                    maxCoverDists[aircraft])

        # Configurations (any combination of [0,1,2+][e|l][T|H])
        # Arrangement is
        # TE|TL|HE|HL
        submat = numpy.array(range(0, 3)).reshape((3, 1))
        config = numpy.empty([81, 4])

        for ii in range(4):
            config[:, ii] = numpy.tile(numpy.repeat(submat, 3**(3 - ii),
                                       axis=0), (3**ii, 1)).flatten()

        # Expected damage vs. configuration for each patch (Existing fires,
        # Phi_E)
        # Precomputed values
        # We need to get the corresponding FFDI index of the actual FFDIs at
        # each location at each time step within the expected damage matrices
        # that are stored in memory from earlier. The first index of the below
        # two arrays is the configuration, while the remaining indices are the
        # FFDIs at each time period in the lookahead.
        # Expected damage vs. configuration for each patch (Potential fires,
        # Phi_P):

        # EXISTING FIRES
        expDE = self.model.getRegion().getExpDE()

        # POTENTIAL FIRES
        expDP = self.model.getRegion().getExpDP()

        # Travel times aircraft to fires

        # Travel times aircraft to bases

        # SET UP THE LINEAR PROGRAM (using PuLP for now)
        relocate = pulp.LpProblem("Fire Resource Relocation", pulp.LpMaximize)

        # INSTANTIATE THE SOLVER (GUROBI FOR NOW)
        solver = pulp.GUROBI()
        relocate.setSolver(solver)
        relocate.solver.buildSolverModel(relocate)

        # Decision Variables ##################################################
        # Patch selected config at time t variables
        Delta_P_im_Vars = [[["Delta_P_" + str(ii+1) + "_" +
                             str(t+1) + "_" + str(m+1)
                             for ii in range(len(patches))]
                            for t in range(lookahead)]
                           for m in range(config.shape[0])]

        # Fire selected config at time t variables
        Delta_E_lm_Vars = [[["Delta_E_" + str(ll+1) + "_" +
                             str(t+1) + "_" + str(m+1)
                             for ll in range(len(fires))]
                            for t in range(lookahead)]
                           for m in range(config.shape[0])]

        # Relocation of aircraft k to location j - Tankers
        Z1_Tank_jk_Vars = [["Z1_Tank_JK_" + str(jj+1) + "_" + str(kk+1)
                            for jj in range(len(bases))]
                           for kk in range(totalTankers)]

        # Relocation of aircraft k to location j - Helicopters
        Z1_Heli_jk_Vars = [["Z1_Tank_JK_" + str(kk+1) + "_" + str(jj+1)
                            for jj in range(len(bases))]
                           for kk in range(totalHelis)]

        # Aircraft k stationed at base j to tackle fire l - Tankers
        Z2_Tank_jkl_Vars = [[["Z2_Tank_JKL_" + str(jj+1) + "_" + str(kk+1) +
                              "_" + str(ll+1)
                              for jj in range(len(bases))]
                             for kk in range(totalTankers)]
                            for ll in range(len(fires))]

        # Aircraft k stationed at base j to tackle fire l - Helicopters
        Z2_Heli_jkl_Vars = [[["Z2_Heli_JKL_" + str(jj+1) + "_" + str(kk+1) +
                              "_" + str(ll+1)
                              for jj in range(len(bases))]
                             for kk in range(len(totalHelis))]
                            for ll in range(len(fires))]

        # Create the LP variables for use in PULP #############################
        Delta_P_im = [[[pulp.LpVariable(Delta_P_im_Vars[ii, t, m],
                                        cat="Binary")
                        for ii in range(len(patches))]
                       for t in range(lookahead)]
                      for m in range(config.shape[0])]

        Delta_E_lm = [[[pulp.LpVariable(Delta_E_lm_Vars[ii, t, m],
                                        cat="Binary")
                        for ii in range(len(fires))]
                       for t in range(lookahead)]
                      for m in range(config.shape[0])]

        Z1_Tank_jk = [[pulp.LpVariable(Z1_Tank_jk_Vars[jj, kk],
                                       cat="Binary")
                       for jj in range(len(bases))]
                      for kk in range(totalTankers)]

        Z1_Heli_jk = [[pulp.LpVariable(Z1_Heli_jk_Vars[jj, kk],
                                       cat="Binary")
                       for jj in range(len(bases))]
                      for kk in range(totalHelis)]

        Z2_Tank_jkl = [[[pulp.LpVariable(Z2_Tank_jkl_Vars[jj, kk, ll],
                                         cat="Binary")
                         for jj in range(len(bases))]
                        for kk in range(totalTankers)]
                       for ll in range(len(fires))]

        Z2_Heli_jkl = [[[pulp.LpVariable(Z2_Heli_jkl_Vars[jj, kk, ll],
                                         cat="Binary")
                         for jj in range(len(bases))]
                        for kk in range(totalHelis)]
                       for ll in range(len(fires))]

        [expDP_im,expDE_lm] = self.expectedDamages()

        # Objective ###########################################################
        relocate += ((pulp.lpSum([[lambda2*(1 - lambda1)*expDP_im[ii, mm] *
                                   Delta_P_im[ii, mm]
                                   for ii in range(len(patches))]
                                  for mm in range(config.shape[0])]) +
                      pulp.lpSum([[lambda2*lambda1*expDE_lm[ll, mm] *
                                   Delta_E_lm[ll, mm]
                                   for ll in range(len(fires))]
                                  for mm in range(config.shape[0])]) +
                      pulp.lpSum([[[(1 - lambda2) *
                                    baseAircraftDistances[0][jj, kk] *
                                    Z1_Tank_jk[jj, kk]
                                    for jj in range(len(bases))]
                                   for kk in range(totalTankers)]]) +
                      pulp.lpSum([[[(1-lambda2) *
                                    baseAircraftDistances[1][jj, kk] *
                                    Z1_Heli_jk[jj, kk]
                                    for jj in range(len(bases))]
                                   for kk in range(totalHelis)]])),
                     "Total weighted expected damage and relocation costs "
                     "over horizon")

        # Constraints #########################################################
        # Existing fire config activation
        for ll in range(len(fires)):
            for mm in range(config.shape[0]):
                relocate += (Delta_E_lm[ll, mm] <=
                             (1 if config[mm, 0] == 0 else
                              (1.0/config[mm, 0]) *
                              sum([sum([aircraftFiresSufficient[0][kk, ll]
                                        for kk in range(totalTankers)])
                                   for jj in range(len(bases))])),
                             "Correct number of early tankers for fire config")

        for ll in range(len(fires)):
            for mm in range(config.shape[0]):
                relocate += (Delta_E_lm[ll, mm] <=
                             (1 if config[mm, 2] == 0 else
                              (1.0/config[mm, 2]) *
                              sum([sum([aircraftFiresSufficient[1][kk, ll]
                                        for kk in range(totalHelis)])
                                   for jj in range(len(bases))])),
                             "Correct number of early helis for fire config")

        for ll in range(len(fires)):
            for mm in range(config.shape[0]):
                relocate += (Delta_E_lm[ll, mm] <=
                             (1 if config[mm, 1] == 0 else
                              (1.0/config[mm, 1]) *
                              sum([sum([1 - aircraftFiresSufficient[0][kk, ll]
                                        for kk in range(totalTankers)])
                                   for jj in range(len(bases))])),
                             "Correct number of late tankers for fire config")

        for ll in range(len(fires)):
            for mm in range(config.shape[0]):
                relocate += (Delta_E_lm[ll, mm] <=
                             (1 if config[mm, 3] == 0 else
                              (1.0/config[mm, 3]) *
                              sum([sum([1 - aircraftFiresSufficient[1][kk, ll]
                                        for kk in range(totalHelis)])
                                   for jj in range(len(bases))])),
                             "Correct number of late helis for fire config")

        # Potential fires config activation
        for ii in range(len(patches)):
            for mm in range(config.shape[0]):
                relocate += (Delta_P_im[ii, mm] <=
                             (1 if config[mm, 0] == 0 else
                              (1.0/config[mm, 0]) *
                              sum([sum([baseNodeSufficient[0][ii, jj] *
                                        (Z1_Tank_jk[jj, kk] -
                                         sum([Z2_Tank_jkl[jj, kk, ll]
                                              for ll in range(len(fires))]))
                                        for kk in range(totalTankers)])
                                   for jj in range(len(bases))])),
                             "Correct number of early tankers for potential "
                             "fires config")

        for ii in range(len(patches)):
            for mm in range(config.shape[0]):
                relocate += (Delta_P_im[ii, mm] <=
                             (1 if config[mm, 0] == 0 else
                              (1.0/config[mm, 0]) *
                              sum([sum([baseNodeSufficient[0][ii, jj] *
                                        (Z1_Tank_jk[jj, kk] -
                                         sum([Z2_Tank_jkl[jj, kk, ll]
                                              for ll in range(len(fires))]))
                                        for kk in range(totalTankers)])
                                   for jj in range(len(bases))])),
                             "Correct number of early tankers for potential "
                             "fires config")

    # This is the main assignment problem we solve as it allows us to account
    # for positions of aircraft, their proximities to fires
    def assignmentTwo(self, randCont, fireSeverityMap, assignmentsCurr,
                      currLocs, cumHoursCurr, resourceTypes, ffdi):
        pass

    def assignmentsHeuristic(self, assignmentsCurr, fires, cumHoursCurr,
                             currLocs, resourceTypes, baseAssignments, relocs,
                             lambda2):
        # This is a simple greedy heuristic whereby an upper limit is first
        # decided for all fires (the same for each fire).
        # Max aircraft to be allocated to any fire (we upper limit set as 5 of
        # each type for now with a lower limit of 1). This is arbitrary, then
        # again, the Max Cover and P-Median formulations are arbitrary in
        # comparison to the assignment problems as they do not take into
        # consideration actual fire damage but only proxies.

        # FIRST ASSIGN AIRCRAFT TO FIRES ######################################
        assignmentsNew = numpy.zeros(assignmentsCurr.shape)
        kMax = lambda2*4+1
        bases = len(self.model.getRegion().getStations()[0])

        # Compute the distances between aircraft and the fires. We only want
        # the closest aircraft with enough hours to fight the fire
        dists2Fire = self.currPos2Fire(currLocs, fires)

        # Next, sort the fires by severity and start assigning aircraft to them
        # in order (i.e. fill demand for most severe fires first)
        sortedIdx = numpy.argsort(idx)[::-1]

        remainingMask = numpy.ones(len(resourceTypes))
        resources = self.model.getRegion().getResourceTypes()
        speeds = ((resourceTypes == 0)*resources[0].getSpeed() +
                  (resourceTypes == 1)*resources[1].getSpeed())

        for sidx in sortedIdx:
            tankers = (numpy.argsort(
                dists2Fire[sortedIdx[sidx],
                           numpy.multiply(remainingMask, resourcesType == 0)] *
                (numpy.divide(dists2Fire[sortedIdx[sidx], :], speeds) <
                    cumHours)))

            helicopters = (numpy.argsort(
                dists2Fire[sortedIdx[sidx],
                           numpy.multiply(remainingMask, resourcesType == 1)] *
                (numpy.divide(dists2Fire[sortedIdx[sidx], :], speeds) <
                    cumHours)))

            assignmentsNew[tankers[0:min(len(tankers), kMax)], 1] = (
                sortedIdx[sidx] + 1)

            assignmentsNew[helicopters[0:min(len(tankers), kMax)], 1] = (
                sortedIdx[sidx] + 1)

            remainingMask[tankers] = 0
            remainingMask[helicopters] = 0

        # NEXT, ASSIGN AIRCRAFT TO BASES
        # Use the bulk relocation variables from earlier to perform the
        # assignments of individual aircraft.
        # We go through each base and select the aircraft that have the
        # shortest travel time to go up to the number required. If the hours of
        # these aircraft are not sufficient to cover the travel distance, then
        # the aircraft are not relocated.
        # N.B. This also means that the relocation variables in the previous
        # linear program may not necessarily be followed exactly as intended.
        # This is a shortcoming of this approach.
        for base1 in range(bases):
            for base2 in range(bases):
                distB1B2 = self.model.getRegion().getStationDistances()[0][
                    base1, base2]
                # Number of aircraft from base1 to base2
                # TANKERS
                noSwap = reloc[0][base1][base2]
                # Tankers initially at the base that are free to move
                initB1Free = numpy.nonzero(numpy.multiply(
                    resourceTypes == 0,
                    numpy.multiply(assignmentsCurr[:, 0] == base1, cumHours >=
                                   numpy.divide(distB1B2, speeds))))
                # Tankers initially at the base that are NOT free to move
                initB1Fixed = numpy.nonzero(
                    numpy.multiply(resourceTypes == 0,
                                   numpy.multiply(assignmentsCurr[:, 0] ==
                                                  base1,
                                                  cumHours <
                                                  numpy.divide(distB1B2,
                                                               speeds))))

                mask = numpy.zeros(len(assignmentsCurr[:, 0]))

                mask[initB1Free] = 1
                # Tankers sorted by remaining hours, with only sufficient
                # remaining hours allowed to relocate
                toRelocate = numpy.argsort(numpy.multiply(cumHours,
                                                          mask))[::-1]
                noReloc = min(len(numpy.nonzero(toRelocate)), noSwap)

                # Assign the aircraft to the bases
                assignmentsNew[toRelocate[0:noReloc], 0] = base2
                assignmentsNew[initB1Fixed, 0] = base1

                # HELICOPTERS
                noSwap = reloc[1][base1][base2]
                # Helicopters initially at the base that are free to move
                initB1Free = numpy.nonzero(
                    numpy.multiply(resourceTypes == 1,
                                   numpy.multiply(assignmentsCurr[:, 0] ==
                                                  base1,
                                                  cumHours >=
                                                  numpy.divide(distB1B2,
                                                               speeds))))

                # Helicopters initially at the base that are NOT free to move
                initB1Fixed = numpy.nonzero(
                    numpy.multiply(resourceTypes == 1,
                                   numpy.multiply(assignmentsCurr[:, 0] ==
                                                  base1, cumHours <
                                                  numpy.divide(distB1B2,
                                                               speeds))))

                mask = numpy.zeros(len(assignmentsCurr[:, 0]))

                mask[initB1Free] = 1
                # Tankers sorted by remaining hours, with only sufficient
                # remaining hours allowed to relocate
                toRelocate = numpy.argsort(
                    numpy.multiply(cumHours, mask))[::-1]

                noReloc = min(len(numpy.nonzero(toRelocate)), noSwap)

                # Assign the aircraft to the bases
                assignmentsNew[toRelocate[0:noReloc], 0] = base2
                assignmentsNew[initB1Fixed, 0] = base1

        # We can return the assignments now
        return assignmentsNew

    def assingmentsSub1(self, assignmentsCurr, fireSeverityMap, cumHoursCurr,
                        currLocs, resourceTypes, baseAssignments, relocs,
                        lambda2):
        # To find the assignment of individual aircraft to bases and fires, we
        # use the following simple assignment program that assigns the nearest
        # aircraft to the current fires. The goal is to minimise travel times
        # to fires based on current positions and available hours. Aircraft
        # must be able to perform at least one pass before running out of
        # available hours. We only use one tanker and one helicopter unless
        # there are idle aircraft stationed within a twenty minute attack
        # distance, in which case we can assign another one tanker and one
        # helicopter.
        # Travel speeds of the two types of aircraft
        travelSpeeds = numpy.empty(2)

        for aircraft in range(2):
            travelSpeeds[aircraft] = (self.modelGetResourceTypes()[aircraft]
                                      .getSpeed())

        # Compute the distances between aircraft and the fires. We only want
        # the closest aircraft with enough hours to fight the fire
        dists2Fire = self.currPos2Fire(currLocs, fireSeverityMap)

        # First, we take the relocations found already to determine which
        # aircraft from each base to relocate to the corresponding new bases.
        # We use the relocation variables from earlier. The relocation problem
        # seeks to maximise the remaining travel hours of aircraft for the
        # purpose of relocation (note, this may not be the end result as
        # aircraft will also fight fires, which may require more than one
        # attack. This is uncertain.)
        assignment = pulp.LpProblem("Assignment of aircraft to fires",
                                    pulp.LpMinimize)

        # Objective

        # Next, aircraft are allocated to fires so as to minimise travel time
        # for the first pass.
        # Max aircraft to be allocated to any fire (we upper limit set as 5 of
        # each type for now with a lower limit of 1). This is arbitrary, then
        # again, the Max Cover and P-Median formulations are arbitrary in
        # comparison to the assignment problems as they do not take into
        # consideration actual fire damage but only proxies.
        kMax = lambda2*4+1

        assignment = pulp.LpProblem("Assignment of aircraft to bases",
                                    pulp.LpMinimize)

        return assignmentsNew

    def currPos2Fire(self, currLocs, fires):
        # Computes the distance between each aircraft and each fire

        xFire = numpy.array([fires[fire].getLocation()[0]
                             for fire in range(len(fires))])

        yFire = numpy.array([fires[fire].getLocation()[1]
                             for fire in range(len(fires))])

        xAircraft = numpy.array([currLocs[aircraft][0]
                                 for aircraft in currLocs.shape[0]])

        yAircraft = numpy.array([currLocs[aircraft][1]
                                 for aircraft in currLocs.shape[0]])

        noFires = xFire.shape
        noAircraft = xAircraft.shape

        X1 = numpy.tile(xFire.transpose(), (noAircraft[0], 1))
        X2 = numpy.tile(xAircraft, (noFires[0], 1)).transpose()
        Y1 = numpy.tile(yFire.transpose(), (noAircraft[0], 1))
        Y2 = numpy.tile(yAircraft, (noFires[0], 1)).transpose()

        dists2Fire = numpy.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)

        return dists2Fire

    def fireSeverity(self, firesOld, assignments, currLocs, cumHours,
                     resourceTypes, ffdi):
        # This routine performs the simulation for a single time step

        # EXISTING FIRES ######################################################
        # First, compute the extinguishing success for each active fire given
        # the allocations
        timeStep = self.model.getStepSize()  # In hours
        damage = 0.0
        currLocsNew = numpy.copy(currLocs)
        cumHoursNew = numpy.copy(cumHours)

        firesNew = []
        fireIdx = 1
        for fire in firesOld:
            # Find out the aircraft assigned to this fire
            fireNew = copy.copy(fire)
            veg = self.model.getRegion().getVegetation()[
                tuple(fire.getLocation())]
            ffdiRange = (self.model.getRegion().getVegetations()[veg]
                         .getFFDIRange())
            rocMeanRange = (self.model.getRegion().getVegetations[veg]
                            .getROCA2PerHour()[0])
            rocSDRange = (self.model.getRegion().getVegetations[veg]
                          .getROCA2PerHour()[1])
            succTankerRange = (self.model.getRegion().getVegetations[veg]
                               .getExtinguishingSuccess()[0])
            succHeliRange = (self.model.getRegion().getVegetations[veg]
                             .getExtinguishingSuccess()[1])
            ffdiFire = ffdi[tuple(fire.getLocation())]
            assignedAircraft = numpy.nonzero(assignments[:, 1] == fireIdx)
            damage = damage + Simulation.fightFire(self.model, fire,
                                                   assignedAircraft,
                                                   currLocsNew, cumHoursNew,
                                                   resourceTypes, ffdiFire,
                                                   veg, ffdiRange,
                                                   rocMeanRange, rocSDRange,
                                                   succTankerRange,
                                                   succHeliRange, timeStep)
            if fireNew.getSize() > 0:
                firesNew.append(fireNew)

        # POTENTIAL FIRES #####################################################
        occurrenceProbPatches = numpy.array(len(self.model.getRegion()
                                                .getVegetations()))
        fireGrowthRatePatches = numpy.array(len(self.model.getRegion()
                                                .getVegetations()))

        for patch in range(len(self.model.getRegion().getPatches())):
            veg = self.model.getRegion().getVegetation()[patch]
            occurrenceProbsRange = (self.model.getRegion().getVegetation()[veg]
                                    .getOccurrence())
            rocMeanRange = (self.model.getRegion().getVegetations[veg]
                            .getROCA2PerHour()[0])
            rocSDRange = (self.model.getRegion().getVegetations[veg]
                          .getROCA2PerHour()[1])
            ffdiRange = (self.model.getRegion().getVegetations[veg]
                         .getFFDIRange())
            ffdis = ffdiRange.size
            ffdiMinIdx = math.floor((ffdi[patch] - ffdiRange[0])*(ffdis-1) /
                                    (ffdiRange[ffdis-1] - ffdiRange[0]))
            ffdiMaxIdx = ffdiMinIdx + 1

            if ffdiMinIdx < 0:
                ffdiMinIdx = 0
                ffdiMaxIdx = 1
            elif ffdiMaxIdx >= ffdis:
                ffdiMinIdx = ffdis - 2
                ffdiMaxIdx = ffdis - 1

            xd = ((ffdi[patch] - ffdiRange[ffdiMinIdx]) /
                  (ffdiRange[ffdiMaxIdx] - ffdiRange[ffdiMinIdx]))
            occurrenceProbPatches[patch] = (
                xd*occurrenceProbsRange[ffdiMinIdx] +
                (1 - xd)*occurrenceProbsRange[ffdiMaxIdx])
            rocMean = (xd*rocMeanRange[ffdiMinIdx] +
                       (1 - xd)*rocMeanRange[ffdiMaxIdx])
            rocSD = xd*rocSDRange[ffdiMinIdx] + (1-xd)*rocSDRange[ffdiMaxIdx]
            fireGrowthRatePatches[patch] = numpy.random.normal(rocMean, rocSD)

        # Using these probabilities, compute the number of fires occurring and
        # the times at which they occur.
        noFiresPerPatch = (
            numpy.multiply(numpy.divide(1, occurrenceProbPatches),
                           numpy.log(1 - numpy.random.uniform(
                               0,
                               1,
                               len(occurrenceProbPatches))))).astype(int)

        totalNewFires = noFiresPerPatch.sum()
        newFires = numpy.empty(totalNewFires)
        iterator = 0
        nonZeroPatches = numpy.nonzero(noFiresPerPatch)

        for patch in range(len(numpy.nonzero(noFiresPerPatch))):
            newFires[iterator:(iterator +
                               noFiresPerPatch[nonZeroPatches[patch]] - 1)] = (
                               nonZeroPatches[patch])
            iterator = iterator + noFiresPerPatch[nonZeroPatches[patch]]

        newFires = numpy.array(newFires,
                               numpy.random.uniform(0, 1, totalNewFires) *
                               timeStep)
        # Now sort by time of occurrence
        newFires = newFires[:, numpy.argsort(newFires[:, 1])]

        # Now fight each of these fires using the nearest available aircraft at
        # the time of ignition
        elapsedTime = 0
        start = 0
        newFireObjs = []

        while elapsedTime < self.model.getTimeStep():
            timeStep = 0
            if start < (len(newFires[:, 1]) - 1):
                timeStep = newFires[start + 1, 1] - elapsedTime
            else:
                timeStep = self.model.getStepSize()

            # Find aircraft to assign to this new fire ########################
            # Get the nearest available helicopter and air tanker
            xfire = (self.model.getRegion().getPatches()[newFires[start][0]]
                     .getCentroid()[0])
            yfire = (self.model.getRegion().getPatches()[newFires[start][0]]
                     .getCentroid()[1])

            [nearestTanker, nearestHeli] = self.assignNearestAvailable(
                assignments, currLocs, cumHours, resourceTypes, [xfire, yfire],
                self.model.getStepSize() - elapsedTime)

            if nearestTanker > 0:
                assignments[nearestTanker] = start + len(firesOld)
            if nearestHeli > 0:
                assignments[nearestHeli] = start + len(firesOld)

            # Append the fire to the list of active fires
            fire = Fire()
            fire.setLocation(self.model.getRegion().getPatches()[
                newFires[start]].getCentroid())
            newFireObjs.append(fire)

            # Fight this new fire (plus other new fires still active) up to the
            # start of the next fire
            for fireIdx in range(len(newFireObjs)):
                fire = newFireObjs[fireIdx]
                veg = (model.getRegion().getVegetation()[tuple(
                    fire.getLocation())])
                ffdiRange = (model.getRegion().getVegetations()[veg]
                             .getFFDIRange())
                rocMeanRange = (model.getRegion().getVegetations[veg]
                                .getROCA2PerHour()[0])
                rocSDRange = (model.getRegion().getVegetations[veg]
                              .getROCA2PerHour()[1])
                succTankerRange = (model.getRegion().getVegetations[veg]
                                   .getExtinguishingSuccess()[0])
                succHeliRange = (model.getRegion().getVegetations[veg]
                                 .getExtinguishingSuccess()[1])
                ffdiFire = ffdi[tuple(fire.getLocation())]

                if fire.getSize() > 0:
                    assignedAircraft = numpy.nonzero(assignments[:, 1] == (
                        fireIdx + len(firesOld) + 1))
                    damage = damage + Simulation.fightFire(
                        self.model, fire, assignedAircraft, currLocsNew,
                        cumHoursNew, resourceTypes, ffdiFire, veg, ffdiRange,
                        rocMeanRange, rocSDRange, succTankerRange,
                        succHeliRange, timeStep)
                    # If this pass extinguished the fire, the aircraft become
                    # available again
                    if fire.getSize() == 0:
                        assignments[numpy.nonzero(assignments[:, 1] == (
                            fireIdx + len(firesOld) + 1))] = 0

            if start < (len(newFires[1, :]) - 1):
                elapsedTime = newFires[start + 1, 1]
            else:
                elapsedTime = self.model.getTimeStep()

            start = start + 1

        for fireIdx in range(totalNewFires):
            if fire.getSize() > 0:
                firesNew.append(fire)

        # Randomly generate new fires based on the FFDIs in each of the patches
        # We first need the indices of the bounding FFDIs in the FFDI list for
        # each vegetation.
#        ffdiMatrix = []
#        occurrenceMatrix = []
#        for vegetation in self.model.getRegion().getVegetations():
#            ffdiMatrix.append(vegetation.getFFDIRange())
#            occurrenceMatrix.append(vegetation.getOccurrence())
#
#        ffdiMatrix = numpy.array(ffdiMatrix)
#        occurrenceMatrix = numpy.array(occurrenceMatrix)

        # We have computed the new locations for the aircraft assigned to fires
        # (Both existing and new fires generated in this period).
        # Now we need to determine the positions of the aircraft that were not
        # assigned to fires. They may have been in the process of relocating.
        idleAircraft = numpy.nonzero(assignments[:, 1] == 0)

        for aircraft in idleAircraft:
            # If the distance to the assigned base is less than the distance
            # the aircraft can travel in this period, move to the base and
            # update its travel hours.
            baseIdx = assignments[idleAircraft[aircraft], 0]
            xInit = currLocs[idleAircraft[aircraft]][0]
            yInit = currLocs[idleAircraft[aircraft]][1]
            xBase = self.model.getStations()[0][baseIdx].getLocation()[0]
            yBase = self.model.getStations()[0][baseIdx].getLocation()[1]
            curr2Base = math.sqrt((xBase-xInit)**2 + (yBase-yInit)**2)
            speed = resources[resourceTypes[idleAircraft[aircraft]]].getSpeed()

            if curr2Base < speed*timeStep:
                # Aircraft makes it to destination
                currLocsNew[idleAircraft[aircraft][0]] = xBase
                currLocsNew[idleAircraft[aircraft][1]] = yBase
                cumHoursNew[assignedAircraft[aircraft]] = cumHours[
                    assignedAircraft[aircraft]] - curr2Base/speed
            else:
                # Aircraft is still on its way
                remDist2Base = curr2Base - trem*speed
                xD2B = remDist2Base/curr2Base
                currLocsNew[idleAircraft[aircraft][0]] = xBase + xD2B*(xInit -
                                                                       xBase)
                currLocsNew[idleAircraft[aircraft][1]] = yBase + xD2B*(yInit -
                                                                       yBase)
                cumHoursNew[assignedAircraft[aircraft]] = cumHours[
                    assignedAircraft[aircraft]] - timeStep

        damage = damage + sum(additionalSeverity)

        # Use the times at which the fires occur to determine the damage and
        # fire size by the end of the period. Add to the existing severity
        # matrix and recorded damage
        # Return the results
        return [cumHoursNew, firesNew, currLocsNew, damage]

    @staticmethod
    def fightFire(model, fire, assignments, currLocs, cumHours, resourceTypes,
                  ffdi, veg, ffdiRange, rocMeanRange, rocSDRange,
                  succTankerRange, succHeliRange, timeStep):
        # Get the probabilities of success for each aircraft type
        severity = fire.getSize()
        resources = model.getRegion().getResourceTypes()
        ffdis = len(ffdiRange)
        ffdiMinIdx = math.floor((ffdi - ffdiRange[0])*(ffdis - 1) /
                                (ffdiRange[ffdis-1] - ffdiRange[0]))
        ffdiMaxIdx = ffdiMinIdx + 1

        if ffdiMinIdx < 0:
            ffdiMinIdx = 0
            ffdiMaxIdx = 1
        elif ffdiMaxIdx >= ffdis:
            ffdiMinIdx = ffdis - 2
            ffdiMaxIdx = ffdis - 1

        xd = ((ffdi[tuple(fire.getLocation())] - ffdiRange[ffdiMinIdx]) /
              (ffdiRange[ffdiMaxIdx] - ffdiRange[ffdiMinIdx]))

        # Rate of change per hour
        rocMean = (xd*rocMeanRange[ffdiMinIdx] + (1 - xd) *
                   rocMeanRange[ffdiMaxIdx])
        rocSD = (xd*rocSDRange[ffdiMinIdx] + (1 - xd) *
                 rocSDRange[ffdiMaxIdx])

        # Extinguishing success for each type of aircraft. We only have
        # tankers and helicopters for now.
        svs = numpy.empty(2)
        svs[0] = (xd*succTankerRange[ffdiMinIdx] + (1 - xd) *
                  succTankerRange[ffdiMaxIdx])
        svs[1] = (xd*succHeliRange[ffdiMinIdx] + (1 - xd) *
                  succHeliRange[ffdiMaxIdx])

        # Figure out how many times each aircraft assigned to the fire will
        # visit
        visits = []
        totalVisits = numpy.zeros(len(assignments))
        # init2Fires = numpy.zeros(len(assignedAircraft))
        # base2Fires = numpy.zeros(len(assignedAircraft))

        # Simulate attacks
        for aircraft in range(len(assignedAircraft)):
            # Distances to destinations are how far the aircraft are to the
            # assigned fire.
            # Aircraft can slightly overshoot the remaining number of hours
            # but they cannot leave for a new attack if they have already
            # done so.
            baseIdx = assignments[assignedAircraft[aircraft], 0]
            remHours = (resources[resourceTypes[assignedAircraft[aircraft]]]
                        .getMaxDailyHours() -
                        cumHours[assignedAircraft[aircraft]])
            xInit = currLocs[assignedAircraft[aircraft]][0]
            yInit = currLocs[assignedAircraft[aircraft]][1]
            xBase = model.getStations()[0][baseIdx].getLocation()[0]
            yBase = model.getStations()[0][baseIdx].getLocation()[1]
            xFire = fire.getLocation()[0]
            yFire = fire.getLocation()[1]
            init2Fire = math.sqrt((xFire-xInit)**2 + (yFire-yInit)**2)
            base2Fire = math.sqrt((xBase-xFire)**2 + (yBase-yFire)**2)
            speed = (resources[resourceTypes[assignedAircraft[aircraft]]]
                     .getSpeed())
            # init2Fires[aircraft] = init2Fire
            # base2Fires[aircraft] = base2Fire

            # Is the current position near enough to visit at least once?
            trem = timeStep

            if init2Fire < speed*timeStep:
                # Visit: 1. Time of visit
                #        2. Aircraft ID
                visit = [None]*2
                visit[0] = assignedAircraft[aircraft]
                visit[1] = init2Fire/speed
                trem = trem - init2Fire/speed
                visits.append(visit)
                totalVisits[aircraft] = totalVisits[aircraft] + 1

            # Now check when other possible visits may occur from the
            # aircraft moving between the fire and base
            tripTime = 2*base2Fire/speed

            while trem > 0 and remHours > 0:
                if 2*base2Fire < speed*trem:
                    visit = [None]*2
                    visit[0] = aircraft
                    visit[1] = 2*base2Fire/speed + visits[len(visits)-1][1]
                    trem = trem - 2*base2Fire/speed
                    visits.append(visit)
                    totalVisits[aircraft] = totalVisits[aircraft] + 1
                    trem = trem - tripTime
                    remHours = remHours - tripTime

        # Now sort the aircraft visits and perform them. Before each visit,
        # grow the fire and compute the incremental damage. Then, have the
        # aircraft visit. If successful in extinguishing, set fire severity
        # to zero and find locations of all attacking aircraft (send back
        # to base if in the air).
        visits = numpy.array(visits)
        visits = visits[visits[:, 1].argsort()]

        extinguished = False
        elapsedTime = 0.0
        visit = 0
        totalVisits = len(visits)
        damage = 0.0

        while (not(extinguished) and elapsedTime < timeStep and visit <
               totalVisits):
            timeInterval = visits[visit, 1] - elapsedTime
            aircraft = int(visits[visit, 0])

            # Compute growth of fire up until the visit as well as the
            # cumulative damage incurred
            severityOld = severity
            severity = (severity*numpy.random.normal(rocMean, rocSD) *
                        timeInterval)
            damage = damage + (severity - severityOld)
            elapsedTime = elapsedTime + timeInterval

            # Compute whether the fire is extinguished by the visit
            if numpy.random.uniform(0, 1) < svs[resourceTypes[aircraft]]:
                extinguished = True
                fire.setFinalSize(severity)
                fire.setSize(0.0)
                fire.setEnd(elapsedTime)

        if not(extinguished):
            elapsedTime = timeStep
            fire.setSize(severity)

        # Time available to relocate back to base from extinguishing
        trem = timeStep - elapsedTime

        # Compute the final positions of all aircraft for this time period
        # given the success of fighting the fire. Also record the number of
        # flying hours used up.
#        cumHoursNew = numpy.copy(cumHours)
        for aircraft in range(len(assignedAircraft)):
            # Find location at time of extinguishing
            visitIdxes = numpy.nonzero(visits[:, 0] <= elapsedTime)
            baseIdx = assignments[assignedAircraft[aircraft], 0]
            xInit = currLocs[assignedAircraft[aircraft]][0]
            yInit = currLocs[assignedAircraft[aircraft]][1]
            xBase = self.model.getStations()[0][baseIdx].getLocation()[0]
            yBase = self.model.getStations()[0][baseIdx].getLocation()[1]
            xFire = fire.getLocation()[0]
            yFire = fire.getLocation()[1]
            init2Fire = math.sqrt((xFire-xInit)**2 + (yFire-yInit)**2)
            base2Fire = math.sqrt((xBase-xFire)**2 + (yBase-yFire)**2)
            init2Base = math.sqrt((xInit-xBase)**2 + (yInit-yBase)**2)
            speed = (resources[resourceTypes[assignedAircraft[aircraft]]]
                     .getSpeed())

            if len(visitIdxes) > 0:
                # See how long it was since the last visit to determine
                # location
                finalVisitTime = visits[visitIdxes[len(visitIdxes) - 1][1]]

                if extinguished:
                    # Aircraft will return to base
                    if trem*speed >= base2Fire:
                        # Aircraft diverts back to base immediately and
                        # makes it there before the next period no matter
                        # its position
                        currLocs[assignedAircraft[aircraft][0]] = xBase
                        currLocs[assignedAircraft[aircraft][1]] = yBase
                        if elapsedTime - finalVisitTime > base2Fire/speed:
                            # Aircraft is returning to fire so needs to turn
                            # around
                            cumHours[assignedAircraft[aircraft]] = (
                                cumHours[assignedAircraft[aircraft]] -
                                elapsedTime -
                                (elapsedTime -
                                 finalVisitTime -
                                 base2Fire/speed))
                        else:
                            # Just let the aircraft continute back to base
                            cumHours[assignedAircraft[aircraft]] = (
                                cumHours[assignedAircraft[aircraft]] -
                                elapsedTime -
                                base2Fire/speed +
                                (elapsedTime - finalVisitTime))
                    else:
                        # Position of aircraft at time of extinguishing
                        distSinceFinalVisit = speed*(elapsedTime -
                                                     finalVisitTime)

                        if distSinceFinalVisit > base2Fire:
                            # Aircraft is heading to fire at the end
                            dist2Base = distSinceFinalVisit - base2Fire

                            if dist2Base/speed < trem:
                                # Aircraft will make it back to base in time
                                currLocs[assignedAircraft[aircraft][0]] = xBase
                                currLocs[assignedAircraft[aircraft][1]] = yBase
                                cumHours[assignedAircraft[aircraft]] = (
                                    cumHours[assignedAircraft[aircraft]] -
                                    elapsedTime -
                                    dist2Base/speed)
                            else:
                                # Aircraft will still be on its way at the end
                                remDist2Base = dist2Base - trem*speed
                                xD2B = remDist2Base/base2Fire
                                currLocs[assignedAircraft[aircraft][0]] = (
                                    xBase + xD2B*(xFire - xBase))
                                currLocs[assignedAircraft[aircraft][1]] = (
                                    yBase + xD2B*(yFire - yBase))
                                cumHours[assignedAircraft[aircraft]] = (
                                    cumHours[assignedAircraft[aircraft]] -
                                    timeStep)
                        else:
                            # Aircraft is returning to base
                            dist2Base = base2Fire - distSinceFinalVisit

                            if dist2Base/speed < trem:
                                # Aircraft will make it back to base in time
                                currLocs[assignedAircraft[aircraft][0]] = xBase
                                currLocs[assignedAircraft[aircraft][1]] = yBase
                                cumHours[assignedAircraft[aircraft]] = (
                                    cumHours[assignedAircraft[aircraft]] -
                                    elapsedTime - dist2Base/speed)
                            else:
                                # Aircraft will still be on its way at the end
                                remDist2Base = dist2Base - trem*speed
                                xD2B = remDist2Base/base2Fire
                                currLocs[assignedAircraft[aircraft][0]] = (
                                    xBase + xD2B*(xFire - xBase))
                                currLocs[assignedAircraft[aircraft][1]] = (
                                    yBase + xD2B*(yFire - yBase))
                                cumHours[assignedAircraft[aircraft]] = (
                                    cumHours[assignedAircraft[aircraft]] -
                                    timeStep)
                else:
                    # Aircraft will continue its mission
                    # Time elapsed between final visit and end of period
                    distSinceFinalVisit = speed*(timeStep - finalVisitTime)

                    if distSinceFinalVisit > base2Fire:
                        # Aircraft is heading back to fire
                        dist2Base = distSinceFinalVisit - base2Fire

                        xD2B = dist2Base/base2Fire
                        currLocs[assignedAircraft[aircraft][0]] = (
                            xBase + xD2B*(xFire - xBase))
                        currLocs[assignedAircraft[aircraft][1]] = (
                            yBase + xD2B*(yFire - yBase))
                    else:
                        # Aircraft is heading back to base
                        dist2Base = base2Fire - distSinceFinalVisit

                        xD2B = dist2Base/base2Fire
                        currLocs[assignedAircraft[aircraft][0]] = (
                            xBase + xD2B*(xFire - xBase))
                        currLocs[assignedAircraft[aircraft][1]] = (
                            yBase + xD2B*(yFire - yBase))

                    cumHours[assignedAircraft[aircraft]] = (
                        cumHours[assignedAircraft[aircraft]] - timeStep)
            else:
                # The aircraft has not reached the location it was sent to
                if extinguished == True:
                    # Aircraft will head back to base
                    dist2Base = init2Base + speed*elapsedTime

                    if trem*speed >= dist2Base:
                        # Aircraft makes it back in time no matter what
                        currLocs[assignedAircraft[aircraft][0]] = xBase
                        currLocs[assignedAircraft[aircraft][1]] = yBase

                        if elapsedTime - finalVisitTime > base2Fire/speed:
                            # Aircraft is returning to fire so needs to turn around
                            cumHours[assignedAircraft[aircraft]] = (
                                cumHours[assignedAircraft[aircraft]] -
                                elapsedTime -
                                (elapsedTime -
                                 finalVisitTime -
                                 base2Fire/speed))
                        else:
                            # Just let the aircraft continute back to base
                            cumHoursNew[assignedAircraft[aircraft]] = (
                                cumHours[assignedAircraft[aircraft]] -
                                elapsedTime -
                                base2Fire/speed +
                                (elapsedTime - finalVisitTime))

                    else:
                        # Aircraft will not make it back all the way in time
                        xD2B = dist2Base/base2Fire
                        currLocs[assignedAircraft[aircraft][0]] = (
                            xBase + xD2B*(xFire - xBase))
                        currLocs[assignedAircraft[aircraft][1]] = (
                            yBase + xD2B*(yFire - yBase))
                        cumHours[assignedAircraft[aircraft]] = (
                            cumHours[assignedAircraft[aircraft]] - timeStep)
                else:
                    # Aircraft will continue with its mission
                    dist2Base = init2Base + speed*timeStep
                    xD2B = dist2Base/base2Fire
                    currLocs[assignedAircraft[aircraft][0]] = xBase + xD2B*(
                        xFire - xBase)
                    currLocs[assignedAircraft[aircraft][1]] = yBase + xD2B*(
                        yFire - yBase)
                    cumHours[assignedAircraft[aircraft]] = (
                        cumHours[assignedAircraft[aircraft]] - timeStep)

        return damage

    @staticmethod
    def assignNearestAvailable(model, assignments, currLocs, cumHours,
                               resourceTypes, fireLoc, timeToFight):
        # Get available aircraft first
        resources = model.getRegion().getResourceTypes()
        maxDaily = numpy.empty(2)
        speed = numpy.empty(2)
        speed[0] = resources[0].getSpeed()
        speed[1] = resources[1].getSpeed()
        maxDaily[0] = resources[0].getMaxDailyHours()
        maxDaily[1] = resources[1].getMaxDailyHours()
        xfire = fireLoc[0]
        yfire = fireLoc[1]

        # Distances to fire measured in time to fire (only two aircraft atm)
        aircraft2Fire = numpy.divide(numpy.sqrt((xfire - currLocs[:, 0])**2 +
                                                (yfire - currLocs[:, 1])**2),
                                     speed[0]*(resourceTypes == 0) +
                                     speed[1]*(resourceTypes == 1))

        # Tankers
        available = numpy.nonzero(
            numpy.multiply(aircraft2Fire < timeToFight,
                           numpy.multiply(aircraft2Fire + cumHours <
                                          maxDaily[0],
                                          numpy.multiply(
                                              resourceTypes == 0,
                                              assignments[0, :]==0))))
        tankerDists = numpy.array([aircraft2Fire[available], available])
        tankerDistsIdxSorted = numpy.argsort(tankerDists[0, :])
        nearestTanker = tankerDistsIdxSorted[0]

        # Helicopters
        available = numpy.nonzero(
            numpy.multiply(aircraft2Fire < timeToFight,
                           numpy.multiply(aircraft2Fire + cumHours <
                                          maxDaily[1],
                                          numpy.multiply(
                                              resourceTypes == 1,
                                              assignments[0, :] == 0))))

        tankerDists = numpy.array([aircraft2Fire[available], available])
        tankerDistsIdxSorted = numpy.argsort(tankerDists[0, :])
        nearestHeli = tankerDistsIdxSorted[0]

        return [nearestTanker, nearestHeli]

    def aggregateWeatherData(self, ffdi, fires):

        patches = self.model.getRegion().getPatches()
        severityAgg = numpy.zeros(patches)
        lookahead = self.model.getLookahead()
        ffdiAgg = [None]*lookahead

        for t in range(lookahead):
            iterator = 0
            ffdiAgg[t] = numpy.empty(patches)

            for patch in self.model.getRegion().getPatches():
                ffdiAgg[t][iterator] = (sum([ffdi[ii-1]
                                             for ii in patch.getIndices()]) /
                                        len(patch.getIndices()))
                iterator = iterator + 1

        for fire in fires:
            fireX = fire.getLocation()[0]
            fireY = fire.getLocation()[1]

            patchesX = numpy.array([patches[ii].getCentroid()[0]
                                    for ii in range(len(patches))])
            patchesY = numpy.array([patches[ii].getCentroid()[1]
                                    for ii in range(len(patches))])

            dists = numpy.sqrt((patchesX - fireX)**2 + (patchesY - fireY)**2)
            severityAgg[numpy.argsort(dists)[0]] = fire.getSize()

        return [ffdiAgg, severityAgg]

    @staticmethod
    def pathRecomputation(self, t, state_t, maps):
        # Return recomputed VALUES as a vector across the paths
        return 0

    @staticmethod
    def multiLocLinReg(self, predictors, regressors):
        pass

    @staticmethod
    def computeFFDI(temp, rh, wind, df):
        return 2*numpy.exp(-0.45 + 0.987*numpy.log(df) -
                           0.0345*rh + 0.0338*temp + 0.0234*wind)
