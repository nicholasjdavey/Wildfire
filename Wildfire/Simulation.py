# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:32:32 2017

@author: davey
"""

import numpy
import math
import scipy
from pulp import *

class Simulation():
    # Class for defining a simulation run
    simulations = 0

    def __init__(self):
        # Constructs an instance
        self.fireSeverity = numpy.empty([0,0])
        self.dangerIndex = numpy.empty([0,0])
        self.rain = numpy.empty([0,0])
        self.humidity = numpy.empty([0,0])
        self.wind = numpy.empty([0,0])
        self.temperature = numpy.empty([0,0])
        self.experimentalScenario = None
        self.controls = []
        self.model = None
        self.id = Simulation.simulations
        Simulation.simulations = Simulation.simulations + 1

    def getModel(self):
        return self.model

    def setModel(self,m):
        self.model = m

    def getFireSeverity(self):
        return self.fireSeverity

    def setFireSeverity(self,fs):
        self.fireSeverity(fs)

    def getDangerIndex(self):
        return self.dangerIndex

    def setDangerIndex(self,di):
        self.dangerIndex = di

    def getRain(self):
        return self.rain

    def setRain(self,rain):
        self.rain = rain

    def getHumidity(self):
        return self.humidity

    def setHumidity(self,h):
        self.humidity = h

    def getWind(self):
        return self.wind

    def setWind(self,w):
        self.wind = w

    def getTemperature(self):
        return self.temperature

    def setTemperature(self,t):
        self.temperature = t

    def getExperimentalScenario(self):
        return self.experimentalScenario

    def setExperimentalScenario(self,es):
        self.experimentalScenario = es

    def getControls(self):
        return self.controls

    def setControls(self,c):
        self.controls = c

    def simulate(self):
        # Generate exogenous forward paths for weather and fire starts and save
        # The forward paths are just the Danger Indices at each location
        exogenousPaths = self.forwardPaths()

        # Generate random control paths and store
        randCont = self.randomControls()

        # Generate endogenous fire growth paths given the above and store
        [endogenousPaths,damageMaps] = self.endogenousPaths(exogenousPaths,randCont)

        # Use the Monte Carlo paths to compute the policy maps
        rovMaps = self.rov(exogenousPaths,randCont,endogenousPaths,damageMaps)

    def forwardPaths(self):
        # We don't need to store the precipitation, wind, and temperature matrices
        # over time. We only need the resulting danger index

        paths = []

        for path in range(self.model.getROVPaths()):
            paths.append(self.initialForwardPath())

        return paths

    def rov(self,exogenousPaths,randCont,endogenousPaths):
        pass

    def randomControls(self):
        randControls = numpy.random.choice(range(len(self.model.getControls())),self.model.getROVPaths()*self.model.getTotalSteps()).reshape(self.model.getROVPaths(),self.model.getTotalSteps()).reshape(self.model.getROVPaths(),self.model.getTotalSteps())

        return randControls

    def endogenousPaths(self,ep,rc):
        # We store the actual fires and their sizes

        paths = []
        cumulativeDamages = []

        [initialMap,initialAss,initialLocs,cumHours0,resourceTypes] = self.initialiseAssignments()

        for path in range(self.model.getROVPaths()):
            [fireSeverity,cumulativeDamage] = self.initialEndogenousPath(ep[path],rc[path],initialMap,initialAss,initialLocs,cumHours0,resourceTypes)
            paths.append([fireSeverity,cumulativeDamage])

        return paths

    def initialForwardPath(self):
        region = self.model.getRegion()
        regionSize = region.getX().size
        timeSteps = self.model.getTotalSteps()

        rain = numpy.empty([timeSteps+1,regionSize])
        rain[0] = region.getRain()
        precipitation = numpy.empty([timeSteps+1,regionSize])
        precipitation[0] = region.getHumidity()
        temperatureMin = numpy.empty([timeSteps+1,regionSize])
        temperatureMin[0] = region.getTemperatureMin()
        temperatureMax = numpy.empty([timeSteps+1,regionSize])
        temperatureMax[0] = region.getTemperatureMax()
        windNS = numpy.empty([timeSteps+1,regionSize])
        windNS[0] = region.getWindN()
        windEW = numpy.empty([timeSteps+1,regionSize])
        windEW[0] = region.getWindE()
        FFDI = numpy.empty([timeSteps+1,regionSize])
        FFDI[0] = region.getDangerIndex()
        windRegimes = numpy.empty([timeSteps+1])
        windRegimes[0] = region.getWindRegime()

        wg = region.getWeatherGenerator()

        # Simulate the path forward from time zero to the end
        for ii in range(timeSteps):
            # Compute weather
            wg.computeWeather(rain,precipitation,temperatureMin,temperatureMax,windRegimes,windNS,windEW,FFDI,ii)
            pass

        return FFDI

    def initialEndogenousPath(self,ep,rc,initialMap,initialAss,initialLocs,cumHours0,resourceTypes):
        timeSteps = self.model.getTotalSteps()

        # Keep a whole map of fire in the region. We do not keep track of all
        # the randomly-generated numbers used for fire growth and success (or
        # even first starts for that matter) as it is expected that the fires
        # will have different successes and may even be different fires in the
        # first place. This will save a lot of memory.
        fireSeverityMap = [None]*timeSteps
        assignments = [None]*timeSteps
        currLocs = [None]*timeSteps
        cumHours = [None]*timeSteps
        cumDamage = numpy.empty(timeSteps)
        # Initial cumulative damage is zero (we don't care what happened prior
        # to our study period).
        cumDamage[0] = 0.0
        fireSeverityMap[0] = initialMap
        assignments[0] = initialAss
        currLocs[0] = initialLocs
        cumHours[0] = cumHours0

        for ii in range(timeSteps):
            control = rc[ii]
            locationProgram = self.model.getNestedOptMethod()

            # NESTED OPTIMISATION #############################################
            # Optimise aircraft locations given selected control and state
            assignments[ii] = self.optimalLocations(control,fireSeverityMap[ii],assignments[ii],currLocs[ii],cumHours[ii],resourceTypes,ep[ii],locationProgram)

            # Given the locations found for this control, update the fire
            # severities for the next time period. We use the probabilities. We
            # also compute the new damage incurred and the hours used.
            [cumHours[ii+1],fireSeverityMap[ii+1],currLocs[ii+1],damage] = self.fireSeverity(fireSeverityMap[ii],assignments[ii+1],currLocs[ii+1],cumHours[ii+1],resourceTypes,ep[ii])
            cumDamage[ii+1] = cumDamage[ii] + damage

        return [fireSeverityMap,assignments,currLocs,cumHours,cumDamage]

    def initialiseAssignments(self):
        # Find where all of the fires currently are
        fireSeverityMap = self.model.getRegion().getFireSeverity()

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
                initialLocations.append([airStrip.getLocation()[0],airStrip.getLocation()[1]])
                cumHours.append(resources[0].getMaxDailyHours())

            for heli in range(len(currentHelis)):
                aircraftPositions.append(airStripIdx)
                resourceTypes.append(1)
                initialLocations.append([airStrip.getLocation()[0],airStrip.getLocation()[1]])
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
        initialAss = numpy.array([initialBaseAss,initialFireAss]).transpose()

        return [fireSeverityMap,initialAss,initialLocations,cumHours,resourceTypes]

    def comparator(self,ffdi,time):
        comparators = numpy.empty(self.region.getX().size(),1)
        # Serial
        for ii in range(self.region.getX().size):
            # Linear interpolation. Assume ffdis evenly spaced
            veg = self.region.getVegetation()[ii]
            ffdiRange = self.region.getVegetations[veg].getFFDIRange()
            occurrenceProbs = self.region.getVegetations[veg].getOccurrence()
            ffdis = ffdiRange.size
            ffdiMinIdx = math.floor((ffdi[time][ii]-ffdiRange[0])*(ffdis-1)/(ffdiRange[ffdis] - ffdiRange[0]))
            ffdiMaxIdx = ffdiMinIdx + 1

            if ffdiMinIdx < 0:
                ffdiMinIdx = 0
                ffdiMaxIdx = 1
            elif ffdiMaxIdx >= ffdis:
                ffdiMinIdx = ffdis - 2
                ffdiMaxIdx = ffdis - 1

            xd = (ffdi[time][ii]-ffdiRange[ffdiMinIdx])/(ffdiRange[ffdiMaxIdx] - ffdiRange[ffdiMinIdx])

            comparators[ii] = xd*occurrenceProbs[ffdiMinIdx] + (1-xd)*occurrenceProbs[ffdiMaxIdx]

        return comparators

    def optimalLocations(self,randCont,fireSeverityMap,assignmentsCurr,currLocs,cumHoursCurr,resourceTypes,ffdi,locationProgram):
        switch = {
            0: self.maxCover,
            1: self.pCenter,
            2: self.assignmentOne,
            3: self.assignmentTwo
        }

        prog = switch.get(locationProgram)
        assignments = prog(randCont,fireSeverityMap,assignmentsCurr,currLocs,cumHoursCurr,resourceTypes,ffdi)

        return assignments

    def maxCover(self,randCont,fireSeverityMap,assignmentsCurr,currLocs,cumHoursCurr,resourceTypes,ffdi):
        # We only consider the maximum cover of 1 tanker and 1 helicopter for now
        maxCoverDists = numpy.empty(2)

        for aircraft in range(2):
            speed = self.model.getResourceTypes()[aircraft].getSpeed()
            maxTime = self.model.getCoverTime()/60
            maxCoverDists[aircraft] = speed*maxTime

        # We only get air bases for now
        bases = len(self.model.getRegion().getStations()[0])
        baseNodeSufficient = [None]*2
        baseTPrev = [sum(assignmentsCurr[resourceTypes==0,0]==jj) for jj in range(bases)]
        baseHPrev = [sum(assignmentsCurr[resourceTypes==1,0]==jj) for jj in range(bases)]

        for aircraft in range(2):
            baseNodeSufficient[aircraft] = self.model.getRegion().getStationPatchDistances()[0] <= maxCoverDists[aircraft]

        # SET UP THE LINEAR PROGRAM (using PuLP for now)
        relocate = LpProblem("Fire Resource Relocation", LpMaximize)

        # Decision variables
        # First, names for parameters and variables
        patches = baseNodeSufficient[0].shape[0]
        bases = baseNodeSufficient[0].shape[1]
        totalTankers = numpy.sum(resourceTypes==0)
        totalHelis = numpy.sum(resourceTypes==1)
        lambda1 = self.model.getControls()[randCont].getLambda1()
        lambda2 = self.model.getControls()[randCont].getLambda2()
        baseSpacingsTanker = self.model.getRegion().getStationDistances()[0]
        baseSpacingsHeli = self.model.getRegion().getStationDistances()[0]

        BasesT = ["BaseT_" + str(ii+1) for ii in range(bases)]
        BasesH = ["BaseH_" + str(ii+1) for ii in range(bases)]
        PatchCovered = ["PatchCover_" + str(ii+1) for ii in range(patches)]
        PatchCoveredT = ["PatchCoverT_" + str(ii+1) for ii in range(patches)]
        PatchCoveredH = ["PatchCoverH_" + str(ii+1) for ii in range(patches)]
        RelocT = [["RelocT_" + str(jj+1) + "_" + str(ii+1) for ii in  range(bases)] for jj in range(bases)]
        RelocH = [["RelocH_" + str(jj+1) + "_" + str(ii+1) for ii in  range(bases)] for jj in range(bases)]
        DSupplyT = ["SupplyT_" + str(ii+1) for ii in range(bases)]
        DSupplyH = ["SupplyH_" + str(ii+1) for ii in range(bases)]
        DDemandT = ["DemandT_" + str(ii+1) for ii in range(bases)]
        DDemandH = ["DemandH_" + str(ii+1) for ii in range(bases)]

        baseTVars = [LpVariable(BasesT[ii],lowBound=0,upBound=bases[ii].getMaxTankers(),cat="Integer") for ii in range(bases)]
        baseHVars = [LpVariable(BasesH[ii],lowBound=0,upBound=bases[ii].getMaxHelicopters(),cat="Integer") for ii in range(bases)]
        patchCoverVars = [LpVariable(PatchCovered[ii],cat="Binary") for ii in range(patches)]
        patchCoverTVars = [LpVariable(PatchCoveredT[ii],cat="Binary") for ii in range(patches)]
        patchCoverHVars = [LpVariable(PatchCoveredH[ii],cat="Binary") for ii in range(patches)]
        relocTVars = [[LpVariable(RelocT[ii][jj],lowBound=0,upBound=totalTankers,cat="Integer") for ii in range(bases)] for jj in range(bases)]
        relocHVars = [[LpVariable(RelocH[ii][jj],lowBound=0,upBound=totalHelis,cat="Integer") for ii in range(bases)] for jj in range(bases)]
        supplyTVars = [LpVariable(DSupplyT[ii],lowBound=0,upBound=totalTankers,cat="Integer") for ii in range(bases)]
        supplyHVars = [LpVariable(DSupplyH[ii],lowBound=0,upBound=totalHelis,cat="Integer") for ii in range(bases)]
        demandTVars = [LpVariable(DDemandT[ii],lowBound=0,upBound=totalTankers,cat="Integer") for ii in range(bases)]
        demandHVars = [LpVariable(DDemandH[ii],lowBound=0,upBound=totalHelis,cat="Integer") for ii in range(bases)]

        # The LP uses an aggregation of the weather generator data (FFDI and FireSeverityMap)
        [ffdiAgg,fireSeverityAgg] = self.aggregateWeatherData(ffdi,fireSeverityMap)

        # Objective
        # Due to the aggregated nature of this program, we cannot track individual aircraft.
        # Instead, the objective assumes that for relocation, aircraft are
        # initially at their respective bases (even though this may not be the
        # case in reality).
        relocate += (lpSum([lambda1*((1-lambda2)*ffdiAgg[ii] + lambda2*fireSeverityAgg[ii])*patchCoverVars[ii] for ii in range(patches)]) - lpSum([(1-lambda1)*(baseSpacingsTanker[ii][jj]*relocTVars[ii][jj] + baseSpacingsHeli[ii][jj]*relocHVars[ii][jj]) for ii in range(bases) for jj in range(bases)])), "Total location and relocation cost"
        # Constraints
        relocate += lpSum([baseTVars[jj] for jj in range(bases)]) == totalTankers, "Sum of tankers is total"
        relocate += lpSum([baseHVars[jj] for jj in range(bases)]) == totalHelis, "Sum of helicopters is total"

        for ii in range(patches):
            relocate += (patchCoverVars[ii] - 0.5*patchCoverTVars[ii] - 0.5*patchCoverHVars[ii]) <= 0, "Patch " + str(ii) + " covered by both aircraft"

        for ii in range(patches):
            relocate += (patchCoverTVars[ii] - lpSum([baseTVars[jj]*baseNodeSufficient[0][ii,jj] for jj in range(bases)])) <= 0, "Patch " + str(ii) + " covered by at least one tanker"

        for ii in range(patches):
            relocate += (patchCoverHVars[ii] - lpSum([baseHVars[jj]*baseNodeSufficient[1][ii,jj] for jj in range(bases)])) <= 0, "Patch " + str(ii) + " covered by at least one helicopter"

        # The signs in Chow and Regan (2011), INFOR appear to be the wrong way around. Corrected here.
        for jj in range(bases):
            relocate += (-supplyTVars[jj] - baseTVars[jj] + baseTPrev[jj]) == 0, "Supply of tankers from base " + str(jj)
            relocate += (-supplyHVars[jj] - baseHVars[jj] + baseHPrev[jj]) == 0, "Supply of helicopters from base " + str(jj)
            relocate += (-demandTVars[jj] + baseTVars[jj] - baseTPrev[jj]) == 0, "Demand of tankers by base " + str(jj)
            relocate += (-demandHVars[jj] + baseHVars[jj] - baseHPrev[jj]) == 0, "Demand of helicopters by base " + str(jj)

        for jj in range(bases):
            relocate += lpSum([relocTVars[jj][ii] for ii in range(bases)]) == supplyTVars[jj], "Supply flow conservation for tankers for base " + str(jj)
            relocate += lpSum([relocHVars[jj][ii] for ii in range(bases)]) == supplyHVars[jj], "Supply flow conservation for helicopters for base " + str(jj)
            relocate += lpSum([relocTVars[ii][jj] for ii in range(bases)]) == demandTVars[jj], "Demand flow conservation for tankers for base " + str(jj)
            relocate += lpSum([relocHVars[ii][jj] for ii in range(bases)]) == demandHVars[jj], "Demand flow conservation for helicopters for base " + str(jj)

        relocate.writeLP("Relocation.lp")
        relocate.solve()

        # SEND THE ASSIGNMENT OUTPUT TO THE LINEAR PROGRAM
#        print("Status: ", LpStatus[relocate.status])
#        print("Value: ", value(relocate.objective))

        # Extract the optimal values for the variables
        varsdict= {}
        for var in relocate.variables():
            varsdict[var.name] = var.varValue

        # Base assignments
        baseAss = numpy.empty([bases,2])
        for base in range(bases):
            # Tankers
            baseAss[base,0] = varsdict['BaseT_' + str(base)]
            # Helicopters
            baseAss[base,1] = varsdict['BaseH_' + str(base)]

        # Relocations
        relocs = []
        relocs.append(numpy.empty([bases,bases]))
        relocs.append(numpy.empty([bases,bases]))

        for base1 in range(bases):
            for base2 in range(bases):
                relocs[0][base1][base2] = varsdict["RelocT_" + str(base1) + "_" + str(base2)]
                relocs[1][base1][base2] = varsdict["RelocH_" + str(base1) + "_" + str(base2)]

        assignmentsNew = self.assignmentsHeuristic(assignmentsCurr,fireSeverityMap,cumHoursCurr,currLocs,resourceTypes,baseAss,relocs,lambda2)

        return assignmentsNew

    def pCenter(self,randCont,fireSeverityMap,assignmentsCurr,currLocs,cumHoursCurr,resourceTypes,ffdi):
        # This is the second type of relocation program. Like the Max Cover
        # formulation, we only compute aggregate values, not assignments of
        # individual aircraft.

        # We only get air bases for now
        bases = len(self.model.getRegion().getStations[0])
        baseTPrev = [sum(assignmentsCurr[resourceTypes==0,0]==jj) for jj in range(bases)]
        baseHPrev = [sum(assignmentsCurr[resourceTypes==1,0]==jj) for jj in range(bases)]

        # SET UP THE LINEAR PROGRAM
        relocate = LpProblem("Fire Resource Relocation",LpMinimize)

        # Decision variables
        # First, names for parameters and variables
        patches = len(self.model.getRegion().getPatches())
        bases = len(self.model.getRegion().getStations()[0])
        totalTankers = numpy.sum(resourceTypes==0)
        totalHelis = numpy.sum(resourceTypes==1)
        lambda1 = self.model.getControls()[randCont].getLambda1()
        lambda2 = self.model.getControls()[randCont].getLambda2()
        baseSpacingsTanker = self.model.getRegion().getStationDistances()[0]
        baseSpacingsHeli = self.model.getRegion().getStationDistances()[0]
        basePatchSpacingsTanker = self.model.getRegion().getStationDistances()[0]
        basePatchSpacingsHeli = self.model.getRegion().getStationDistances()[0]

        BasesT = ["BaseT_" + str(ii+1) for ii in range(bases)]
        BasesH = ["BaseH_" + str(ii+1) for ii in range(bases)]
        RelocT = [["RelocT_" + str(jj+1) + "_" + str(ii+1) for ii in  range(bases)] for jj in range(bases)]
        RelocH = [["RelocH_" + str(jj+1) + "_" + str(ii+1) for ii in  range(bases)] for jj in range(bases)]
        DSupplyT = ["SupplyT_" + str(ii+1) for ii in range(bases)]
        DSupplyH = ["SupplyH_" + str(ii+1) for ii in range(bases)]
        DDemandT = ["DemandT_" + str(ii+1) for ii in range(bases)]
        DDemandH = ["DemandH_" + str(ii+1) for ii in range(bases)]

        baseTVars = [LpVariable(BasesT[ii],lowBound=0,upBound=bases[ii].getMaxTankers(),cat="Integer") for ii in range(bases)]
        baseHVars = [LpVariable(BasesH[ii],lowBound=0,upBound=bases[ii].getMaxHelicopters(),cat="Integer") for ii in range(bases)]
        relocTVars = [[LpVariable(RelocT[ii][jj],lowBound=0,upBound=totalTankers,cat="Integer") for ii in range(bases)] for jj in range(bases)]
        relocHVars = [[LpVariable(RelocH[ii][jj],lowBound=0,upBound=totalHelis,cat="Integer") for ii in range(bases)] for jj in range(bases)]
        supplyTVars = [LpVariable(DSupplyT[ii],lowBound=0,upBound=totalTankers,cat="Integer") for ii in range(bases)]
        supplyHVars = [LpVariable(DSupplyH[ii],lowBound=0,upBound=totalHelis,cat="Integer") for ii in range(bases)]
        demandTVars = [LpVariable(DDemandT[ii],lowBound=0,upBound=totalTankers,cat="Integer") for ii in range(bases)]
        demandHVars = [LpVariable(DDemandH[ii],lowBound=0,upBound=totalHelis,cat="Integer") for ii in range(bases)]

        # The LP uses an aggregation of the weather generator data (FFDI and FireSeverityMap)
        [ffdiAgg,fireSeverityAgg] = self.aggregateWeatherData(ffdi,fireSeverityMap)

        # Objective
        # Due to the aggregated nature of this program, we cannot track individual aircraft.
        # Instead, the objective assumes that for relocation, aircraft are
        # initially at their respective bases (even though this may not be the
        # case in reality).
        relocate += (lpSum([lambda1*((1-lambda2)*ffdiAgg[ii] + lambda2*fireSeverityAgg[ii])*(baseTVars[jj]*basePatchSpacingsTanker[ii][jj]+baseHVars[jj]*basePatchSpacingsHeli[ii][jj]) for ii in range(patches) for jj in range(bases)] + [baseSpacingsTanker[ii][jj]*relocTVars[ii][jj] + baseSpacingsHeli[ii][jj]*relocHVars[ii][jj] for ii in range(bases) for jj in range(bases)]))

        # Constraints
        relocate += lpSum([baseTVars[jj] for jj in range(bases)]) == totalTankers, "Sum of tankers is total"
        relocate += lpSum([baseHVars[jj] for jj in range(bases)]) == totalHelis, "Sum of helicopters is total"

        # The signs in Chow and Regan (2011), INFOR appear to be the wrong way around. Corrected here.
        for jj in range(bases):
            relocate += (-supplyTVars[jj] - baseTVars[jj] + baseTPrev[jj]) == 0, "Supply of tankers from base " + str(jj)
            relocate += (-supplyHVars[jj] - baseHVars[jj] + baseHPrev[jj]) == 0, "Supply of helicopters from base " + str(jj)
            relocate += (-demandTVars[jj] + baseTVars[jj] - baseTPrev[jj]) == 0, "Demand of tankers by base " + str(jj)
            relocate += (-demandHVars[jj] + baseHVars[jj] - baseHPrev[jj]) == 0, "Demand of helicopters by base " + str(jj)

        for jj in range(bases):
            relocate += lpSum([relocTVars[jj][ii] for ii in range(bases)]) == supplyTVars[jj], "Supply flow conservation for tankers for base " + str(jj)
            relocate += lpSum([relocHVars[jj][ii] for ii in range(bases)]) == supplyHVars[jj], "Supply flow conservation for helicopters for base " + str(jj)
            relocate += lpSum([relocTVars[ii][jj] for ii in range(bases)]) == demandTVars[jj], "Demand flow conservation for tankers for base " + str(jj)
            relocate += lpSum([relocHVars[ii][jj] for ii in range(bases)]) == demandHVars[jj], "Demand flow conservation for helicopters for base " + str(jj)

        relocate.writeLP("Relocation.lp")
        relocate.solve()

        # Extract the optimal vlaues for the variables
        varsdict = {}
        for var in relocate.variables():
            varsdict[var.name] = var.varValue

        # Base assignments
        baseAss = numpy.empty([bases,2])
        for base in range(bases):
            # Tankers
            baseAss[base,0] = varsdict['BaseT_' + str(base)]
            # Helicopters
            baseAss[base,1] = varsdict['BaseH_' + str(base)]

        # Relocations
        relocs = []
        relocs.append(numpy.empty([bases,bases]))
        relocs.append(numpy.empty([bases,bases]))

        for base1 in range(bases):
            for base2 in range(bases):
                relocs[0][base1][base2] = varsdict["RelocT_" + str(base1) + "_" + str(base2)]
                relocs[1][base1][base2] = varsdict["RelocH_" + str(base1) + "_" + str(base2)]

        assignmentsNew = self.assignmentsHeuristic(assignmentsCurr,fireSeverityMap,cumHoursCurr,currLocs,resourceTypes,baseAss,relocs,lambda2)

        return assignmentsNew

    def assignmentOne(self,randCont,fireSeverityMap,assignmentsCurr,currLocs,cumHoursCurr,resourceTypes,ffdi):
        bases = len(self.model.getRegion().getStations()[0])
        patches = len(self.model.getRegion().getPatches())
        
        # We need threshold for early and late attacks
        maxCoverDists = numpy.empty(2)

        for aircraft in range(2):
            speed = self.model.getResourceTypes()[aircraft].getSpeed()
            maxTime = self.model.getCoverTime()/60
            maxCoverDists[aircraft] = speed*maxTime
        
        # Previous period locations (binary)
        
        # Configurations (any combination of [0,1,2+][e|l][T|H])
        # Arrangement is
        # TE|TL|HE|HL
        submat = numpy.array(range(0,3)).reshape((3,1))
        config = numpy.empty([81,4])
        
        for ii in range(4):
            config[:,ii] = numpy.tile(numpy.repeat(submat,3**(3-ii),axis=0),(3**ii,1)).flatten()

        # Expected damage vs. configuration for each patch (Existing fires, Phi_E)
        # Precomputed values
        # EXISTING FIRES

        # POTENTIAL FIRES

        # Expected damage vs. configuration for each patch (Potential fires, Phi_P)

        # Travel times aircraft to fires

        # Travel times aircraft to bases
        
        pass

    # This is the main assignment problem we solve as it allows us to account for
    # positions of aircraft, their proximities to fires
    def assignmentTwo(self,randCont,fireSeverityMap,assignmentsCurr,currLocs,cumHoursCurr,resourceTypes,ffdi):
        pass

    def assignmentsHeuristic(self,assignmentCurr,fireSeverityMap,cumHoursCurr,currLocs,resourceTypes,baseAssignments,relocs,lambda2):
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
        idx = numpy.nonzero(fireSeverityMap)
        bases = len(self.model.getRegion().getStations()[0])

        # Compute the distances between aircraft and the fires. We only want the
        # closest aircraft with enough hours to fight the fire
        dists2Fire = self.currPos2Fire(currLocs,fireSeverityMap)

        # Next, sort the fires by severity and start assigning aircraft to them
        # in order (i.e. fill demand for most severe fires first)
        sortedIdx = numpy.argsort(idx)[::-1]

        remainingMask = numpy.ones(len(resourceTypes))
        resources = self.model.getRegion().getResourceTypes()
        speeds = (resourceTypes == 0)*resources[0].getSpeed() + (resourceTypes == 1)*resources[1].getSpeed()

        for sidx in sortedIdx:
            tankers = numpy.argsort(dists2Fire[sortedIdx[sidx],numpy.multiply(remainingMask,resourcesType == 0)]*(numpy.divide(dists2Fire[sortedIdx[sidx],:],speeds) < cumHours))
            helicopters = numpy.argsort(dists2Fire[sortedIdx[sidx],numpy.multiply(remainingMask,resourcesType == 1)]*(numpy.divide(dists2Fire[sortedIdx[sidx],:],speeds) < cumHours))
            assignmentsNew[tankers[0:min(len(tankers),kMax)],1] = sortedIdx[sidx] + 1
            assignmentsNew[helicopters[0:min(len(tankers),kMax)],1] = sortedIdx[sidx] + 1
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
                distB1B2 = self.model.getRegion().getStationDistances()[0][base1,base2]
                # Number of aircraft from base1 to base2
                # TANKERS
                noSwap = reloc[0][base1][base2]
                # Tankers initially at the base that are free to move
                initB1Free = numpy.nonzero(numpy.multiply(resourceTypes == 0,numpy.multiply(assignmentsCurr[:,0] == base1,cumHours >= numpy.divide(distB1B2,speeds))))
                # Tankers initially at the base that are NOT free to move
                initB1Fixed = numpy.nonzero(numpy.multiply(resourceTypes == 0,numpy.multiply(assignmentsCurr[:,0] == base1,cumHours < numpy.divide(distB1B2,speeds))))
                mask = numpy.zeros(len(assignmentsCurr[:,0]))

                mask[initB1Free] = 1
                # Tankers sorted by remaining hours, with only sufficient
                # remaining hours allowed to relocate
                toRelocate = numpy.argsort(numpy.multiply(cumHours,mask))[::-1]
                noReloc = min(len(numpy.nonzero(toRelocate)),noSwap)

                # Assign the aircraft to the bases
                assignmentsNew[toRelocate[0:noReloc],0] = base2
                assignmentsNew[initB1Fixed,0] = base1

                # HELICOPTERS
                noSwap = reloc[1][base1][base2]
                # Helicopters initially at the base that are free to move
                initB1Free = numpy.nonzero(numpy.multiply(resourceTypes == 1,numpy.multiply(assignmentsCurr[:,0] == base1,cumHours >= numpy.divide(distB1B2,speeds))))
                # Helicopters initially at the base that are NOT free to move
                initB1Fixed = numpy.nonzero(numpy.multiply(resourceTypes == 1,numpy.multiply(assignmentsCurr[:,0] == base1,cumHours < numpy.divide(distB1B2,speeds))))
                mask = numpy.zeros(len(assignmentsCurr[:,0]))

                mask[initB1Free] = 1
                # Tankers sorted by remaining hours, with only sufficient
                # remaining hours allowed to relocate
                toRelocate = numpy.argsort(numpy.multiply(cumHours,mask))[::-1]
                noReloc = min(len(numpy.nonzero(toRelocate)),noSwap)

                # Assign the aircraft to the bases
                assignmentsNew[toRelocate[0:noReloc],0] = base2
                assignmentsNew[initB1Fixed,0] = base1

        # We can return the assignments now
        return assignmentsNew

    def assingmentsSub1(self,assignmentsCurr,fireSeverityMap,cumHoursCurr,currLocs,resourceTypes,baseAssignments,relocs,lambda2):
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
            travelSpeeds[aircraft] = self.modelGetResourceTypes()[aircraft].getSpeed()

        # Compute the distances between aircraft and the fires. We only want the
        # closest aircraft with enough hours to fight the fire
        dists2Fire = self.currPos2Fire(currLocs,fireSeverityMap)

        # First, we take the relocations found already to determine which
        # aircraft from each base to relocate to the corresponding new bases.
        # We use the relocation variables from earlier. The relocation problem
        # seeks to maximise the remaining travel hours of aircraft for the
        # purpose of relocation (note, this may not be the end result as
        # aircraft will also fight fires, which may require more than one
        # attack. This is uncertain.)
        assignment = LpProblem("Assignment of aircraft to fires",LpMinimize)

        # Objective

        # Next, aircraft are allocated to fires so as to minimise travel time
        # for the first pass.
        # Max aircraft to be allocated to any fire (we upper limit set as 5 of
        # each type for now with a lower limit of 1). This is arbitrary, then
        # again, the Max Cover and P-Median formulations are arbitrary in
        # comparison to the assignment problems as they do not take into
        # consideration actual fire damage but only proxies.
        kMax = lambda2*4+1

        assignment = LpProblem("Assignment of aircraft to bases",LpMinimize)

        return assignmentsNew

    def currPos2Fire(self,currLocs,fireSeverityMap):
        # Computes the distance between each aircraft and each fire
        fires = numpy.nonzero(fireSeverityMap)

        xFire = numpy.array([self.model.getRegion().getX()[fires[fire]] for fire in range(fires)])
        yFire = numpy.array([self.model.getRegion().getY()[fires[fire]] for fire in range(fires)])
        xAircraft = numpy.array([currLocs[aircraft][0] for aircraft in currlocs.shape[0]])
        yAircraft = numpy.array([currLocs[aircraft][1] for aircraft in currlocs.shape[0]])

        noFires = xFire.shape
        noAircraft = xAircraft.shape

        X1 = numpy.tile(xFire.transpose(),(noAircraft[0],1))
        X2 = numpy.tile(xAircraft,(noFires[0],1)).transpose()
        Y1 = numpy.tile(yFire.transpose(),(noAircraft[0],1))
        Y2 = numpy.tile(yAircraft,(noFires[0],1)).transpose()

        dists2Fire = numpy.sqrt((X1-X2)**2+(Y1-Y2)**2)

        return dists2Fire

    def fireSeverity(self,fireSeverityMap,assignments,currLocs,cumHours,resourceTypes,ffdi):
        # This routine performs the simulation for a single time step

        # First, compute the extinguishing success for each fire given the allocations
        idx = numpy.nonzero(fireSeverityMap)
        timeStep = self.model.getStepSize() # In hours
        fireSeverityMapNew = numpy.copy(fireSeverityMap)
        endAttackTime = numpy.zeros(len(fireSeverityMap))+timeStep
        damage = 0.0
        currLocsNew = numpy.copy(currLocs)

        for fire in range(len(idx)):
            # Find out the aircraft assigned to this fire
            assignedAircraft = numpy.nonzero(assignments[:,1] == fire)

            # Get the probabilities of success for each aircraft type
            severity = fireSeverityMap[idx[fire]]
            veg = self.model.getRegion().getVegetation()[idx[fire]]
            ffdiRange = self.model.getRegion().getVegetations[veg].getFFDIRange()
            rocMeanRange = self.model.getRegion().getVegetations[veg].getROCA2PerHour()[0]
            rocSDRange = self.model.getRegion().getVegetations[veg].getROCA2PerHour()[1]
            succTankerRange = self.model.getRegion().getVegetations[veg].getExtinguishingSuccess()[0]
            succHeliRange = self.model.getRegion().getVegetations[veg].getExtinguishingSuccess()[1]
            resources = self.model.getRegion().getResourceTypes()
            ffdis = ffdiRange.size
            ffdiMinIdx = math.floor((ffdi[idx[fire]]-ffdiRange[0])*(ffdis-1)/(ffdiRange[ffdis] - ffdiRange[0]))
            ffdiMaxIdx = ffdiMinIdx + 1

            if ffdiMinIdx < 0:
                ffdiMinIdx = 0
                ffdiMaxIdx = 1
            elif ffdiMaxIdx >= ffdis:
                ffdiMinIdx = ffdis - 2
                ffdiMaxIdx = ffdis - 1

            xd = (ffdi[idx[fire]]-ffdiRange[ffdiMinIdx])/(ffdiRange[ffdiMaxIdx] - ffdiRange[ffdiMinIdx])

            # Rate of change per hour
            rocMean = xd*rocMeanRange[ffdiMinIdx] + (1-xd)*rocMeanRange[ffdiMaxIdx]
            rocSD = xd*rocSDRange[ffdiMinIdx] + (1-xd)*rocSDRange[ffdiMaxIdx]

            # Extinguishing success for each type of aircraft. We only have
            # tankers and helicopters for now.
            svs = numpy.empty(2)
            svs[0] = xd*succTankerRange[ffdiMinIdx] + (1-xd)*succTankerRange[ffdiMaxIdx]
            svs[1] = xd*succHeliRange[ffdiMinIdx] + (1-xd)*succHeliRange[ffdiMaxIdx]

            # Figure out how many times each aircraft assigned to the fire will
            # visit
            visits = []
            totalVisits = numpy.zeros(len(assignedAircraft))
            #init2Fires = numpy.zeros(len(assignedAircraft))
            #base2Fires = numpy.zeros(len(assignedAircraft))

            # Simulate attacks
            for aircraft in range(len(assignedAircraft)):
                # Distances to destinations are how far the aircraft are to the
                # assigned fire.
                # Aircraft can slightly overshoot the remaining number of hours
                # but they cannot leave for a new attack if they have already
                # done so.
                baseIdx = assignments[assignedAircraft[aircraft],0]
                remHours = resources[resourceTypes[assignedAircraft[aircraft]]].getMaxDailyHours() - cumHours[assignedAircraft[aircraft]]
                xInit = currLocs[assignedAircraft[aircraft]][0]
                yInit = currLocs[assignedAircraft[aircraft]][1]
                xBase = self.model.getStations()[0][baseIdx].getLocation()[0]
                yBase = self.model.getStations()[0][baseIdx].getLocation()[1]
                #yFire = self.model.getRegion().getPatches()[idx[fire]].getCentroid()[1]
                xFire = self.model.getRegion().getX()[idx[fire]]
                yFire = self.model.getRegion().getY()[idx[fire]]
                init2Fire = math.sqrt((xFire-xInit)**2 + (yFire-yInit)**2)
                base2Fire = math.sqrt((xBase-xFire)**2 + (yBase-yFire)**2)
                speed = resources[resourceTypes[assignedAircraft[aircraft]]].getSpeed()
                #init2Fires[aircraft] = init2Fire
                #base2Fires[aircraft] = base2Fire

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
            visits = visits[visits[:,1].argsort()]

            extinguished = False
            elapsedTime = 0.0
            visit = 0
            totalVisits = len(visits)

            while not(extinguished) and elapsedTime < timeStep and visit < totalVisits:
                timeInterval = visits[visit,1] - elapsedTime
                aircraft = int(visits[visit,0])

                # Compute growth of fire up until the visit as well as the
                # cumulative damage incurred
                severityOld = severity
                severity = severity*numpy.random.normal(rocMean,rocSD)*timeInterval
                damage = damage + (severity - severityOld)
                elapsedTime = elapsedTime + timeInterval

                # Compute whether the fire is extinguished by the visit
                if numpy.random.uniform(0,1) < svs[resourceTypes[aircraft]]:
                    extinguished = True
                    fireSeverityMapNew[idx[fire]] = severity
                    endAttackTime[idx[fire]] = elapsedTime

            if not(extinguished):
                elapsedTime = timeStep

            # Time available to relocate back to base from extinguishing
            trem = timeStep - elapsedTime

            # Compute the final positions of all aircraft for this time period
            # given the success of fighting the fire. Also record the number of
            # flying hours used up.
            cumHoursNew = numpy.copy(cumHours)
            for aircraft in range(len(assignedAircraft)):
                # Find location at time of extinguishing
                visitIdxes = numpy.nonzero(visits[:,0] <= elapsedTime)
                baseIdx = assignments[assignedAircraft[aircraft],0]
                xInit = currLocs[assignedAircraft[aircraft]][0]
                yInit = currLocs[assignedAircraft[aircraft]][1]
                xBase = self.model.getStations()[0][baseIdx].getLocation()[0]
                yBase = self.model.getStations()[0][baseIdx].getLocation()[1]
                xFire = self.model.getRegion().getPatches()[idx[fire]].getCentroid()[0]
                yFire = self.model.getRegion().getPatches()[idx[fire]].getCentroid()[1]
                init2Fire = math.sqrt((xFire-xInit)**2 + (yFire-yInit)**2)
                base2Fire = math.sqrt((xBase-xFire)**2 + (yBase-yFire)**2)
                init2Base = math.sqrt((xInit-xBase)**2 + (yInit-yBase)**2)
                speed = resources[resourceTypes[assignedAircraft[aircraft]]].getSpeed()

                if len(visitIdxes) > 0:
                    # See how long it was since the last visit to determine
                    # location
                    finalVisitTime = visits[visitIdxes[len(visitIdxes)-1][1]]

                    if extinguished == True:
                        # Aircraft will return to base
                        if trem*speed >= base2Fire:
                            # Aircraft diverts back to base immediately and
                            # makes it there before the next period no matter
                            # its position
                            currLocsNew[assignedAircraft[aircraft][0]] = xBase
                            currLocsNew[assignedAircraft[aircraft][1]] = yBase
                            if elapsedTime - finalVisitTime > base2Fire/speed:
                                # Aircraft is returning to fire so needs to turn around
                                cumHoursNew[assignedAircraft[aircraft]] = cumHours[assignedAircraft[aircraft]] - elapsedTime - (elapsedTime - finalVisitTime - base2Fire/speed)
                            else:
                                # Just let the aircraft continute back to base
                                cumHoursNew[assignedAircraft[aircraft]] = cumHours[assignedAircraft[aircraft]] - elapsedTime - base2Fire/speed + (elapsedTime - finalVisitTime)
                        else:
                            # Position of aircraft at time of extinguishing
                            distSinceFinalVisit = speed*(elapsedTime - finalVisitTime)

                            if distSinceFinalVisit > base2Fire:
                                # Aircraft is heading to fire at the end
                                dist2Base = distSinceFinalVisit - base2Fire

                                if dist2Base/speed < trem:
                                    # Aircraft will make it back to base in time
                                    currLocsNew[assignedAircraft[aircraft][0]] = xBase
                                    currLocsNew[assignedAircraft[aircraft][1]] = yBase
                                    cumHoursNew[assignedAircraft[aircraft]] = cumHours[assignedAircraft[aircraft]] - elapsedTime - dist2Base/speed
                                else:
                                    # Aircraft will still be on its way at the end
                                    remDist2Base = dist2Base - trem*speed
                                    xD2B = remDist2Base/base2Fire
                                    currLocsNew[assignedAircraft[aircraft][0]] = xBase + xD2B*(xFire - xBase)
                                    currLocsNew[assignedAircraft[aircraft][1]] = yBase + xD2B*(yFire - yBase)
                                    cumHoursNew[assignedAircraft[aircraft]] = cumHours[assignedAircraft[aircraft]] - timeStep
                            else:
                                # Aircraft is returning to base
                                dist2Base = base2Fire - distSinceFinalVisit

                                if dist2Base/speed < trem:
                                    # Aircraft will make it back to base in time
                                    currLocsNew[assignedAircraft[aircraft][0]] = xBase
                                    currLocsNew[assignedAircraft[aircraft][1]] = yBase
                                    cumHoursNew[assignedAircraft[aircraft]] = cumHours[assignedAircraft[aircraft]] - elapsedTime - dist2Base/speed
                                else:
                                    # Aircraft will still be on its way at the end
                                    remDist2Base = dist2Base - trem*speed
                                    xD2B = remDist2Base/base2Fire
                                    currLocsNew[assignedAircraft[aircraft][0]] = xBase + xD2B*(xFire - xBase)
                                    currLocsNew[assignedAircraft[aircraft][1]] = yBase + xD2B*(yFire - yBase)
                                    cumHoursNew[assignedAircraft[aircraft]] = cumHours[assignedAircraft[aircraft]] - timeStep
                    else:
                        # Aircraft will continue its mission
                        # Time elapsed between final visit and end of period
                        distSinceFinalVisit = speed*(timeStep - finalVisitTime)

                        if distSinceFinalVisit > base2Fire:
                            # Aircraft is heading back to fire
                            dist2Base = distSinceFinalVisit - base2Fire

                            xD2B = dist2Base/base2Fire
                            currLocsNew[assignedAircraft[aircraft][0]] = xBase + xD2B*(xFire - xBase)
                            currLocsNew[assignedAircraft[aircraft][1]] = yBase + xD2B*(yFire - yBase)
                        else:
                            # Aircraft is heading back to base
                            dist2Base = base2Fire - distSinceFinalVisit

                            xD2B = dist2Base/base2Fire
                            currLocsNew[assignedAircraft[aircraft][0]] = xBase + xD2B*(xFire - xBase)
                            currLocsNew[assignedAircraft[aircraft][1]] = yBase + xD2B*(yFire - yBase)

                        cumHoursNew[assignedAircraft[aircraft]] = cumHours[assignedAircraft[aircraft]] - timeStep
                else:
                    # The aircraft has not reached the location it was sent to
                    if extinguished == True:
                        # Aircraft will head back to base
                        dist2Base = init2Base + speed*elapsedTime

                        if trem*speed >= dist2Base:
                            # Aircraft makes it back in time no matter what
                            currLocsNew[assignedAircraft[aircraft][0]] = xBase
                            currLocsNew[assignedAircraft[aircraft][1]] = yBase

                            if elapsedTime - finalVisitTime > base2Fire/speed:
                                # Aircraft is returning to fire so needs to turn around
                                cumHoursNew[assignedAircraft[aircraft]] = cumHours[assignedAircraft[aircraft]] - elapsedTime - (elapsedTime - finalVisitTime - base2Fire/speed)
                            else:
                                # Just let the aircraft continute back to base
                                cumHoursNew[assignedAircraft[aircraft]] = cumHours[assignedAircraft[aircraft]] - elapsedTime - base2Fire/speed + (elapsedTime - finalVisitTime)

                        else:
                            # Aircraft will not make it back all the way in time
                            xD2B = dist2Base/base2Fire
                            currLocsNew[assignedAircraft[aircraft][0]] = xBase + xD2B*(xFire - xBase)
                            currLocsNew[assignedAircraft[aircraft][1]] = yBase + xD2B*(yFire - yBase)
                            cumHoursNew[assignedAircraft[aircraft]] = cumHours[assignedAircraft[aircraft]] - timeStep
                    else:
                        # Aircraft will continue with its mission
                        dist2Base = init2Base + speed*timeStep
                        xD2B = dist2Base/base2Fire
                        currLocsNew[assignedAircraft[aircraft][0]] = xBase + xD2B*(xFire - xBase)
                        currLocsNew[assignedAircraft[aircraft][1]] = yBase + xD2B*(yFire - yBase)
                        cumHoursNew[assignedAircraft[aircraft]] = cumHours[assignedAircraft[aircraft]] - timeStep

        # We have computed the new locations for the aircraft assigned to fires
        # Now we need to determine the positions of the aircraft that were not
        # assigned to fires. They may have been in the process of relocating.
        idleAircraft = numpy.nonzero(assignments[:,1] == 0)

        for aircraft in idleAircraft:
            # If the distance to the assigned base is less than the distance the
            # aircraft can travel in this period, move to the base and update
            # its travel hours.
            baseIdx = assignments[idleAircraft[aircraft],0]
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
                cumHoursNew[assignedAircraft[aircraft]] = cumHours[assignedAircraft[aircraft]] - curr2Base/speed
            else:
                # Aircraft is still on its way
                remDist2Base = curr2Base - trem*speed
                xD2B = remDist2Base/curr2Base
                currLocsNew[idleAircraft[aircraft][0]] = xBase + xD2B*(xInit - xBase)
                currLocsNew[idleAircraft[aircraft][1]] = yBase + xD2B*(yInit - yBase)
                cumHoursNew[assignedAircraft[aircraft]] = cumHours[assignedAircraft[aircraft]] - timeStep

        # Next, compute the creation of new fires. We only compute these at the
        # end of the time step as it they do not influence the current aircraft
        # allocations in the current problem formulation. This can be changed
        # as more realism is added into the program (for future publication).
        # These fires go unchecked.
        occurrenceProbPatches = numpy.array(len(self.model.getRegion().getVegetations()))
        fireGrowthRatePatches = numpy.array(len(self.model.getRegion().getVegetations()))

        for patch in range(len(self.model.getRegion().getPatches())):
            veg = self.model.getRegion().getVegetation()[patch]
            occurrenceProbsRange = self.model.getRegion().getVegetation()[veg].getOccurrence()
            rocMeanRange = self.model.getRegion().getVegetations[veg].getROCA2PerHour()[0]
            rocSDRange = self.model.getRegion().getVegetations[veg].getROCA2PerHour()[1]
            ffdiRange = self.model.getRegion().getVegetations[veg].getFFDIRange()
            ffdis = ffdiRange.size
            ffdiMinIdx = math.floor((ffdi[patch]-ffdiRange[0])*(ffdis-1)/(ffdiRange[ffdis] - ffdiRange[0]))
            ffdiMaxIdx = ffdiMinIdx + 1

            if ffdiMinIdx < 0:
                ffdiMinIdx = 0
                ffdiMaxIdx = 1
            elif ffdiMaxIdx >= ffdis:
                ffdiMinIdx = ffdis - 2
                ffdiMaxIdx = ffdis - 1

            xd = (ffdi[patch]-ffdiRange[ffdiMinIdx])/(ffdiRange[ffdiMaxIdx] - ffdiRange[ffdiMinIdx])
            occurrenceProbPatches[patch] = xd*occurrenceProbsRange[ffdiMinIdx] + (1-xd)*occurrenceProbsRange[ffdiMaxIdx]
            rocMean = xd*rocMeanRange[ffdiMinIdx] + (1-xd)*rocMeanRange[ffdiMaxIdx]
            rocSD = xd*rocSDRange[ffdiMinIdx] + (1-xd)*rocSDRange[ffdiMaxIdx]
            fireGrowthRatePatches[patch] = numpy.random.normal(rocMean,rocSD)

        # Using these probabilities, compute the chances of fires occurring and
        # the time at which they occur (if a fire already exists in a patch,
        # it is assumed that the resources assigned to it are not diverted to
        # also fight this fire.
        newFires = float(occurrenceProbPatches > numpy.random.uniform(0,1,len(occurrenceProbPatches)))
        fireStarts = numpy.zeros(len(occurrenceProbPatches))
        starts = numpy.random.uniform(0,1,int(sum(newFires)))*timeStep
        fireStarts[numpy.nonzero(fireStarts)] = starts

        additionalSeverity = numpy.multiply(numpy.multiply((timeStep - fireStarts),fireGrowthRatePatches),(fireStarts > 0.0))

        fireSeverityMapNew = fireSeverityMapNew + additionalSeverity
        damage = damage + sum(additionalSeverity)

        # Use the times at which the fires occur to determine the damage and
        # fire size by the end of the period. Add to the existing severity
        # matrix and recorded damage
        # Return the results
        return [cumHoursNew,fireSeverityMapNew,currLocsNew,damage]

    def aggregateWeatherData(self,ffdi,fireSeverityMap):

        patches = len(self.model.getRegion().getPatches())
        ffdiAgg = numpy.empty(patches)
        severityAgg = numpy.empty(patches)

        iterator = 0
        for patch in self.model.getRegion().getPatches():
            ffdiAgg[iterator] = sum([ffdi[ii-1] for ii in patch.getIndices()])/len(patch.getIndices())
            severityAgg[iterator] = sum([fireSeverityMap[ii-1] for ii in patch.getIndices()])/len(patch.getIndices())
            iterator = iterator + 1

        return [ffdiAgg,severityAgg]

    def pathRecomputation(self,t,state_t,maps):
        # Return recomputed VALUES as a vector across the paths
        return 0

    def multiLocLinReg(self,predictors,regressors):
        pass

    @staticmethod
    def computeFFDI(temp,rh,wind,df):
        return 2*numpy.exp(-0.45+0.987*numpy.log(df)-0.0345*rh+0.0338*temp+0.0234*wind)
