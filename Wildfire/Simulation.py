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
        randControls = numpy.random.choice(range(self.model.getControls().size),self.model.getROVPaths()*self.model.getTotalSteps()).reshape(self.model.getROVPaths(),self.model.getTotalSteps()).reshape(self.model.getROVPaths(),self.model.getTotalSteps())

        return randControls

    def endogenousPaths(self,ep,rc):
        # We store the actual fires and their sizes

        paths = []
        cumulativeDamages = []

        [initialMap,initialAss,initialLocs,cumHours0,resourceTypes] = self.initialiseAssignments()

        for path in range(self.model.getROVPaths()):
            [fireSeverity,cumulativeDamage] = self.initialEndogenousPath(ep[path],rc[path],initialMap,initialAss,initialLocs,cumHours0,resourceTypes)
            paths.append(fireSeverity,cumulativeDamage)

        return paths

    def initialForwardPath(self):
        region = self.model.getRegion()
        regionSize = region.getX().size
        timeSteps = self.model.getTotalSteps()

        rain = numpy.empty([timeSteps,regionSize])
        rain[0] = region.getRain()
        precipitation = numpy.empty([timeSteps,regionSize])
        precipitation[0] = region.getHumidity()
        temperatureMin = numpy.empty([timeSteps,regionSize])
        temperatureMin[0] = region.getTemperatureMin()
        temperatureMax = numpy.empty([timeSteps,regionSize])
        temperatureMax[0] = region.getTemperatureMax()
        windNS = numpy.empty([timeSteps,regionSize])
        windNS[0] = region.getWindN()
        windEW = numpy.empty([timeSteps,regionSize])
        windEW[0] = region.getWindE()
        FFDI = numpy.empty([timeSteps,regionSize])
        FFDI[0] = region.getDangerIndex()
        windRegimes = numpy.empty([timeSteps])
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

        for airStrip in self.model.getRegion().getStations()[0]:
            currentTankers = airStrip.getAirTankers()
            currentHelis = airStrip.getHelicopters()

            totalTankers = totalTankers + currentTankers.size
            totalHelis = totalHelis + currentHelis.size

            for tanker in range(len(currentTankers)):
                aircraftPositions.append(airStripIdx)
                resourceTypes.append(0)
                initialLocations.append([airStrip.getLocation()[0],airStrip.getLocation()[1]])

            for heli in range(len(currentHelis)):
                aircraftPositions.append(airStripIdx)
                resourceTypes.append(1)
                initialLocations.append([airStrip.getLocation()[0],airStrip.getLocation()[1]])

            airStripIdx = airStripIdx + 1

        initialBaseAss = numpy.array(aircraftPositions)
        resourceTypes = numpy.array(resourceTypes)
        initialLocations = numpy.array(initialLocations)

        # First column of the assignment is the base assignment. The second is
        # the assignment to fires. Initially, no aircraft are assigned to fires
        # until the location program is run.
        initialFireAss = numpy.zeros(initialBaseAss.size)
        initialAss = numpy.array((initialBaseAss.reshape(initialBaseAss.size,1),initialFireAss.reshape(initialFireAss.size,1)),axis=1)

        # All aircraft start with no hours on the clock
        cumHours = numpy.zeros(totalTankers + totalHelis)

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
            1: maxCover,
            2: pCenter,
            3: assignmentOne,
            4: assignmentTwo
        }
        
        prog = switch.get(locationProgram)
        assignments = prog(randCont,fireSeverityMap,assignmentsCurr,currLocs,cumHoursCurr,resourceTypes,ffdi)

        return assignments

    def maxCover(self,randCont,fireSeverityMap,assignmentsCurr,currLocs,cumHoursCurr,resourceTypes,ffdi):
        # We only consider the maximum cover of 1 tanker and 1 helicopter for now        
        maxCoverDists = numpy.empty(2)
        
        for aircraft in range(2):
            speed = self.model.getResourceTypes()[aircraft].getSpeed()
            maxTime = self.model.getCoverTime()
            maxCoverDists[aircraft] = speed*maxTime
        
        baseNodeSufficient = [None]*2
        baseTPrev = [sum(assignmentsCurr[assignmentsCurr[:,1]==0,0]==jj) for jj in bases]
        baseHPrev = [sum(assignmentsCurr[assignmentsCurr[:,1]==1,0]==jj) for jj in bases]
        
        for aircraft in range(2):
            baseNodeSufficient[aircraft] = self.model.getRegion.getStationPatchDistances()[0] <= maxCoverDists[aircraft]

        # SET UP THE LINEAR PROGRAM (using PuLP for now)
        relocate = LpProblem("Fire Resource Relocation", LpMinimize)
        
        # Decision variables
        # First, names for parameters and variables
        patches = baseNodeSufficient[0].shape[1]
        bases = baseNodeSufficient[0].shape[0]
        totalTankers = sum(numpy.nonzero(assignmentsCurr[:,1]==0))
        totalHelis = sum(numpy.nonzero(assignmentsCurr[:,1]==1))
        lambda1 = self.model.getControls[randCont][0]
        lambda2 = self.model.getControls[randCont][1]
        baseSpacingsTanker = self.model.getRegion.getStationDistances()[0]
        baseSpacingsHeli = self.model.getRegion.getStationDistances()[1]
        
        BasesT = ["BaseT_" + str(ii+1) for ii in range(bases)]
        BasesH = ["BaseH_" + str(ii+1) for ii in range(bases)]
        PatchCovered = ["PatchCover_" + str(ii+1) for ii in range(patches)]
        PatchCoveredT = ["PatchCoverT_" + str(ii+1) for ii in range(patches)]
        PatchCoveredH = ["PatchCoverH_" + str(ii+1) for ii in range(patches)]
        RelocT = [["RelocT_" + str(jj+1) + "_" + str(ii+1) for ii in  range(bases)] for jj in range(bases)]
        RelocH = [["RelocT_" + str(jj+1) + "_" + str(ii+1) for ii in  range(bases)] for jj in range(bases)]
        DSupplyT = ["SupplyT_" + str(ii+1) for ii in range(bases)]
        DSupplyH = ["SupplyH_" + str(ii+1) for ii in range(bases)]
        DDemandT = ["DemandT_" + str(ii+1) for ii in range(bases)]
        DDemandH = ["DemandH_" + str(ii+1) for ii in range(bases)]
        
        baseTVars = [LpVariable(BasesT[ii],lowBound=0,upBound=totalTankers,cat="Integer") for ii in range(bases)]
        baseHVars = [LpVariable(BasesH[ii],lowBound=0,upBound=totalHelis,cat="Integer") for ii in range(bases)]
        patchCoverVars = [LpVariable(PatchCovered[ii],cat="Binary") for ii in range(bases)]
        patchCoverTVars = [LpVariable(PatchCovered[ii],cat="Binary") for ii in range(bases)]
        patchCoverHVars = [LpVariable(PatchCovered[ii],cat="Binary") for ii in range(bases)]
        relocTVars = [[LpVariable(RelocT[ii][jj],lowBound=0,upBound=totalTankers,cat="Integer") for ii in range(bases) for jj in range(bases)]]
        relocHVars = [[LpVariable(RelocH[ii][jj],lowBound=0,upBound=totalHelis,cat="Integer") for ii in range(bases) for jj in range(bases)]]
        supplyTVars = [LpVariable(DSupplyT[ii],lowBound=0,upBound=totalTankers,cat="Integer") for ii in range(bases)]
        supplyHVars = [LpVariable(DSupplyH[ii],lowBound=0,upBound=totalHelis,cat="Integer") for ii in range(bases)]
        demandTVars = [LpVariable(DDemandT[ii],lowBound=0,upBound=totalTankers,cat="Integer") for ii in range(bases)]
        demandHVars = [LpVariable(DDemandH[ii],lowBound=0,upBound=totalHelis,cat="Integer") for ii in range(bases)]

        # Objective        
        relocate += lpSum([lambda1*((1-lambda2)*ffdi[ii] + lambda2*fireSeverityMap[ii])*patchCoverVars[ii] for ii in range(bases)] - (1-lambda1)*[baseSpacingsTanker[ii][jj]*relocTVars[ii][jj] + baseSpacingsHeli[ii][jj]*relocHVars[ii][jj] for ii in range(bases) for jj in range(bases)]), "Total location and relocation cost"
        
        # Constraints
        relocate += lpSum([baseTVars[jj] for jj in range(bases)]) == totalTankers, "Sum of tankers is total"
        relocate += lpSum([baseHVars[jj] for jj in range(bases)]) == totalHelis, "Sum of helicopters is total"
        
        for ii in range(patches):
            relocate += lpSum(patchCoverVars[ii] - 0.5*patchCoverTVars[ii] - 0.5*patchCoverHVars[ii]) <= 0, "Patch " + str(ii) + " covered by both aircraft"
        
        for ii in range(patches):
            relocate += lpSum(patchCoverTVars[ii] - lpSum([baseTVars[jj] for jj in bases])) <= 0, "Patch " + str(ii) + " covered by at least one tanker"
        
        for ii in range(patches):
            relocate += lpSum(patchCoverHVars[ii] - lpSum([baseHVars[jj] for jj in bases])) <= 0, "Patch " + str(ii) + " covered by at least one helicopter"
        
        for jj in range(bases):
            relocate += lpSum(-supplyTVars[jj] + baseTVars[jj] - baseTPrev[jj]) <= 0, "Supply of tankers from base " + str(jj)
            relocate += lpSum(-supplyHVars[jj] + baseHVars[jj] - baseHPrev[jj]) <= 0, "Supply of helicopters from base " + str(jj)
            relocate += lpSum(-demandTvars[jj] - baseTVars[jj] + baseTPrev[jj]) <= 0, "Demand of tankers by base " + str(jj)
            relocate += lpSum(-demandHvars[jj] - baseHVars[jj] + baseHPrev[jj]) <= 0, "Demand of helicopters by base " + str(jj)
        
        for jj in range(bases):
            relocate += lpSum([relocTVars[ii][jj] for ii in range(bases)]) == supplyTVars[jj], "Supply flow conservation for tankers for base " + str(jj)
            relocate += lpSum([relocHVars[ii][jj] for ii in range(bases)]) == supplyHVars[jj], "Supply flow conservation for helicopters for base " + str(jj)
        
        relocate.solve()
        
        # SEND THE ASSIGNMENT OUTPUT TO THE LINEAR PROGRAM
    
    def pCenter(self,randCont,fireSeverityMap,assignmentsCurr,currLocs,cumHoursCurr,resourceTypes,ffdi):
        pass
    
    def assignmentOne(self,randCont,fireSeverityMap,assignmentsCurr,currLocs,cumHoursCurr,resourceTypes,ffdi):
        pass
    
    def assignmentTwo(self,randCont,fireSeverityMap,assignmentsCurr,currLocs,cumHoursCurr,resourceTypes,ffdi):
        pass

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
            init2Fires = numpy.zeros(len(assignedAircraft))
            base2Fires = numpy.zeros(len(assignedAircraft))

            # Simulate attacks
            for aircraft in range(len(assignedAircraft)):
                # Distances to destinations are how far the aircraft are to the
                # assigned fire.
                baseIdx = assignments[assignedAircraft[aircraft],0]
                xInit = currLocs[assignedAircraft[aircraft]][0]
                yInit = currLocs[assignedAircraft[aircraft]][1]
                xBase = self.model.getStations()[0][baseIdx].getLocation()[0]
                yBase = self.model.getStations()[0][baseIdx].getLocation()[1]
                xFire = self.model.getRegion().getPatches()[idx[fire]].getCentroid()[0]
                yFire = self.model.getRegion().getPatches()[idx[fire]].getCentroid()[1]
                init2Fire = math.sqrt((xFire-xInit)**2 + (yFire-yInit)**2)
                base2Fire = math.sqrt((xBase-xFire)**2 + (yBase-yFire)**2)
                speed = resources[resourceTypes[assignedAircraft[aircraft]]].getSpeed()
                init2Fires[aircraft] = init2Fire
                base2Fires[aircraft] = base2Fire

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

                while trem > 0:
                    if 2*base2Fire < speed*trem:
                        visit = [None]*2
                        visit[0] = aircraft
                        visit[1] = 2*base2Fire/speed + visits[len(visits)-1][1]
                        trem = trem - 2*base2Fire/speed
                        visits.append(visit)
                        totalVisits[aircraft] = totalVisits[aircraft] + 1
                        trem = trem - tripTime

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
            # given the success of fighting the fire
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
                                else:
                                    # Aircraft will still be on its way at the end
                                    remDist2Base = dist2Base - trem*speed
                                    xD2B = remDist2Base/base2Fire
                                    currLocsNew[assignedAircraft[aircraft][0]] = xBase + xD2B*(xFire - xBase)
                                    currLocsNew[assignedAircraft[aircraft][1]] = yBase + xD2B*(yFire - yBase)
                            else:
                                # Aircraft is returning to base
                                dist2Base = base2Fire - distSinceFinalVisit

                                if dist2Base/speed < trem:
                                    # Aircraft will make it back to base in time
                                    currLocsNew[assignedAircraft[aircraft][0]] = xBase
                                    currLocsNew[assignedAircraft[aircraft][1]] = yBase
                                else:
                                    # Aircraft will still be on its way at the end
                                    remDist2Base = dist2Base - trem*speed
                                    xD2B = remDist2Base/base2Fire
                                    currLocsNew[assignedAircraft[aircraft][0]] = xBase + xD2B*(xFire - xBase)
                                    currLocsNew[assignedAircraft[aircraft][1]] = yBase + xD2B*(yFire - yBase)
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
                else:
                    # The aircraft has not reached the location it was sent to
                    if extinguished == True:
                        # Aircraft will head back to base
                        dist2Base = init2Base + speed*elapsedTime

                        if trem*speed >= dist2Base:
                            # Aircraft makes it back in time no matter what
                            currLocsNew[assignedAircraft[aircraft][0]] = xBase
                            currLocsNew[assignedAircraft[aircraft][1]] = yBase
                        else:
                            # Aircraft will not make it back all the way in time
                            xD2B = dist2Base/base2Fire
                            currLocsNew[assignedAircraft[aircraft][0]] = xBase + xD2B*(xFire - xBase)
                            currLocsNew[assignedAircraft[aircraft][1]] = yBase + xD2B*(yFire - yBase)
                    else:
                        # Aircraft will continue with its mission
                        dist2Base = init2Base + speed*timeStep
                        xD2B = dist2Base/base2Fire
                        currLocsNew[assignedAircraft[aircraft][0]] = xBase + xD2B*(xFire - xBase)
                        currLocsNew[assignedAircraft[aircraft][1]] = yBase + xD2B*(yFire - yBase)

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
        return [cumHours,fireSeverityMapNew,currLocsNew,damage]

    def pathRecomputation(self,t,state_t,maps):
        # Return recomputed VALUES as a vector across the paths
        return 0

    def multiLocLinReg(self,predictors,regressors):
        pass

    @staticmethod
    def computeFFDI(temp,rh,wind,df):
        return 2*numpy.exp(-0.45+0.987*numpy.log(df)-0.0345*rh+0.0338*temp+0.0234*wind)
