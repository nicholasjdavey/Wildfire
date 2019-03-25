# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:10:43 2017

@author: davey
"""

import sys
import math
from Tanker import Tanker
from Heli import Heli
from AirStrip import AirStrip
import numpy
import copy


class Region():
    # Class for defining a study region

    def __init__(self):
        # Constructs an instance
        self.ffdiRange = numpy.empty(0)
        self.patches = []
        self.stations = []
        self.x = numpy.empty([0, 0])
        self.y = numpy.empty([0, 0])
        self.z = numpy.empty([0, 0])
        self.vertices = []
        self.north = numpy.empty([0, 0])
        self.south = numpy.empty([0, 0])
        self.east = numpy.empty([0, 0])
        self.west = numpy.empty([0, 0])
        self.fireSeverity_0 = numpy.empty([0, 0])
        self.fireAge_0 = numpy.empty([0, 0])
        self.dangerIndex_0 = numpy.empty([0, 0])
        self.rain_0 = numpy.empty([0, 0])
        self.humidity_0 = numpy.empty([0, 0])
        self.wind_0 = [numpy.empty([0, 0]), numpy.empty([0, 0])]
        self.temperature_0_min = numpy.empty([0, 0])
        self.temperature_0_max = numpy.empty([0, 0])
        self.vegetation = numpy.empty([0, 0])
        self.windRegime = 0
        self.windN = numpy.empty([0, 0])
        self.windE = numpy.empty([0, 0])
        self.stationDistances = numpy.empty([0, 0])
        self.stationPatchDistances = numpy.empty([0, 0])
        self.weatherGenerator = None
        self.expectedPotentialDamage = []
        self.expectedExistingDamage = []
        self.fires = []
        self.vegetations = []
        self.resources = []
        self.airTankers = []
        self.helicopters = []
        self.firetrucks = []
        self.intermediateBases = [[],[]]
        self.assignments_0 = numpy.empty([0, 0])
        self.name = ""
        self.expectedDamagePotential = numpy.empty([0, 0])
        self.expectedDamageExisting = numpy.empty([0, 0])

    def getFFDIRange(self):
        return self.ffdiRange

    def setFFDIRange(self, r):
        self.ffdiRange = r

    def getPatches(self):
        return self.patches

    def setPatches(self, p):
        self.patches = p

    def getStations(self):
        return self.stations

    def setStations(self, s):
        self.stations = s

    def getX(self):
        return self.x

    def setX(self, x):
        self.x = x

    def getY(self):
        return self.y

    def setY(self, y):
        self.y = y

    def getZ(self):
        return self.z

    def setZ(self, z):
        self.z = z

    def getVertices(self):
        return self.vertices

    def setVertices(self, v):
        self.vertices = v

    def getNorth(self):
        return self.north

    def setNorth(self, n):
        self.north = n

    def getSouth(self):
        return self.south

    def setSouth(self, s):
        self.south = s

    def getEast(self):
        return self.east

    def setEast(self, e):
        self.east = e

    def getWest(self):
        return self.west

    def setWest(self, w):
        self.west = w

    def getFireSeverity(self):
        return self.fireSeverity

    def setFireSeverity(self, s):
        self.fireSeverity = s

    def getFireAge(self):
        return self.fireAge

    def setFireAge(self, a):
        self.fireAge = a

    def getDangerIndex(self):
        return self.dangerIndex_0

    def setDangerIndex(self, di):
        self.dangerIndex_0 = di

    def getRain(self):
        return self.rain_0

    def setRain(self, r):
        self.rain_0 = r

    def getHumidity(self):
        return self.humidity_0

    def setHumidity(self, h):
        self.humidity_0 = h

    def getWind(self):
        return self.wind_0

    def setWind(self, w):
        self.wind_0 = w

    def getTemperatureMin(self):
        return self.temperature_0_min

    def setTemperatureMin(self, t):
        self.temperature_0_min = t

    def getTemperatureMax(self):
        return self.temperature_0_max

    def setTemperatureMax(self, t):
        self.temperature_0_max = t

    def getVegetation(self):
        return self.vegetation

    def setVegetation(self, v):
        self.vegetation = v

    def getStationDistances(self):
        return self.stationDistances

    def setStationDistances(self, d):
        self.stationDistances = d

    def getStationPatchDistances(self):
        return self.stationPatchDistances

    def setStationPatchDistances(self, d):
        self.stationPatchDistances = d

    def getWeatherGenerator(self):
        return self.weatherGenerator

    def setWeatherGenerator(self, wg):
        self.weatherGenerator = wg

    def getExpectedPD(self):
        return self.expectedPotentialDamage

    def setExpectedPD(self, d):
        self.expectedPotentialDamage = d

    def getExpectedED(self):
        return self.expectedExistingDamage

    def setExpectedED(self, d):
        self.expectedExistingDamage = d

    def getFires(self):
        return self.fires

    def setFires(self, f):
        self.fires = f

    def getVegetations(self):
        return self.vegetations

    def setVegetations(self, v):
        self.vegetations = v

    def getResources(self):
        return self.resources

    def setResources(self, r):
        self.resources = r

    def getAirTankers(self):
        return self.airTankers

    def setAirTankers(self, t):
        self.airTankers = t

    def getHelicopters(self):
        return self.helicopters

    def setHelicopters(self, h):
        self.helicopters = h

    def getFiretrucks(self):
        return self.firetrucks

    def setFiretrucks(self, f):
        self.firetrucks = f

    def getIntermediateBases(self):
        return self.intermediateBases

    def setIntermediateBases(self, b):
        self.intermediateBases = b

    def getAssignments(self):
        return self.assignments_0

    def setAssignments(self, a):
        self.assignments_0 = a

    def getName(self):
        return self.name

    def setName(self, n):
        self.name = n

    def getWindRegime(self):
        return self.windRegime

    def setWindRegime(self, r):
        self.windRegime = r

    def getWindN(self):
        return self.windN

    def setWindN(self, w):
        self.windN = w

    def getWindE(self):
        return self.windE

    def setWindE(self, w):
        self.windE = w

    def getExpDE(self):
        return self.expectedDamageExisting

    def setExpDE(self, d):
        self.expectedDamageExisting = d

    def getExpDP(self):
        return self.expectedDamagePotential

    def setExpDP(self, d):
        self.expectedDamagePotential = d

    def configureRegion(self, simulation):
        """ First, get the speed of the slowest aircraft type """
        resourceTypes = simulation.getModel().getResourceTypes()
        bases = copy.copy(self.getStations()[0])

        speeds = [resourceType.getSpeed()
                  for resourceType in resourceTypes
                  if isinstance(resourceType, (Tanker, Heli))]

        speedMin = min(speeds)

        """ Find un-connected sub-graphs and progressively create connecting
        nodes if needed. """

        while True:
            """ Get connectivity of airbases alone """
            connectivity = numpy.array([[0]*len(bases)]*len(bases))

            for ii in range(len(bases)):
                for jj in range(ii, len(bases)):
                    connected = (
                        1
                        if (Region.geoDist(bases[ii].getLocation(),
                                    bases[jj].getLocation())
                            / speedMin < 1.0)
                        else 0)

                    if connected:
                        connectivity[ii, jj] = 1
                        connectivity[jj, ii] = 1

            [connectivity, order] = Region.buildAdjacencyMatrix(connectivity)

            connected = (True if connectivity.sum() == connectivity.size
                         else False)
            if connected:
                break

            """ Build intermediate nodes to complete connections (if needed) """
            intermediateNew = Region.buildIntermediateBases(connectivity,
                                                            order, bases,
                                                            speedMin)
            bases.extend(intermediateNew)
            self.intermediateBases[0].extend(intermediateNew)

    def configureIntermediateFires(self, simulation, activeFires, resources):
        """ We only connect to bases within a 1 hr travel distance or the
        nearest base outside this distance via intermediate nodes. We do not
        consider other possible routes to add. Again, like the base config,
        we can develop more advanced techniques later. """
        interFires = []
        interFiresR0 = []
        interFiresR0ACs = []

        resourceTypes = simulation.getModel().getResourceTypes()
        speeds = [resourceType.getSpeed()
                  for resourceType in resourceTypes]

        speedMin = min(speeds)

        bases = self.stations[0] + self.intermediateBases[0]

        covers = numpy.zeros([len(activeFires), len(bases)])
        distances = numpy.zeros([len(activeFires), len(bases)])

        """ First, let's build intermediate bases from bases to fires (if
        needed) """
        for ii in range(covers.shape[0]):
            for jj in range(covers.shape[1]):
                distances[ii, jj] = (
                        Region.geoDist(activeFires[ii].getLocation(),
                        bases[jj].getLocation()) / speedMin)

                if (distances[ii, jj] <= 1):
                    covers[ii, jj] = 1

            if sum(covers[ii, :]) == 0:
                """ Need to create intermediate nodes """
                newNodes = math.ceil(distances[ii, :].max())
                idx = distances[ii, :].argmax()

                for jj in range(newNodes):
                    xNew = (bases[idx].getLocation()[0]
                            + (jj + 1) * (bases[idx].getLocation()[0]
                                          - activeFires[ii].getLocation()[0])
                            / (newNodes + 1))

                    yNew = (bases[idx].getLocation()[1]
                            + (jj + 1) * (bases[idx].getLocation()[1]
                                          - activeFires[ii].getLocation()[1])
                            / (newNodes + 1))

                    interFire = AirStrip()
                    interFire.setLocation(numpy.array([xNew, yNew]))
                    interFire.setMaxTankers(math.inf)
                    interFire.setMaxHelicopters(math.inf)
                    interFires.append(interFire)

        """ Now, let's build intermediate bases from fires to initial
        aircraft positions (if needed). We will endeavour to create at least
        one direct path to an aircraft for each fire. """
        covers = numpy.zeros([len(activeFires), len(resources)])
        distances = numpy.zeros([len(activeFires), len(resources)])

        for ii in range(covers.shape[0]):
            for jj in range(covers.shape[1]):
                distances[ii, jj] = (
                        Region.geoDist(activeFires[ii].getLocation(),
                        resources[jj].getLocation()) / speedMin)

                if (distances[ii, jj] <= 1):
                    covers[ii, jj] = 1

            if sum(covers[ii, :]) == 0:
                """ Need to create intermediate nodes """
                newNodes = math.ceil(distances[ii, :].min())
                idx = distances[ii, :].argmax()

                for jj in range(newNodes):
                    xNew = (resources[idx].getLocation()[0]
                            + (jj + 1) * (resources[idx].getLocation()[0]
                                          - activeFires[ii].getLocation()[0])
                            / (newNodes + 1))

                    yNew = (resources[idx].getLocation()[1]
                            + (jj + 1) * (resources[idx].getLocation()[1]
                                          - activeFires[ii].getLocation()[1])
                            / (newNodes + 1))

                    interFireR0 = AirStrip()
                    interFireR0.setLocation(numpy.array([xNew, yNew]))
                    interFireR0.setMaxTankers(math.inf)
                    interFireR0.setMaxHelicopters(math.inf)
                    interFiresR0.append(interFireR0)
                    """ This base should only be accessible by this aircraft
                    """
                    interFiresR0ACs.append(idx)

        """ Now, let's build intermediate bases from aircraft to bases (if
        needed). We will endeavour to create at least one direct path from an
        aircraft to a base. """
        covers = numpy.zeros([len(resources), len(bases)])
        distances = numpy.zeros([len(resources), len(bases)])

        for ii in range(covers.shape[0]):
            for jj in range(covers.shape[1]):
                distances[ii, jj] = (
                        Region.geoDist(bases[jj].getLocation(),
                        resources[ii].getLocation()) / speedMin)

                if (distances[ii, jj] <= 1):
                    covers[ii, jj] = 1

            if sum(covers[ii, :]) == 0:
                """ Need to create intermediate nodes """
                newNodes = math.ceil(distances[ii, :].min())
                idx = distances[ii, :].argmax()

                for jj in range(newNodes):
                    xNew = (bases[idx].getLocation()[0]
                            + (jj + 1) * (bases[idx].getLocation()[0]
                                          - resources[ii].getLocation()[0])
                            / (newNodes + 1))

                    yNew = (bases[idx].getLocation()[1]
                            + (jj + 1) * (bases[idx].getLocation()[1]
                                          - resources[ii].getLocation()[1])
                            / (newNodes + 1))

                    interFireR0 = AirStrip()
                    interFireR0.setLocation(numpy.array([xNew, yNew]))
                    interFireR0.setMaxTankers(math.inf)
                    interFireR0.setMaxHelicopters(math.inf)
                    interFiresR0.append(interFireR0)
                    """ This base should only be accessible by this aircraft
                    """
                    interFiresR0ACs.append(idx)

        """ Now let's build the temporary nodes between fires and initial
        aircraft positions """
        return [interFires, interFiresR0, interFiresR0ACs]

    @staticmethod
    def buildIntermediateBases(connectivity, order, bases, speedMin):
        """ At the moment, we only have a very rudimentary, inefficient method
        to connect nodes. A more advanced technique can be introduced later for
        more strategic new nodes. For now, just introduce good bases when
        creating the input data. """
        intermediateNew = Region.buildIntermediateBase(connectivity,
                                                       order, bases, speedMin)

        return intermediateNew

    @staticmethod
    def buildIntermediateBase(connectivity, order, bases, speedMin):
        """ In the interests of time, all we want to do here is to connect the
        nearby clusters of points to other clusters by only one intermediate
        path. Better networks can be built via better designs at a later
        stage. """
        clusters = 1
        starts = [0]
        clusterPoints = [[]]

        for ii in range(connectivity.shape[0]):
            if connectivity[ii, starts[-1]] != 1:
                clusters += 1
                starts.append(ii)
                clusterPoints.append([])
                clusterPoints[-1].append(order[ii])
            else:
                clusterPoints[-1].append(order[ii])

        clusterMinDists = numpy.array([[math.inf]*clusters]*clusters)
        clusterMinPairs = numpy.array([[None]*clusters]*clusters)

        minOverall = math.inf
        minPair = []

        for ii in range(0, clusters):
            for jj in range(ii+1, clusters):
                minDist = math.inf

                for b1 in clusterPoints[ii]:
                    for b2 in clusterPoints[jj]:
                        dist = Region.geoDist(bases[b1].getLocation(),
                                              bases[b2].getLocation())

                        if dist < minDist:
                            minDist = dist
                            clusterMinPairs[ii, jj] = [b1, b2]
                            clusterMinPairs[jj, ii] = [b2, b1]

                            if dist < minOverall:
                                minPair = [b1, b2]
                                minOverall = dist

                clusterMinDists[ii, jj] = minDist
                clusterMinDists[jj, ii] = minDist

        """ Use the new pair to build (an) intermediate node(s) """
        newNodes =  math.floor(minOverall / speedMin)

        for ii in range(newNodes):
            xNew = (bases[minPair[0]].getLocation()[0]
                    + (ii + 1) * (bases[minPair[1]].getLocation()[0]
                                  - bases[minPair[0]].getLocation()[0])
                    / (newNodes + 1))

            yNew = (bases[minPair[0]].getLocation()[1]
                    + (ii + 1) * (bases[minPair[1]].getLocation()[1]
                                  - bases[minPair[0]].getLocation()[1])
                    / (newNodes + 1))

            intermediateNode = AirStrip()
            intermediateNode.setLocation(numpy.array([xNew, yNew]))
            intermediateNode.setMaxTankers(math.inf)
            intermediateNode.setMaxHelicopters(math.inf)

        try:
            return [intermediateNode]
        except:
            return []

    @staticmethod
    def buildAdjacencyMatrix(closeness):
        """ Ready output array """
        adjacency = copy.copy(closeness)

        """ First, complete the connections """
        explored = numpy.zeros(closeness.shape)
        changed = True

        """ Progressively check for connections """
        while changed:
            changed = 0

            """ Iterate over all elements and check for possible additions """
            for ii in range(0, closeness.shape[0]):
                for jj in range(0, closeness.shape[0]):
                    if adjacency[ii, jj] and ii != jj and not explored[ii, jj]:
                        explored[ii, jj] = 1
                        for kk in range(0, closeness.shape[0]):
                            if kk != ii:
                                if adjacency[jj,kk] == 1:
                                    if adjacency[ii,kk] == 0:
                                        adjacency[ii,kk] = 1
                                        adjacency[kk,ii] = 1
                                        changed = 1

        [adjacency, order] = Region.blockDiagonalAdjacency(adjacency)

        return [adjacency, order]

    @staticmethod
    def blockDiagonalAdjacency(matrix):
        """ This only works for symmetric matrices """
        matNew = copy.copy(matrix)
        blocks = 0
        blockStarts = [0]
        orders = [ii for ii in range(0, matrix.shape[0])]
        ii = 0
        newBlock = 0

        while ii < matNew.shape[0]:
            row = matNew[ii]

            if row[blockStarts[blocks]] != 1:
                newBlock = 1
                jj = ii
                while jj+1 < matrix.shape[0] and newBlock:
                    jj += 1
                    if matNew[jj, blockStarts[blocks]] == 1:
                        newBlock = 0

                if not newBlock:
                    """ Pivot rows """
                    tempRow = copy.copy(matNew[jj])
                    matNew[jj] = matNew[ii]
                    matNew[ii] = tempRow
                    tempCol = copy.copy(matNew[:, jj])
                    matNew[:, jj] = matNew[:, ii]
                    matNew[:, ii] = tempCol
                    idx = copy.copy(orders[jj])
                    orders[jj] = orders[ii]
                    orders[ii] = idx
                else:
                    blocks += 1
                    blockStarts.append(ii)
            else:
                matNew[ii] = row

            ii += 1

        return [matNew, orders]

    @staticmethod
    def geoDist(x1d, x2d):
        x1 = [x1d[0] * math.pi/180, x1d[1] * math.pi/180]
        x2 = [x2d[0] * math.pi/180, x2d[1] * math.pi/180]
        a = (math.sin(0.5 * (x2[1] - x1[1])) ** 2
             + math.cos(x1[0]) * math.cos(x2[0])
             * math.sin(0.5 * (x2[0] - x1[0])) ** 2)
        c = math.sqrt(a)
        dist = 2 * 6371 * math.asin(c)
        return dist