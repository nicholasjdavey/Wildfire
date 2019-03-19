# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:10:43 2017

@author: davey
"""

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
        self.intermediateBases = []
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
        self.intermediateBases = []
        pass

    def configureIntermediateFires(self, simulation, resources):
        interFires = []
        interFiresR0 = []
        return [interFires, interFiresR0]

    @staticmethod
    def buildAdjacencyMatrix(closeness):
        adjacency = copy.copy(closeness)

        """ First, complete the connections """
        changed = 1

        while changed:
            changed = 0

            """ Iterate over all elements and check for possible additions """
            for ii in range(0, closeness.shape[0]):
                for jj in range(0, closeness.shape[0]):
                    if closeness[ii,jj] and ii != jj:
                        for kk in range(0, closeness.shape[0]):
                            if kk != ii:
                                if closeness[jj,kk] == 1:
                                    if closeness[ii,kk] == 0:
                                        closeness[ii,kk] = 1
                                        closeness[kk,ii] = 1
                                        changed = 1


        [adjacency, order] = Region.blockDiagonalAdjacency(adjacency)

    @staticmethod
    def blockDiagonalAdjacency(matrix):
        """ This only works for symmetric matrices """
        matNew = copy.copy(matrix)
        blocks = 0
        blockStarts = [0]
        ii = 0
        newBlock = 0
        while ii < matNew.shape[0]:
            row = matNew[ii]
            if row[blockStarts[blocks]] != 1:
                newBlock = 1
                jj = ii
                while jj+1 < matrix.shape[0] or newBlock:
                    jj += 1
                    if matrix[jj, blockStarts[blocks]] == 1:
                        newBlock = 0
                if not newBlock:
                    print("got here")
                    tempRow = copy.copy(matrix[jj])
                    matrix[jj] = matrix[ii]
                    matrix[ii] = tempRow
                else:
                    blocks += 1
                    blockStarts[blocks] = ii
            else:
                matNew[ii] = row
            ii += 1
            return matNew