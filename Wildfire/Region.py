# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:10:43 2017

@author: davey
"""

import numpy
from Process import Process

class Region():
    # Class for defining a study region
    
    def __init__(self):
        # Constructs an instance
        self.patches = []
        self.stations = []
        self.x = numpy.empty([0,0])
        self.y = numpy.empty([0,0])
        self.z = numpy.empty([0,0])
        self.north = numpy.empty([0,0])
        self.south = numpy.empty([0,0])
        self.east = numpy.empty([0,0])
        self.west = numpy.empty([0,0])
        self.fireSeverity_0 = numpy.empty([0,0])
        self.fireAge_0 = numpy.empty([0,0])
        self.dangerIndex_0 = numpy.empty([0,0])
        self.rain_0 = numpy.empty([0,0])
        self.humidity_0 = numpy.empty([0,0])
        self.wind_0 = [numpy.empty([0,0]),numpy.empty([0,0])]
        self.temperature_0 = numpy.empty([0,0])
        self.vegetation = numpy.empty([0,0])
        self.stationDistances = numpy.empty([0,0])
        self.stationPatchDistances = numpy.empty([0,0])
        self.weatherGenerator = None
        self.endArea = None
        self.attackSuccess = None
        self.fires = []
        self.name = ""
        
    def getPatches(self):
        return self.patches
        
    def setPatches(self,p):
        self.patches = p
        
    def getStations(self):
        return self.stations
        
    def setStations(self,s):
        self.stations = s

    def getX(self):
        return self.x

    def setX(self,x):
        self.x = x

    def getY(self):
        return self.y

    def setY(self,y):
        self.y = y

    def getZ(self):
        return self.z

    def setZ(self,z):
        self.z = z

    def getNorth(self):
        return self.north

    def setNorth(self,n):
        self.north = n

    def getSouth(self):
        return self.south

    def setSouth(self,s):
        self.south = s

    def getEast(self):
        return self.east

    def setEast(self,e):
        self.east = e

    def getWest(self):
        return self.west

    def setWest(self,w):
        self.west = w

    def getFireSeverity(self):
        return self.fireSeverity
        
    def setFireSeverity(self,s):
        self.fireSeverity = s

    def getFireAge(self):
        return self.fireAge

    def setFireAge(self,a):
        self.fireAge = a
        
    def getDangerIndex(self):
        return self.dangerIndex_0
        
    def setDangerIndex(self,di):
        self.dangerIndex_0 = di
        
    def getRain(self):
        return self.rain_0
        
    def setRain(self,r):
        self.rain_0 = r
        
    def getHumidity(self):
        return self.humidity_0
        
    def setHumidity(self,h):
        self.humidity_0 = h
        
    def getWind(self):
        return self.wind_0
        
    def setWind(self,w):
        self.wind_0 = w
        
    def getTemperature(self):
        return self.temperature_0
        
    def setTemperature(self,t):
        self.temperature_0 = t
        
    def getVegetation(self):
        return self.vegetation
    
    def setVegetation(self,v):
        self.vegetation = v
        
    def getStationDistances(self):
        return self.stationDistances
        
    def setStationDistances(self,d):
        self.stationDistances =d
        
    def getStationPatchDistances(self):
        return self.stationPatchDistances
        
    def setStationPatchDistances(self,d):
        self.stationPatchDistances = d
        
    def getWeatherGenerator(self):
        return self.weatherGenerator

    def setWeatherGenerator(self,wg):
        self.weatherGenerator = wg
        
    def getEndAreaProbDist(self):
        return self.endArea
        
    def setEndAreaProbDist(self,ea):
        self.endArea = ea
        
    def getAttackSuccessProbDist(self):
        return self.attackSuccess
        
    def setAttackSuccessProbDist(self,succ):
        self.attackSuccess = succ
        
    def getFires(self):
        return self.fires
        
    def setFires(self,f):
        self.fires = f
        
    def getName(self):
        return self.name
        
    def setName(self,n):
        self.name = n
        
