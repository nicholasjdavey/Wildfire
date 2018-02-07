# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:33:36 2017

@author: davey
"""

import numpy

class Station():
    # Class for managing fire-fighting resource sources

    def __init__(self):
        # Constructs an instance
        self.location = numpy.empty([0,0])
        self.capacity = 0.0
        self.engagedFires = []
        self.coverTimes = numpy.empty([0,0])
        self.coveredPatches = []

    def getLocation(self):
        return self.location

    def setLocation(self,loc):
        self.location = loc

    def getCapacity(self):
        return self.capacity

    def setCapacity(self,c):
        self.capacity = c

    def getEngagedFires(self):
        return self.engagedFires

    def setEngagedFires(self,f):
        self.engagedFires = f

    def getCoverTimes(self):
        return self.coverTimes

    def setCoverTimes(self,ct):
        self.coverTimes = ct

    def getCoveredPatches(self):
        return self.coveredPatches

    def setCoveredPatches(self,cp):
        self.coveredPatches = cp
