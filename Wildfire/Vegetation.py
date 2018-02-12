# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:33:07 2017

@author: davey
"""

import numpy

class Vegetation():
    # Class for defining vegetation types

    def __init__(self):
        # Constructs an instance
        self.name = ""
        self.flammability = 0.0
        self.ffdiRange = numpy.empty([0,0])
        self.occurrence = numpy.empty([0,0])
        self.rocA2PerHour = numpy.empty([0,0])
        self.resourceParams = []
        self.extinguishingSuccess = numpy.empty([0,0])
        
    def getName(self):
        return self.name
        
    def setName(self,n):
        self.name = n
        
    def getFlammability(self):
        return self.flammability
        
    def setFlammability(self,f):
        self.flammability = f
        
    def getFFDIRange(self):
        return self.ffdiRange
        
    def setFFDIRange(self,f):
        self.ffdiRange = f
        
    def getOccurrence(self):
        return self.occurrence
        
    def setOccurrence(self,o):
        self.occurrence = o
        
    def getROCA2PerHour(self):
        return self.rocA2PerHour
        
    def setROCA2PerHour(self,r):
        self.rocA2PerHour = r
        
    def getResourceParams(self):
        return self.resourceParams
        
    def setResourceParams(self,r):
        self.resourceParams = r
        
    def getExtinguishingSuccess(self):
        return self.extinguishingSuccess
        
    def setExtinguishingSuccess(self,e):
        self.extinguishingSuccess = e