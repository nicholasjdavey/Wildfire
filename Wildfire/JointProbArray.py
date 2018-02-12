# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:32:56 2017

@author: davey
"""

import numpy

class JointProbArray():
    # Class for defining a joint probability multi-dimensional array

    def __init__(self):
        # Constructs an instance
        self.name = ''
        self.params = []
        self.ranges = []
        self.probabilities = numpy.empty([0,0])
    
    def getName(self):
        return self.name
        
    def setName(self,n):
        self.name = n
        
    def getParams(self):
        return self.params
        
    def setParams(self,p):
        self.params = p
        
    def getRanges(self):
        return self.ranges
        
    def setRanges(self,r):
        self.ranges = r
        
    def getProbabilities(self):
        return self.probabilities
        
    def setProbabilities(self,p):
        self.probabilities = p
        
    def interpolate(self,predictors):
        blyat = 0
    