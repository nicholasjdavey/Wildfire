# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:33:16 2017

@author: davey
"""

import numpy

class Patch():
    # Class for defining a homegenous sub-region

    def __init__(self):
        # Constructs an instance
        self.vertices = numpy.empty([0,0])
        self.centriod = numpy.empty([0,0])
        self.area = numpy.empty([0,0])
        self.vegetation = numpy.array([0,0])
        self.averageDanger = []
        self.averageFireSeverity = []
        self.averageTemperature = []
        self.averageHumidity = []
        self.averageWind = []
        self.regionIndices = numpy.array([0,0])

    def getVertices(self):
        return self.vertices

    def setVertices(self,v):
        self.vertices = v

    def getCentroid(self):
        return self.centroid

    def setCentroid(self,c):
        self.centroid = c

    def getArea(self):
        return self.area

    def setArea(self,a):
        self.area = a

    def getAvDanger(self):
        return self.averageDanger

    def setAvDanger(self,d):
        self.averageDanger = d

    def appendDanger(self,d):
        self.averageDanger.append(d)

    def getAvSeverity(self):
        return self.averageFireSeverity

    def setAvSeverity(self,s):
        self.averageFireSeverity = s

    def appendSeverity(self,s):
        self.averageFireSeverity.append(s)

    def getIndices(self):
        return self.indices

    def setIndices(self,i):
        self.indices = i
