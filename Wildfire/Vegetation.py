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
        self.ffdiRange = numpy.empty([0, 0])
        self.occurrence = numpy.empty([0, 0])
        self.rocA2PerHourMean = numpy.empty([0, 0])
        self.rocA2PerHourSD = numpy.empty([0, 0])
        self.initialSuccess = numpy.empty([0, 0])
        self.initialSize = numpy.empty([0, 0])

    def getName(self):
        return self.name

    def setName(self, n):
        self.name = n

    def getFlammability(self):
        return self.flammability

    def setFlammability(self, f):
        self.flammability = f

    def getFFDIRange(self):
        return self.ffdiRange

    def setFFDIRange(self, f):
        self.ffdiRange = f

    def getOccurrence(self):
        return self.occurrence

    def setOccurrence(self, o):
        self.occurrence = o

    def getROCA2PerHourMean(self):
        return self.rocA2PerHourMean

    def setROCA2PerHourMean(self, r):
        self.rocA2PerHourMean = r

    def getROCA2PerHourSD(self):
        return self.rocA2PerHourSD

    def setROCA2PerHourSD(self, r):
        self.rocA2PerHourSD = r

    def getInitialSuccess(self):
        return self.initialSuccess

    def setInitialSuccess(self, e):
        self.initialSuccess = e

    def getInitialSize(self):
        return self.initialSize

    def setInitialSize(self, e):
        self.initialSize = e
