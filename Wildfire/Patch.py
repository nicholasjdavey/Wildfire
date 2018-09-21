# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:33:16 2017

@author: davey
"""

import numpy
from Fire import Fire


class Patch():
    # Class for defining a homegenous sub-region
    patches = 0

    def __init__(self):
        # Constructs an instance
        self.vertices = numpy.empty([0, 0])
        self.centroid = numpy.empty([0, 0])
        self.area = 0.0
        self.vegetation = numpy.array([0, 0])
        self.averageDanger = []
        self.averageFireSeverity = []
        self.averageTemperature = []
        self.averageHumidity = []
        self.averageWind = []
        self.regionIndices = numpy.array([0, 0])
        self.patchID = Patch.patches
        Patch.patches += 1

    def getVertices(self):
        return self.vertices

    def setVertices(self, v):
        self.vertices = v

    def getCentroid(self):
        return self.centroid

    def setCentroid(self, c):
        self.centroid = c

    def getArea(self):
        return self.area

    def setArea(self, a):
        self.area = a

    def getAvDanger(self):
        return self.averageDanger

    def setAvDanger(self, d):
        self.averageDanger = d

    def appendDanger(self, d):
        self.averageDanger.append(d)

    def getAvSeverity(self):
        return self.averageFireSeverity

    def setAvSeverity(self, s):
        self.averageFireSeverity = s

    def appendSeverity(self, s):
        self.averageFireSeverity.append(s)

    def getIndices(self):
        return self.indices

    def setIndices(self, i):
        self.indices = i

    def newFires(self, model, ffdi):
        vegetation = model.getRegion().getVegetations()[self.vegetation[0]]

        occ = numpy.interp(ffdi,
                           vegetation.getFFDIRange(),
                           vegetation.getOccurrence())

        newFires = numpy.random.poisson(occ)
        newFiresList = []

        for f in range(len(newFires)):
            size = numpy.interp(ffdi,
                               vegetation.getFFDIRange(),
                               vegetation.getInitialSize())
            fire = Fire.Fire()
            fire.setLocation(self.centroid)
            fire.setSize(size)
            fire.setInitialSize(size)
            fire.setPatchID(self.patchID)
            newFiresList.append(fire)

        return newFiresList

    def growFire(self, model, ffdi, random=False):
        vegetation = model.getRegion().getVegetations()[
                model.getRegion().getPatches()[self.patchID]
                .getVegetations()[0][0]]

        grMean = numpy.interp(ffdi,
                              vegetation.getFFDIRange(),
                              vegetation.getROCA2PerHourMean())

        if random:
            grSD = max(0,
                       numpy.interp(ffdi,
                                vegetation.getFFDIRange(),
                                vegetation.getROCA2PerHourSD()))
        else:
            grSD = 0

        self.size = self.size*(1 + grMean + grSD)