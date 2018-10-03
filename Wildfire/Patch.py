# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:33:16 2017

@author: davey
"""

import numpy
import random
from shapely.affinity import affine_transform
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate

from Fire import Fire


class Patch():
    # Class for defining a homegenous sub-region
    patches = 0

    def __init__(self):
        # Constructs an instance
        self.vertices = numpy.empty([0, 0])
        self.centroid = numpy.empty([0, 0])
        self.area = 0.0
        self.vegetation = None
        self.averageDanger = []
        self.averageFireSeverity = []
        self.averageTemperature = []
        self.averageHumidity = []
        self.averageWind = []
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

    def getVegetation(self):
        return self.vegetation

    def setVegetation(self, v):
        self.vegetation = v

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

    def newFires(self, model, ffdi, configID):
        occ = numpy.interp(ffdi,
                           self.vegetation.getFFDIRange(),
                           self.vegetation.getOccurrence()[configID])

        newFires = numpy.random.poisson(occ*self.area)
        newFiresList = []

        for f in range(newFires):
            size = numpy.interp(ffdi,
                               self.vegetation.getFFDIRange(),
                               self.vegetation.getInitialSize()[configID])
            fire = Fire()
            fire.setLocation(self.randomPatchPoint())
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

    def randomPatchPoint(self):
        "Return list of k points chosen uniformly at random inside polygon."
        areas = []
        transforms = []
        for t in triangulate(Polygon(self.vertices)):
            areas.append(t.area)
            (x0, y0), (x1, y1), (x2, y2), _ = t.exterior.coords
            transforms.append([x1 - x0, x2 - x0, y2 - y0, y1 - y0, x0, y0])

        weights = [areas[ii]/sum(areas) for ii in range(len(areas))]

        transform = numpy.random.choice(range(len(transforms)), 1, p=weights)
        x, y = [random.random() for _ in range(2)]
        if x + y > 1:
            p = Point(1 - x, 1 - y)
        else:
            p = Point([x, y])
        pointPoints = affine_transform(p, transforms[transform[0]]).coords.xy
        point = [pointPoints[ii][0] for ii in range(2)]

        return point

    def computeArea(self):
        """ Computes the area of a patch based on its vertices """
        x = [self.vertices[ii][0] for ii in range(len(self.vertices))]
        y = [self.vertices[ii][1] for ii in range(len(self.vertices))]

        self.area = (
                0.5*numpy.abs(numpy.dot(x, numpy.roll(y,1))
                -numpy.dot(y,numpy.roll(x,1))))

    def computeCentroid(self):
        """ Computes the centroid of a patch based on its vertices """
        ref_polygon = Polygon(self.vertices)

        self.centroid = [ref_polygon.centroid.x,
                         ref_polygon.centroid.y]