# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:33:16 2017

@author: davey
"""

import numpy
import random
import math
from pyproj import Proj
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
        self.areas = numpy.empty([0, 0])
        self.vegetation = None
        self.averageDanger = []
        self.averageFireSeverity = []
        self.averageTemperature = []
        self.averageHumidity = []
        self.averageWind = []
        self.patchID = Patch.patches
        self.shapefileIndex = 0
        Patch.patches += 1

    def getShapefileIndex(self):
        return self.shapefileIndex

    def setShapefileIndex(self, i):
        self.shapefileIndex = i

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

    def newFires(self, model, ffdi, time, configWeights):
        occ = numpy.interp(ffdi,
                           self.vegetation.getFFDIRange(),
                           self.vegetation.getOccurrence()[time])

        sizeMean = 0.0
        sizeSD = 0.0
        initS = 0.0

        for w, weight in enumerate(configWeights):
            sizeMean += weight*numpy.interp(ffdi,
                                     self.vegetation.getFFDIRange(),
                                     self.vegetation.getInitialSizeMean()[w+1])
            sizeSD += weight*numpy.interp(ffdi,
                                   self.vegetation.getFFDIRange(),
                                   self.vegetation.getInitialSizeSD()[w+1])
            initS += weight*numpy.interp(ffdi,
                                  self.vegetation.getFFDIRange(),
                                  self.vegetation.getInitialSuccess()[w+1])

        newFires = numpy.random.poisson(occ*self.area)
        newFiresList = []

        for f in range(newFires):
            success = True if initS > numpy.random.rand() else False
            size = math.exp(sizeMean + numpy.random.rand() * sizeSD)
            fire = Fire()
            fire.setLocation(self.randomPatchPoint())
            fire.setSize(size)
            fire.setInitialSize(size)
            fire.setFinalSize(size)
            fire.setExtinguished(success)
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

    def randomPatchPointOld(self):
        "Return a point chosen uniformly at random inside polygon."
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


    def randomPatchPoint(self):
        "Return a point chosen uniformly at random inside polygon."
        while True:
            point = (numpy.random.uniform(self.vertices.bounds[0],
                                          self.vertices.bounds[2]),
                     numpy.random.uniform(self.vertices.bounds[1],
                                          self.vertices.bounds[3]))

            if Point(point).within(self.vertices):
                return point

    # DEPRECATED
    def computeArea(self):
        """ Computes the area of a patch based on its vertices """
        self.area = 0.0
        self.areas = numpy.empty(len(self.vertices))

        for ii in range(len(self.vertices)):
            # X is longitude
            lon = [self.vertices[ii][jj][0] for jj in range(len(self.vertices))]
            # Y is latitude
            lat = [self.vertices[ii][jj][1] for jj in range(len(self.vertices))]

            yMax = self.vertices[ii].max()[1]
            yMin = self.vertices[ii].min()[1]
            xMax = self.vertices[ii].max()[0]
            xMin = self.vertices[ii].min()[0]

            pa = Proj("+proj=aea +lat_1=" + str(yMin) + " +lat_2=" + str(yMax)
                      + " +lat_0=" + str((yMin + yMax)*0.5) + " +lon_0="
                      + str((xMin + xMax)*0.5))

            x, y = pa(lon, lat)

            self.areas[ii] = Polygon(zip(x, y)).area
#                    0.5*numpy.abs(numpy.dot(x, numpy.roll(y,1))
#                    -numpy.dot(y,numpy.roll(x,1))))
            self.area += self.areas[ii]

    # DEPRECATED
    def computeCentroid(self):
        """ Computes the centroid of a patch based on its vertices """
        centroids = numpy.empty([len(self.vertices), 2])

        for ii in range(len(self.vertices)):
            ref_polygon = Polygon(self.vertices[ii])

            centroids[ii] = [ref_polygon.centroid.x,
                             ref_polygon.centroid.y]

        self.centroid = numpy.array([
                sum(centroids[ii][0]*self.areas[ii]
                    for ii in range(len(self.vertices)))/
                sum(self.area),
                sum(centroids[ii][1]*self.areas[ii]
                    for ii in range(len(self.vertices)))/
                sum(self.area)])