# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:34:56 2017

@author: davey
"""

import numpy
from datetime import datetime


class Fire():
    # Class for managing fires

    def __init__(self):
        # Constructs an instance
        self.location = numpy.empty([0, 0])
        self.size = 0.0
        self.start = datetime.now()
        self.end = datetime.now()
        self.initialSize = 0.0
        self.finalSize = 0.0
        self.responseEncoding = ""
        self.respondingStations = []
        self.patchID = 0

    # All locations assumed at centre of grid cells for simplicity
    def getLocation(self):
        return self.location

    def setLocation(self, l):
        self.location = l

    def getSize(self):
        return self.size

    def setSize(self, s):
        self.size = s

    def getStart(self):
        return self.start

    def setStart(self, s):
        self.start = s

    def getEnd(self):
        return self.end

    def setEnd(self, e):
        self.end = e

    def getInitialSize(self):
        return self.initialSize

    def setInitialSize(self, i):
        self.initialSize = i

    def getFinalSize(self):
        return self.finalSize

    def setFinalSize(self, f):
        self.finalSize = f

    def getResponseEncoding(self):
        return self.responseEncoding

    def setResponseEncoding(self, r):
        self.responseEncoding = r

    def getRespondingStations(self):
        return self.respondingStations

    def setRespondingStations(self, r):
        self.respondingStations = r

    def getPatchID(self):
        return self.patchID

    def setPatchID(self, pid):
        self.patchID = pid

    def growFire(self, model, ffdi, configID, random=False):
        vegetation = (model.getRegion().getPatches()[self.patchID]
                .getVegetation())

        grMean = numpy.interp(ffdi,
                              vegetation.getFFDIRange(),
                              vegetation.getROCA2PerHourMean()[configID])

        if random:
            grSD = max(0,
                       numpy.interp(ffdi,
                                vegetation.getFFDIRange(),
                                vegetation.getROCA2PerHourSD()[configID]))
        else:
            grSD = 0

        self.size = self.size*(1 + grMean + grSD)
