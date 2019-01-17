# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:34:56 2017

@author: davey
"""

import numpy
import math
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
        self.initialSuccess = False
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

    def getInitialSuccess(self):
        return self.initialSuccess

    def setInitialSuccess(self, i):
        self.initialSuccess = i

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

        grSD = max(0, numpy.interp(ffdi,
                                   vegetation.getFFDIRange(),
                                   vegetation.getROCA2PerHourSD()[configID]))

        success = numpy.interp(ffdi,
                               vegetation.getFFDIRange(),
                               vegetation.getExtendedSuccess()[configID])

        radCurr = (math.sqrt(self.size*10000/math.pi))
        randNo = numpy.random.rand()

        if random:
            radNew = radCurr + math.exp(grMean + grSD * numpy.random.rand())
        else:
            radNew = radCurr + math.exp(grMean + grSD ** 2 / 2)

        # The fire is simply a growing circle. The front progresses from the
        # circumference radially. The growth rate pulled from the vegetation
        # object is in m/hr, so we must convert this to a new size given the
        # current size. Therefore, the current size's radius must be found
        # first.
        self.size = (math.pi * radNew**2)/10000

        return (True if randNo <= success else False)