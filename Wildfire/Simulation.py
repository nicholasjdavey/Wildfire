# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:32:32 2017

@author: davey
"""

import numpy
import math

class Simulation():
    # Class for defining a simulation run
    
    def __init__(self):
        # Constructs an instance
        self.fireSeverity = numpy.empty([0,0])
        self.dangerIndex = numpy.empty([0,0])
        self.rain = numpy.empty([0,0])
        self.humidity = numpy.empty([0,0])
        self.wind = numpy.empty([0,0])
        self.temperature = numpy.empty([0,0])
        self.experimentalScenario = None
        self.controls = []
        self.model = None

    def getModel(self):
        return self.model

    def setModel(self,m):
        self.model = m

    def getFireSeverity(self):
        return self.fireSeverity

    def setFireSeverity(self,fs):
        self.fireSeverity(fs)

    def getDangerIndex(self):
        return self.dangerIndex

    def setDangerIndex(self,di):
        self.dangerIndex = di

    def getRain(self):
        return self.rain

    def setRain(self,rain):
        self.rain = rain

    def getHumidity(self):
        return self.humidity

    def setHumidity(self,h):
        self.humidity = h

    def getWind(self):
        return self.wind

    def setWind(self,w):
        self.wind = w

    def getTemperature(self):
        return self.temperature

    def setTemperature(self,t):
        self.temperature = t

    def getExperimentalScenario(self):
        return self.experimentalScenario

    def setExperimentalScenario(self,es):
        self.experimentalScenario = es

    def getControls(self):
        return self.controls

    def setControls(self,c):
        self.controls = c

    def simulate(self):
        # Generate exogenous forward paths for weather and fires and save
        exogenousPaths = self.forwardPaths()

        # Generate random control paths and store
        randCont = self.randomControls()

        # Generate endogenous fire growth paths given the above and store
        endogenousPaths = self.endogenousPaths(exogenousPaths,randCont)

    def forwardPaths(self):
        # We don't need to store the precipitation, wind, and temperature matrices
        # over time. We only need the resulting danger index

        for path in range(self.model.getROVPaths):
            initialForwardPath()

        return 0

    def randomControls(self):
        return 0

    def endogenousPaths(ep,rc):
        # We store the actual fires and their sizes
        return 0

    def multiLocLinReg(self,predictors,regressors):
        pass

    def initialForwardPath(self):
        regionSize = self.model.getRegion().getX().size
        timeSteps = self.model.getTimeSteps()
        stepSize = self.model.getStepSize()

        precipitation = numpy.empty([stepSize,regionSize[0],regionSize[1]])
        temperature = numpy.empty([stepSize,regionSize[0],regionSize[1]])
        wind = numpy.empty([stepSize,regionSize[0],regionSize[1]])
        FFDI = self.model.getWeatherGenerator().generateFFDI()

    def pathRecomputation(self,t,state_t,maps):
        # Return recomputed VALUES as a vector across the paths
        return 0

    @staticmethod
    def computeFFDI(temp,rh,wind,df):
        return 2*numpy.exp(-0.45+0.987*numpy.log(df)-0.0345*rh+0.0338*temp+0.0234*wind)
