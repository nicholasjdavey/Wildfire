# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:32:32 2017

@author: davey
"""

import numpy
import math
import scipy

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
        # Generate exogenous forward paths for weather and fire starts and save
        exogenousPaths = self.forwardPaths()

        # Generate random control paths and store
        randCont = self.randomControls()

        # Generate endogenous fire growth paths given the above and store
        endogenousPaths = self.endogenousPaths(exogenousPaths,randCont)
        
        # Use the Monte Carlo paths to compute the policy maps
        rovMaps = self.rov(exogenousPaths,randCont,endogenousPaths)

    def forwardPaths(self):
        # We don't need to store the precipitation, wind, and temperature matrices
        # over time. We only need the resulting danger index

        paths = [[] for ii in range(self.model.getROVPaths())]
        
        for path in range(self.model.getROVPaths()):
            paths[path] =self.initialForwardPath()

        return 0
        
    def rov(self,exogenousPaths,randCont,endogenousPaths):
        pass

    def randomControls(self):
        return 0

    def endogenousPaths(ep,rc):
        # We store the actual fires and their sizes
        return 0

    def multiLocLinReg(self,predictors,regressors):
        pass

    def initialForwardPath(self):
        region = self.model.getRegion()
        regionSize = region.getX().size
        timeSteps = self.model.getTotalSteps()
        stepSize = self.model.getStepSize()

        rain = numpy.empty([timeSteps,regionSize])
        rain[0] = region.getRain()
        precipitation = numpy.empty([timeSteps,regionSize])
        precipitation[0] = region.getHumidity()
        temperatureMin = numpy.empty([timeSteps,regionSize])
        temperatureMin[0] = region.getTemperatureMin()
        temperatureMax = numpy.empty([timeSteps,regionSize])
        temperatureMax[0] = region.getTemperatureMax()
        windNS = numpy.empty([timeSteps,regionSize])
        windNS[0] = region.getWindN()
        windEW = numpy.empty([timeSteps,regionSize])
        windEW[0] = region.getWindE()
        FFDI = numpy.empty([timeSteps,regionSize])
        FFDI[0] = region.getDangerIndex()
        windRegimes = numpy.empty([timeSteps])
        windRegimes[0] = region.getWindRegime()
        
        wg = region.getWeatherGenerator()
        
        # Simulate the path forward from time zero to the end
        for ii in range(timeSteps):
            # Compute weather
            wg.computeWeather(rain,precipitation,temperatureMin,temperatureMax,windRegimes,windNS,windEW,FFDI,ii)
            pass            

    def pathRecomputation(self,t,state_t,maps):
        # Return recomputed VALUES as a vector across the paths
        return 0

    @staticmethod
    def computeFFDI(temp,rh,wind,df):
        return 2*numpy.exp(-0.45+0.987*numpy.log(df)-0.0345*rh+0.0338*temp+0.0234*wind)
