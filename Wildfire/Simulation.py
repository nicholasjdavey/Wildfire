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

        precipitation = numpy.empty([timeSteps,regionSize])
        precipitation[0] = region.getRain()
        temperature = numpy.empty([timeSteps,regionSize])
        temperature[0] = region.getTemperature()
        windNS = numpy.empty([timeSteps,regionSize])
        windNS[0] = region.getWindN()
        windEW = numpy.empty([timeSteps,regionSize])
        windEW[0] = region.getWindE()
        FFDI = numpy.empty([timeSteps,regionSize])
        FFDI[0] = region.getDangerIndex()
        
        wg = region.getWeatherGenerator()
        
        # Simulate the path forward from time zero to the end
        for ii in range(timeSteps):
            # 1. Determine wet locations
            # Initial draws of random variable
            w = numpy.random.normal(0,1,regionSize)
            # Now, correlate them using the Cholesky decomposition of the
            # occurrence covariance matrix
            C = numpy.linalg.cholesky(wg.getWetOccurrenceCovariance())
            # Final random vector
            w = numpy.matmul(C,w)
            # Compare to probabilities
            p = numpy.multiply(wg.getWetProbT0Wet(),precipitation[ii]) + numpy.multiply(wg.getWetProbT0Dry(),1-precipitation[ii])
            precipitation[ii+1] = bool(w < scipy.stats.norm.isf(q=p,loc=0,scale=1))
            
            

    def pathRecomputation(self,t,state_t,maps):
        # Return recomputed VALUES as a vector across the paths
        return 0

    @staticmethod
    def computeFFDI(temp,rh,wind,df):
        return 2*numpy.exp(-0.45+0.987*numpy.log(df)-0.0345*rh+0.0338*temp+0.0234*wind)
