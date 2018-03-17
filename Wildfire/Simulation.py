# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:32:32 2017

@author: davey
"""

import numpy
import math
import scipy
import pulp

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
        # The forward paths are just the Danger Indices at each location
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

        paths = []
        
        for path in range(self.model.getROVPaths()):
            paths.append(self.initialForwardPath())           

        return paths
        
    def rov(self,exogenousPaths,randCont,endogenousPaths):
        pass

    def randomControls(self):
        randControls = numpy.random.choice(range(self.model.getControls().size),self.model.getROVPaths()*self.model.getTotalSteps()).reshape(self.model.getROVPaths(),self.model.getTotalSteps()).reshape(self.model.getROVPaths(),self.model.getTotalSteps())
        
        return randControls

    def endogenousPaths(self,ep,rc):
        # We store the actual fires and their sizes
        
        paths = []
        
        for path in range(self.model.getROVPaths()):
            paths.append(self.initialEndogenousPath(ep,rc,path))
            
        return paths

    def multiLocLinReg(self,predictors,regressors):
        pass

    def initialForwardPath(self):
        region = self.model.getRegion()
        regionSize = region.getX().size
        timeSteps = self.model.getTotalSteps()

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
        
        return FFDI
        
    def initialEndogenousPath(self,ep,rc,path):
        regionSize = self.region.getX().size
        timeSteps = self.model.getTotalSteps()
        stepSize = self.model.getStepSize()
        
        # Keep a whole map of fire in the region. We do not keep track of all
        # the randomly-generated numbers used for fire growth and success (or
        # even first starts for that matter) as it is expected that the fires
        # will have different successes and may even be different fires in the
        # first place. This will save a lot of memory.
        fireSeverityMap = []
        aircraftLocations = []
        
        for ii in range(timeSteps):
            control = rc[path,ii]
            
            # NESTED OPTIMISATION #############################################
            # Optimise aircraft locations given selected control
            aircraftLocations.append(self.optimalLocations(rc[path,ii],fireSeverityMap[time]))
            
            # Given the locations found for this control, update the fire
            # severities for the next time period. We use the probabilities.
            fireSeverityMap.append(self.fireSeverity(aircraftLocations,fireSeverityMap[ii],ep[ii]))
            
        return [fireSeverityMap,aircraftLocations]
        
    def comparator(self,ffdi,time):
        comparators = numpy.empty(self.region.getX().size(),1)
        # Serial
        for ii in range(self.region.getX().size):
            # Linear interpolation. Assume ffdis evenly spaced
            veg = self.region.getVegetation()[ii]
            ffdiRange = self.region.getVegetations[veg].getFFDIRange()
            occurrenceProbs = self.region.getVegetations[veg].getOccurrence()
            ffdis = ffdiRange.size
            ffdiMinIdx = math.floor((ffdi[time][ii]-ffdiRange[0])*(ffdis-1)/(ffdiRange[ffdis] - ffdiRange[0]))
            ffdiMaxIdx = ffdiMinIdx + 1
            
            if ffdiMinIdx < 0:
                ffdiMinIdx = 0
                ffdiMaxIdx = 1                
            elif ffdiMaxIdx >= ffdis:
                ffdiMinIdx = ffdis - 2
                ffdiMaxIdx = ffdis - 1

            xd = (ffdi[time][ii]-ffdiRange[ffdiMinIdx])/(ffdiRange[ffdiMaxIdx] - ffdiRange[ffdiMinIdx])
            
            comparators[ii] = xd*occurrenceProbs[ffdiMinIdx] + (1-xd)*occurrenceProbs[ffdiMaxIdx]
        
        return comparators
        
    def optimalLocations(self,randCont,fireSeverityMap):
        pass
    
    def fireSeverity(self,locations,fireSeverityMap,ffdi):
        pass

    def pathRecomputation(self,t,state_t,maps):
        # Return recomputed VALUES as a vector across the paths
        return 0

    @staticmethod
    def computeFFDI(temp,rh,wind,df):
        return 2*numpy.exp(-0.45+0.987*numpy.log(df)-0.0345*rh+0.0338*temp+0.0234*wind)
