# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:10:43 2017

@author: davey
"""

import numpy
from Process import Process

class WeatherGenerator():
    # Class for defining a weather generator for creating the exogenous paths.
    # This includes the fire danger index over time in each path as well as the
    # random fire instances.
    # The weather generator compute the STATION values at each future time step
    # and interpolates the values for all other remaining points in the region.

    def __init__(self):
        self.wetProbT0Wet = numpy.empty([0,0])
        self.wetProbT0Dry = numpy.empty([0,0])
        self.wetOccurrenceCovarianceMatrix = numpy.empty([0,0])
        self.precipAmountCovarianceMatrix = numpy.empty([0,0])
        self.precipAlpha = numpy.empty([0,0])
        # One matrix for each time period
        self.precipBetas = []
        # Non-precipitation variables are performed by a first-order
        # multivariate auto-regression (Wilks 2009) conditional upon wet and
        # dry days => T_t = [A]T_(t-1) + [B]e_t. [A] is block diagonal.
        # One matrix for each time period
        # The means and standard deviations for wet and dry temperatures at
        # different times of day are required for 'de-standardising' the
        # generated next period temperatures, consistent with whether a location
        # is wet or dry at the next period.
        self.tempMeanWet = numpy.empty([0,0])
        self.tempMeanDry = numpy.empty([0,0])
        self.tempSDWet = numpy.empty([0,0])
        self.tempSDDry = numpy.empty([0,0])
        self.tempA = []
        self.tempB = []
        # One for each time period. We assume precipitation occurrence has no
        # impact on the regressions.
        self.windNSA = []
        self.windNSB = []
        self.windEWA = []
        self.windEWB = []

    def getWetProbT0Wet(self):
        return self.wetProbT0Wet
        
    def setWetProbT0Wet(self,p):
        self.wetProbT0Wet = p
        
    def getWetProbT0Dry(self):
        return self.wetProbT0Dry
        
    def setWetProbT0Dry(self,p):
        self.wetProbT0Dry = p
        
    def getWetOccurrenceCovariance(self):
        return self.wetOccurrenceCovariance
    
    def setWetOccurrenceCovariance(self,o):
        self.wetOccurrenceCovariance = o
        
    def getPrecipAmountCovariance(self):
        return self.precipAmountCovariance
        
    def setPrecipAmountCovariance(self,p):
        self.precipAmountCovariance = p
        
    def getPrecipAlpha(self):
        return self.precipAlpha
        
    def setPrecipAlpha(self,a):
        self.precipAlpha = a
        
    def getPrecipBetas(self):
        return self.precipBetas
        
    def setPrecipBetas(self,b):
        self.precipBetas = b
        
    def getTempMeanWet(self):
        return self.tempMeanWet
        
    def setTempMeanWet(self,t):
        self.tempMean = t
        
    def getTempMeanDry(self):
        return self.tempMeanDry
        
    def setTempMeanDry(self,a):
        self.tempMeanDry = a
        
    def getTempSDWet(self):
        return self.tempSDWet
        
    def setTempSDWet(self,sd):
        self.tempSDWet = sd
        
    def getTempSDDry(self):
        return self.tempSDDry
        
    def setTempSDDry(self,sd):
        self.tempSDDry = sd
        
    def getTempA(self):
        return self.tempA
        
    def setTempA(self,a):
        self.tempA = a
        
    def getTempB(self):
        return self.tempB
        
    def setTempB(self,b):
        self.tempB = b
        
    def getWindNSA(self):
        return self.windNSA
        
    def setWindNSA(self,a):
        self.windNSA = a
        
    def getWindNSB(self):
        return self.windNSB
        
    def setWindNSB(self,b):
        self.windNSB = b
        
    def getWindEWA(self):
        return self.windEWA
        
    def setWindEWA(self,a):
        self.windEWA = a
        
    def getWindEWB(self):
        return self.windEWB
        
    def setWindEWB(self,b):
        self.windEWB = b
        

    # Simulate one period
    def computeWeather(self):
        self.computePrecipitation()
        self.computeTemperature()
        self.computeWind()
        self.generateFFDI()
    
    def computePrecipitation(self):
        pass
    
    def computeTemperature(self):
        pass
    
    def computeWind(self):
        pass    
    
    def generateFFDI(self):
        pass
    
    # Compute weather generator parameters from weather station data
    def calculateParametersFromStationData(self):
        self.computePrecipitationOccurrenceConditionalProbs()
        self.computePrecipitationOccurrenceCovariances()
        self.computePrecipitationAmountParameters()
        self.computeTemperatureParameters()
        self.computeWindParameters()
    
    def computePrecipitationOccurrenceConditionalProbs(self):
        pass
    
    def computePrecipitationOccurrenceCovariances(self):
        pass
    
    def computePrecipitationAmountParameters(self):
        pass
    
    def computeTemperatureParameters(self):
        pass
    
    def computeWindParameters(self):
        pass