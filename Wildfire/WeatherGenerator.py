# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:10:43 2017

@author: davey
"""

import numpy
import scipy
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
        self.humidityReductionMean = numpy.empty([0,0])
        self.humidityReductionSD = numpy.empty([0,0])
        self.precipContributionMultiplier = numpy.empty([0,0])
        self.humidityCorrelation = numpy.empty([0,0])
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
        self.region = None

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
        
    def getRegion(self):
        return self.region
        
    def setRegion(self,r):
        self.region = r
        
    def getHumidityReductionMean(self):
        return self.humidityReductionMean
        
    def setHumidityReductionMean(self,h):
        self.humidityReductionMean = h
        
    def getHumidityReductionSD(self):
        return self.humidityReductionSD
        
    def setHumidityReductionSD(self,h):
        self.humidityReductionSD = h
        
    def getPrecipitationContributionMultiplier(self):
        return self.precipContributionMultiplier
        
    def setPrecipitationContributionMultiplier(self,p):
        self.precipContributionMultiplier = p
        
    def getHumidityCorrelation(self):
        return self.humidityCorrelation
        
    def setHumidityCorrelation(self,h):
        self.humidityCorrelation = h

    # Simulate one period
    def computeWeather(self,rain,precipitation,tempMin,tempMax,windNS,windEW,FFDI,time):
        # 1. Determine wet locations
        self.computePrecipitation(rain,precipitation,time)
        
        # 3. Using wet locations, compute temperatures        
        self.computeTemperature(rain,tempMin,tempMax,time)
        
        # 4. Compute wind
        self.computeWind(windNS,windEW,time)
        
        # 5. Use all of these results to compute the danger index
        self.generateFFDI(precipitation,temperature,windNS,windEW,FFDI,time)
    
    def computePrecipitation(self,rain,precipitation,time):
        regionSize = self.region.getX().size
        
        # Initial draws of random variable        
        w = numpy.random.normal(0,1,regionSize)
        # Now, correlate them using the Cholesky decomposition of the
        # occurrence covariance matrix
        C = numpy.linalg.cholesky(self.getWetOccurrenceCovariance())
        # Final random vector
        h = numpy.matmul(C,w)
        # Compare to probabilities
        prob = numpy.multiply(self.getWetProbT0Wet()[:,time],rain[time]) + numpy.multiply(self.getWetProbT0Dry()[:,time],1-rain[time])
        rain[time+1] = h < scipy.stats.norm.ppf(prob)
        
        # 2. Using wet locations, compute precipitation amounts
        # Relative humidity (precipitation) is a function of the relative
        # humidity at the previous time step and the precipitation at this time
        # Current precipitation at rainfall locations
        alpha = self.getAlphas()[time]
        
        betaChoice = alpha < numpy.random.uniform(0,1,regionSize)
        beta = numpy.multiply(self.getPrecipBetas()[0],betaChoice) + numpy.multiply(self.getPrecipBetas()[0],1-betaChoice )
        # Initial draws of random variable
        w = numpy.random.normal(0,1,regionSize)
        # Now correlate using the Cholesky decomposition of the occurrence
        # covariance matrix
        C = numpy.linalg.cholesky(self.getHumidityCorrelation())
        # Final random vector
        h = numpy.matmul(C,w)
        # Compute the precipitation amounts. Only areas that actually experience
        # precipitation are assigned the value calculated here. Multiply by the
        # precipitation multiplier to give the contribution to humidity
        precipitation[time+1] = numpy.multiply(rain[time+1],self.getPrecipitationContributionMultiplier()[time]*(0.13 - beta*numpy.log(1-scipy.stats.norm(0,1).cdf(h))))
        
        # Now add the reduced humidity from the previous period
        precipitation[time+1] = precipitation[time+1] + precipitation[time]*numpy.random.normal(self.getHumidityReductionMean()[time],self.getHumidityReductionSD()[time],regionSize)
    
    def computeTemperature(self,rain,tempMin,tempMax,time):
        regionSize = self.region.getX().size
        
        # First, generate random vector
        e = numpy.random.normal(0,1,regionSize*2)
        
        # Normalise previous time step temperatures conditional on whether
        # previous time step was wet or dry
        # MEANS
        meanDryMin = numpy.multiply(1-rain[time],tempMin[time])/(numpy.sum(1-rain[time]))
        meanWetMin = numpy.multiply(rain[time],tempMin[time])/(numpy.sum(rain[time]))
        meanDryMax = numpy.multiply(1-rain[time],tempMax[time])/(numpy.sum(1-rain[time]))
        meanWetMax = numpy.multiply(rain[time],tempMax[time])/(numpy.sum(rain[time]))
        # STANDARD DEVIATIONS
        sdDryMin = numpy.sqrt(numpy.sum(numpy.multiply(1-rain[time],numpy.power(tempMin[time]-meanDryMin,2))))
        sdWetMin = numpy.sqrt(numpy.sum(numpy.multiply(rain[time],numpy.power(tempMin[time]-meanWetMin,2))))
        sdDryMax = numpy.sqrt(numpy.sum(numpy.multiply(1-rain[time],numpy.power(tempMax[time]-meanDryMax,2))))
        sdWetMax = numpy.sqrt(numpy.sum(numpy.multiply(rain[time],numpy.power(tempMax[time]-meanWetMax,2))))        
        # CREATE THE STANDARDISED TEMPERATURES
        tempMinStd = numpy.multiply(1-rain[time],(tempMin[time]-meanDryMin)/sdDryMin) + numpy.multiply(rain[time],(tempMin[time]-meanWetMin)/sdWetMin)
        tempMaxStd = numpy.multiply(1-rain[time],(tempMax[time]-meanDryMax)/sdDryMax) + numpy.multiply(rain[time],(tempMax[time]-meanWetMax)/sdWetMax)
        tempPrev = numpy.array([tempMinStd.transpose(),tempMaxStd.transpose()])
        # Now compute the current temperature        
        tempNow = numpy.matmul(self.tempA[time],tempPrev)+numpy.matmul(self.tempB,e)
        # Save to arrays
        tempNowReshaped = tempNow.reshape((2,regionSize))
        tempMin[time+1] = numpy.multiply(1-rain[time+1],tempNowReshaped[:,0]*sdDryMin+meanDryMin) + numpy.multiply(rain[time+1],tempNowReshaped[:,0]*sdWetMin+meanWetMin)
        tempMax[time+1] = numpy.multiply(1-rain[time+1],tempNowReshaped[:,1]*sdDryMax+meanDryMax) + numpy.multiply(rain[time+1],tempNowReshaped[:,1]*sdWetMax+meanWetMax)
    
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
