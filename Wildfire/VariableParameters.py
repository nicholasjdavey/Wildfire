# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:33:07 2017

@author: davey
"""

import numpy

class VariableParameters():
    # Class for defining the parameters of the scenario currently being tested

    def __init__(self):
        # Constructs an instance
        speedMultipliers = []
        occurrenceProbMultipliers = []
        damageIntensityMultipliers = []
        weatherUncertMultipliers = []

    def getSpeedMultipliers(self):
        return self.speedMultipliers

    def setSpeedMultipliers(self,sm):
        self.speedMultipliers = sm

    def getOccurrenceProbMultipliers(self):
        return self.occurrenceProbMultipliers

    def setOccurrenceProbMultipliers(self,opm):
        self.occurrenceProbMultipliers = opm

    def getDamageIntensityMultipliers(self):
        return self.damageIntensityMultipliers

    def setDamageIntensityMultipliers(self,dim):
        self.damageIntensityMultipliers = dim

    def getWeatherUncertMultipliers(self):
        return self.weatherUncertMultipliers

    def setWeatherUncertMultipliers(self,wum):
        self.weatherUncertMultipliers = wum
