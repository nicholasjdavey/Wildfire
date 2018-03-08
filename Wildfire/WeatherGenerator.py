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
        self.wetCorrelationMatrix = numpy.empty([0,0])
        self.precipAlpha = numpy.empty([0,0])
        self.precipBeta1 = numpy.empty([0,0])
        self.precipBeta2 = numpy.empty([0,0])
        self.precipCorrelationMatrix = numpy.empty([0,0])
        # Non-precipitation variables are performed by a first-order
        # multivariate auto-regression (Wilks 2009) conditional upon wet and
        # dry days => T_t = [A]T_(t-1) + [B]e_t. [A] is block diagonal.
        self.tempAwet = numpy.empty([0,0])
        self.tempAdry = numpy.empty([0,0])
        self.tempBwet = numpy.empty([0,0])
        self.tempBdry = numpy.empty([0,0])
        self.windNSAwet = numpy.empty([0,0])
        self.windNSAdry = numpy.empty([0,0])
        self.windNSBwet = numpy.empty([0,0])
        self.windNSBdry = numpy.empty([0,0])
        self.windEWAwet = numpy.empty([0,0])
        self.windEWAdry = numpy.empty([0,0])
        self.windEWBwet = numpy.empty([0,0])
        self.windEWBdry = numpy.empty([0,0])

    def calculateParametersFromStationData(self):
        pass

    def computeWeather():
        pass

    def generateFFDI():
        pass
