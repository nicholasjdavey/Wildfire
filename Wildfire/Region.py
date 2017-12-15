# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:10:43 2017

@author: davey
"""

import numpy

class Region():
    # Class for defining a study region
    
    def __init__(self):
        # Constructs an instance
        self.patches = []
        self.stations = []
        self.fireSeverity = numpy.empty([0,0])
        self.dangerIndex = numpy.empty([0,0])
        self.rain = numpy.empty([0,0])
        self.humidity = numpy.empty([0,0])
        self.wind = numpy.empty([0,0])
        self.temperature = numpy.empty([0,0])
        self.stationDistances = numpy.empty([0,0])
        self.simulations = []
        self.name = ""
        
    