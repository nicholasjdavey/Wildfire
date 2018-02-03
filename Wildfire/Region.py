# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:10:43 2017

@author: davey
"""

import numpy
from Process import Process

class Region():
    # Class for defining a study region
    
    def __init__(self):
        # Constructs an instance
        self.patches = []
        self.stations = []
        self.fireSeverity_0 = numpy.empty([0,0])
        self.dangerIndex_0 = numpy.empty([0,0])
        self.rain_0 = numpy.empty([0,0])
        self.humidity_0 = numpy.empty([0,0])
        self.wind_0 = numpy.empty([0,0])
        self.temperature_0 = numpy.empty([0,0])
        self.vegetation = []
        self.stationDistances = numpy.empty([0,0])
        self.stationPatchDistances = numpy.empty([0,0])
        self.weatherGenerator = None
        self.endArea = None
        self.attackSuccess = None
        self.fires = []
        self.name = ""
        
    
