# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:35:25 2017

@author: davey
"""

import numpy
from Station import Station

class AirStrip(Station):
    # Class for defining air strips

    def __init__(self):
        # Constructs an instance
        Station.__init__(self)
        self.airTankers = []
        self.helicopters = []

    def getAirTankers(self):
        return self.airTankers

    def setAirTankers(self,at):
        self.airTankers = at

    def getHelicopters(self):
        return self.helicopters

    def setHelicopters(self,h):
        self.helicopters = h
