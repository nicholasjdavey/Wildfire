# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:35:38 2017

@author: davey
"""

import numpy
from Resource import Resource
from datetime import timedelta

class Tanker(Resource):
    # Class for defining air tanker resources

    def __init__(self):
        # Constructs an instance
        Resource.__init__(self)
        self.flyingHours = timedelta(hours = 0)
        self.maxDailyHours = timedelta(hours = 0)
        
    def getFlyingHours(self):
        return self.flyingHours
        
    def setFlyingHours(self,h):
        self.flyingHours = h
        
    def getMaxDailyHours(self):
        return self.maxDailyHours
        
    def setMaxDailyHours(self,d):
        self.maxDailyHours = d
