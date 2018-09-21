# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:35:45 2017

@author: davey
"""

from Resource import Resource
from datetime import timedelta


class Heli(Resource):
    # Class for defining helicopter support resources

    def __init__(self):
        # Constructs an instance
        Resource.__init__(self)
        self.flyingHours = timedelta(hours = 0)
        self.maxDailyHours = timedelta(hours = 0)

    def getFlyingHours(self):
        return self.flyingHours

    def setFlyingHours(self, h):
        self.flyingHours = h

    def getMaxDailyHours(self):
        return self.maxDailyHours

    def setMaxDailyHours(self, d):
        self.maxDailyHours = d
