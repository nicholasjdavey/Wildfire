# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:35:51 2017

@author: davey
"""

from Resource import Resource


class Land(Resource):
    # Class for defining fire trucks

    def __init__(self):
        self.crewSize = 0
        self.maxDailyHours = 0

    def getCrewSize(self):
        return self.crewSize

    def setCrewSize(self, s):
        self.crewSize = s

    def getMaxDailyHours(self):
        return self.maxDailyHours

    def setMaxDailyHours(self, h):
        self.maxDailyHours = h
