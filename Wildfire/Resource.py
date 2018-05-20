# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:35:57 2017

@author: davey
"""

import numpy


class Resource():
    # Parent class for defining fire-fighting resources

    def __init__(self):
        # Constructs an instance
        self.type = ""
        self.speed = 0.0
        self.capacity = 0.0
        self.location = None
        self.assignedFires = []

    def getType(self):
        return self.type

    def setType(self, t):
        self.type = t

    def getSpeed(self):
        return self.speed

    def setSpeed(self, s):
        self.speed = s

    def getCapacity(self):
        return self.capacity

    def setCapacity(self, c):
        self.capacity = c

    def getLocation(self):
        return self.location

    def setLocation(self, l):
        self.location = l

    def getAssignedFires(self):
        return self.assignedFires

    def setAssignedFires(self, f):
        self.assignedFires = f
