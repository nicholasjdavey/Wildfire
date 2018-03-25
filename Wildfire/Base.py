# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:35:32 2017

@author: davey
"""

import numpy
from Station import Station

class Base(Station):
    # Class for managing land bases for fire trucks

    def __init__(self):
        # Constructs an instance
        Station.__init__(self)
        self.landResources = []
        self.maxResources = 0

    def getLandResources(self):
        return self.landResources
        
    def setLandResources(self,l):
        self.landResources = l
        
    def getMaxResources(self):
        return self.maxResources
        
    def setMaxResources(self,m):
        self.maxResources = m