# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:35:51 2017

@author: davey
"""

import numpy
from Resource import Resource

class Land(Resource):
    # Class for defining fire trucks

    def __init__(self):
        self.crewSize = 0

    def getCrewSize(self):
        return self.crewSize
        
    def setCrewSize(self,s):
        self.crewSize = s