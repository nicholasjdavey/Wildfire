# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:33:36 2017

@author: davey
"""

import numpy

class Station():
    # Class for managing fire-fighting resource sources

    def __init__(self):
        # Constructs an instance
        self.location = numpy.empty([0,0])
        self.capacity = 0.0
        self.engagedFires = []
        self.coverTimes = numpy.empty([0,0])
        self.coveredPatches = []