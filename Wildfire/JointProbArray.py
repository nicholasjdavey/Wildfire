# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:32:56 2017

@author: davey
"""

import numpy

class JointProbArray():
    # Class for defining a joint probability multi-dimensional array

    def __init__(self):
        # Constructs an instance
        self.params = []
        self.ranges = []
        self.probabilities = numpy.empty([0,0])
    