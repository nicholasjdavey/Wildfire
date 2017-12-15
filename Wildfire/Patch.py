# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:33:16 2017

@author: davey
"""

import numpy

class Patch():
    # Class for defining a homegenous sub-region

    def __init__(self):
        # Constructs an instance
        self.vertices = numpy.empty([0,0])
        self.centriod = numpy.empty([0,0])
        self.area = numpy.empty([0,0])
        self.vegetation = None
        self.temperature = None
        self.humidity = None
        self.wind = None
        self.endArea = None
        self.attackSuccess = None
        self.fires = []