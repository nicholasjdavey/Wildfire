# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:35:25 2017

@author: davey
"""

import numpy
import Station

class AirStrip(Station):
    # Class for defining air strips

    def __init__(self):
        # Constructs an instance
        Station.__init__(self)
        self.airTankers = []
        self.helicopters = []