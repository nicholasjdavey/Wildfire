# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:34:50 2017

@author: davey
"""

import numpy

class Process():
    # Class for managing stochastic processes (mean-reverting atm)

    def __init__(self):
        # Constructs an instance
        self.mean = 0.0
        self.noise = 0.0
        self.reversion = 0.0
        self.trend = 0.0
        self.jumpParam = 0.0
        self.jumpProb = 0.0