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