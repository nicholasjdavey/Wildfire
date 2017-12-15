# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:34:56 2017

@author: davey
"""

import numpy
from datetime import datetime

class Fire():
    # Class for managing fires

    def __init__(self):
        # Constructs an instance
        self.location = numpy.empty([0,0])
        self.size = 0.0
        self.start = datetime.now()
        self.end = datetime.now()
        self.initialSize = 0.0
        self.finalSize = 0.0
        self.responseEncoding = ""
        self.respondingStations = []