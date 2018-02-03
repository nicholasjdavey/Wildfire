# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:35:32 2017

@author: davey
"""

import numpy

class Control():
    # Class for managing land bases for fire trucks

    def __init__(self):
        # Constructs an instance
        lambda1 = 0.5
        lambda2 = 0.5

    def getLambda1(self):
        return self.lambda1

    def setLambda1(self,l1):
        self.lambda1 = l1

    def getLambda2(self):
        return self.lambda2

    def setLambda2(self,l2):
        self.lambda2 = l2
