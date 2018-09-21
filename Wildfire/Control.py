# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:35:32 2017

@author: davey
"""


class Control():
    # Class for managing land bases for fire trucks

    def __init__(self):
        # Constructs an instance
        self.lambda1 = 0.5
        self.lambda2 = 0.5
        self.eta1 = 1
        self.eta2 = 1
        self.eta3 = 1

    def getLambda1(self):
        return self.lambda1

    def setLambda1(self, l1):
        self.lambda1 = l1

    def getLambda2(self):
        return self.lambda2

    def setLambda2(self, l2):
        self.lambda2 = l2

    def getEta1(self):
        return self.eta1

    def setEta1(self, eta):
        self.eta1 = eta

    def getEta2(self):
        return self.eta2

    def setEta2(self, eta):
        self.eta2 = eta

    def getEta3(self):
        return self.eta3

    def setEta3(self, eta):
        self.eta3 = eta
