# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:33:07 2017

@author: davey
"""


class ExperimentalScenario():
    # Class for defining the parameters of the scenario currently being tested
    scenarios = 0

    def __init__(self):
        # Constructs an instance
        self.speedMultIdx = 0
        self.occProbMultIdx = 0
        self.damIntMultIdx = 0
        self.weatherUncertMultIdx = 0
        self.scenarioNo = ExperimentalScenario.scenarios
        ExperimentalScenario.scenarios = ExperimentalScenario.scenarios + 1

    def getSpeedMultIdx(self):
        return self.speedMultIdx

    def setSpeedMultIdx(self, smi):
        self.speedMultIdx = smi

    def getOccProbMultIdx(self):
        return self.occProbMultIdx

    def setOccProbMultIdx(self, opi):
        self.occProbMultIdx = opi

    def getDamIntMultIdx(self):
        return self.damIntMultIdx

    def setDamIntMultIdx(self, dimi):
        self.damIntMultIdx = dimi

    def getWeatherUncertMultIdx(self):
        return self.weatherUncertMultIdx

    def setWeatherUncertMultIdx(self, wumi):
        self.weatherUncertMultIdx = wumi
