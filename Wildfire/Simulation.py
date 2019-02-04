# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:32:32 2017

@author: davey
"""

import sys
import time
import numpy
import cplex
import math
import copy
import os
import errno
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as mpp
import matplotlib.collections as mpc
import matplotlib.cm as clrmp
import matplotlib.backends.backend_pdf as pdf
#import matplotlib.colors as colors
from sklearn.preprocessing import PolynomialFeatures

import SimulationNumba

class Simulation():
    # Class for defining a simulation run
    simulations = 0

    def __init__(self):
        # Constructs an instance
        self.fireSeverity = numpy.empty([0, 0])
        self.dangerIndex = numpy.empty([0, 0])
        self.rain = numpy.empty([0, 0])
        self.humidity = numpy.empty([0, 0])
        self.wind = numpy.empty([0, 0])
        self.temperature = numpy.empty([0, 0])
        self.experimentalScenario = None
        self.controls = []
        self.model = None
        self.expectedDamages = []
        self.finalDamageMaps = []
        self.assignmentModel = None
        self.relocationModel = None
        self.realisedAssignments = []
        self.realisedFires = []
        self.realisedStarts = []
        self.initialExtinguished = []
        self.realisedFFDIs = []
        self.aircraftHours = []
        self.id = Simulation.simulations
        self.statePaths = None
        self.mapStates = None
        self.costs2Go = None
        self.mapC2G = None
        self.accumulatedDamages = None
        self.regressionsX = None
        self.regressionsY = None
        self.rSquared = None
        self.tStatistic = None
        self.pVal = None
        self.regModel = None
        self.dynamicC2G = None
        Simulation.simulations += 1

    def getModel(self):
        return self.model

    def setModel(self, m):
        self.model = m

    def getFireSeverity(self):
        return self.fireSeverity

    def setFireSeverity(self, fs):
        self.fireSeverity(fs)

    def getDangerIndex(self):
        return self.dangerIndex

    def setDangerIndex(self, di):
        self.dangerIndex = di

    def getRain(self):
        return self.rain

    def setRain(self, rain):
        self.rain = rain

    def getHumidity(self):
        return self.humidity

    def setHumidity(self, h):
        self.humidity = h

    def getWind(self):
        return self.wind

    def setWind(self, w):
        self.wind = w

    def getTemperature(self):
        return self.temperature

    def setTemperature(self, t):
        self.temperature = t

    def getExperimentalScenario(self):
        return self.experimentalScenario

    def setExperimentalScenario(self, es):
        self.experimentalScenario = es

    def getControls(self):
        return self.controls

    def setControls(self, c):
        self.controls = c

    def getExpectedDamages(self):
        return self.expectedDamages

    def setExpectedDamages(self, d):
        self.expectedDamages = d

    def getFinalDamageMaps(self):
        return self.finalDamageMaps

    def setFinalDamageMaps(self, m):
        self.finalDamageMaps = m

    def getRealisedAssignments(self):
        return self.realisedAssignments

    def setRealisedAssignments(self, ra):
        self.realisedAssignments = ra

    def getRealisedFires(self):
        return self.realisedFires

    def setRealisedFires(self, rf):
        self.realisedFires = rf

    def getRealisedStarts(self):
        return self.realisedStarts

    def setRealisedStarts(self, s):
        self.realisedStarts = s

    def getInitialExtinguished(self):
        return self.initialExtinguished

    def setInitialExtinguished(self, i):
        self.initialExtinguished = i

    def getRealisedFFDIs(self):
        return self.realisedFFDIs

    def setRealisedFFDIs(self, rf):
        self.realisedFFDIs = rf

    def getAircraftHours(self):
        return self.aircraftHours

    def setAircraftHours(self, ah):
        self.aircraftHours = ah

    def getStatePaths(self):
        return self.statePaths

    def setStatePaths(self, p):
        self.statePaths = p

    def getMapStates(self):
        return self.mapStates

    def setMapStates(self, s):
        self.mapStates = s

    def getCosts2Go(self):
        return self.costs2Go

    def setCosts2Go(self, c):
        self.costs2Go = c

    def getMapC2G(self):
        return self.mapC2G

    def setMapC2G(self, c):
        self.mapC2G = c

    def getRegressionsX(self):
        return self.regressionsX

    def setRegressionsX(self, r):
        self.regressionsX = r

    def getRegressionsY(self):
        return self.regressionsY

    def setRegressionsY(self, r):
        self.regressionsY = r

    def getRSquared(self):
        return self.rSquared

    def setRSquared(self, r):
        self.rSquared = r

    def getTStatistic(self):
        return self.tStatistic

    def setTStatistic(self, t):
        self.tStatistic = t

    def getPVal(self):
        return self.pVal

    def setPVal(self, p):
        self.pVal

    def getRegModel(self):
        return self.regModel

    def setRegModel(self, r):
        self.regModel = r

    def getDynamicC2G(self):
        return self.dynamicC2G

    def setDynamicC2G(self, c):
        self.dynamicC2G = c

    def simulate(self):

        switch = {
            0: self.samplePaths,
            1: self.simulateMPC,
            2: self.simulateROV,
            3: self.simulateMPC,
            4: self.simulateMPC,
            5: self.simulateMPC
            }

        algo = self.model.getAlgo()

        if not(algo == 0):
            self.buildLPModel()

        prog = switch.get(algo)

        if algo == 3:
            prog(True)
        else:
            prog()

    """////////////////////////////////////////////////////////////////////////
    /////////////////////// CPLEX Models Used in Program //////////////////////
    ////////////////////////////////////////////////////////////////////////"""

    def buildLPModel(self):
        switch = {
            1: self.buildMaxCover,
            2: self.buildPMedian,
            3: self.buildAssignment2,
            4: self.buildAssignment1
        }

        self.buildModelBase()

        lpModel = self.model.getNestedOptMethod()
        prog = switch.get(lpModel)
        prog()

    def buildModelBase(self):
        """ Builds an assignment model to reuse for computing relocations. It
        does not contain any fires. Rather, it is copied each time it is needed
        and fires are added. The copied model is solved to give the required
        assignments"""

        """ First copy the relocation model """
        bases = self.model.getRegion().getStations()[0]
        patches = self.model.getRegion().getPatches()

        cplxMod = cplex.Cplex()

        configsP = (self.model.getUsefulConfigurationsPotential()
                    if self.model.getNestedOptMethod() > 2
                    else [0])
        configsE = (self.model.getUsefulConfigurationsExisting()
                    if self.model.getNestedOptMethod() > 2
                    else [0])

        """ INDEXED SETS AND LIST LENGTHS """
        cplxMod.R = [ii for ii in
                     range(len(self.model.getRegion().getResources()))]
        cplxMod.B = [ii for ii in
                     range(len(self.model.getRegion().getStations()[0]))]
        lenB = len(cplxMod.B)
        cplxMod.N = [ii for ii in
                     range(len(self.model.getRegion().getPatches()))]
        cplxMod.T = [ii for ii in
                     range(self.model.getTotalSteps())]
        cplxMod.K = [ii for ii in range(len(self.model.getConfigurations()))]
        cplxMod.KP = [cplxMod.K[ii-1] for ii in configsP]
        cplxMod.KE = [cplxMod.K[ii-1] for ii in configsE]
        cplxMod.C = [1, 2, 3, 4]
        cplxMod.M = []
        cplxMod.M_All = [ii for ii in cplxMod.M]

        """ PARAMETERS """

        """ Control Parameters """
        cplxMod.lambdas = {
                ii+1:
                (self.model.getControls()[ii].getLambda1(),
                 self.model.getControls()[ii].getLambda2())
                for ii in range(len(self.model.getControls()))}

        cplxMod.thresholds = [
                self.model.getBaseThreshold(), self.model.getFireThreshold()]

        """ Cumulative flying hours """
        cplxMod.G_R = {
                r: 0
                for r in cplxMod.R}

        """ Max flying hours """
        cplxMod.Gmax_R = {
                r: self.model.getRegion().getResources()[r].getMaxDailyHours()
                for r in cplxMod.R}

        """ Number of aircraft required for component C of configuration K """
        cplxMod.Q_KC = {
                (k, c): self.model.getConfigurations()[k+1][c-1]
                for k in cplxMod.K
                for c in cplxMod.C}

        """ Travel times between aircraft and bases """
        cplxMod.d1_RB = {
                (r, b): 0
                for r in cplxMod.R
                for b in cplxMod.B}

        """ Travel time between resource R and fire M """
        cplxMod.d2_RM = {
                (r, m): 0
                for r in cplxMod.R
                for m in cplxMod.M}

        """ Whether resource R satisfies component C for fire M """
        cplxMod.d2_RCM = {
                (r, c, m): 0
                for r in cplxMod.R
                for c in cplxMod.C
                for m in cplxMod.M}

        """ Distances between base B and patch N"""
        cplxMod.d3_BN = {
                (b, n): self.geoDist(patches[n].getCentroid(),
                                     bases[b].getLocation())
                for b in cplxMod.B
                for n in cplxMod.N}

        """ Whether base B satisfies component C for patch N """
        cplxMod.d3_BCN = {
                (b, c, n):
                (1
                    if (((c == 1 or c == 3) and cplxMod.d3_BN[b, n] <=
                            cplxMod.thresholds[0])
                        or (c == 2 or c == 4
                            and cplxMod.d3_BN[b, n] > cplxMod.thresholds[0]
                            and cplxMod.d3_BN[b, n] <= cplxMod.thresholds[1]))
                    else 0)
                for b in cplxMod.B
                for c in cplxMod.C
                for n in cplxMod.N}

        """ Expected number of fires visible by base B for component C """
        cplxMod.no_CB = {
                (c, b): 1
                for c in cplxMod.C
                for b in cplxMod.B}

        """ DECISION VARIABLES """
        cplxMod.decisionVars = {}
        cplxMod.decisionVarsIdxStarts = {}
        totalVars = 0

        """ Aircraft to base assignments """
        cplxMod.decisionVars["X_RB"] = [
                "X_RB_R" + str(r) + "_B" + str(b)
                for r in cplxMod.R
                for b in cplxMod.B]
        cplxMod.decisionVarsIdxStarts["X_RB"] = totalVars
        totalVars = totalVars + len(cplxMod.decisionVars["X_RB"])

        cplxMod.variables.add(
                types=([cplxMod.variables.type.binary]
                       * len(cplxMod.decisionVars["X_RB"])))

        """ Aircraft availability at base assignments """
        cplxMod.decisionVars["A_RB"] = [
                "A_RB_R" + str(r) + "_B" + str(b)
                for r in cplxMod.R
                for b in cplxMod.B]
        cplxMod.decisionVarsIdxStarts["A_RB"] = totalVars
        totalVars = totalVars + len(cplxMod.decisionVars["A_RB"])

        cplxMod.variables.add(
                types=([cplxMod.variables.type.binary]
                       * len(cplxMod.decisionVars["A_RB"])))

        """ CONSTRAINTS """
        cplxMod.constraintNames = {}
        cplxMod.constraintIdxStarts = {}
        totalConstraints = 0

        """Ensures that an aircraft can only be available at a base if it is
        stationed there"""
        cplxMod.constraintNames["C_3"] = [
                "C_3_R" + str(r) + "_B" + str(b)
                for r in cplxMod.R
                for b in cplxMod.B]
        cplxMod.constraintIdxStarts["C_3"] = totalConstraints
        totalConstraints = totalConstraints + len(
                cplxMod.constraintNames["C_3"])

        startXRB = cplxMod.decisionVarsIdxStarts["X_RB"]
        startARB = cplxMod.decisionVarsIdxStarts["A_RB"]
        varIdxs = {(r, b): [startARB + r*lenB + b, startXRB + r*lenB + b]
                   for r in cplxMod.R
                   for b in cplxMod.B}

        varCoeffs = {(r, b): [1, -1] for r in cplxMod.R for b in cplxMod.B}

        cplxMod.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[r, b],
                                val=varCoeffs[r, b])
                        for r in cplxMod.R
                        for b in cplxMod.B],
                senses=["L"]*(len(varIdxs)),
                rhs=[0]*len(varIdxs))

        """Ensures that an aircraft is either available at a base or is
        attending a fire"""
        cplxMod.constraintNames["C_4"] = ["C_4_R" + str(r) for r in cplxMod.R]
        cplxMod.constraintIdxStarts["C_4"] = totalConstraints
        totalConstraints = totalConstraints + len(
                cplxMod.constraintNames["C_4"])

        varIdxs = {(r): [startARB + r*lenB + b for b in cplxMod.B]
                   for r in cplxMod.R}

        varCoeffs = {(r): [1 for b in cplxMod.B] for r in cplxMod.R}

        cplxMod.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[r],
                                val=varCoeffs[r])
                        for r in cplxMod.R],
                senses=["E"]*(len(varIdxs)),
                rhs=[1]*len(varIdxs))

        """Ensures that an aircraft is assigned to one and only one base"""
        cplxMod.constraintNames["C_5"] = [
                "C_5_R" + str(r)
                for r in cplxMod.R]
        cplxMod.constraintIdxStarts["C_5"] = totalConstraints
        totalConstraints = totalConstraints + len(
                cplxMod.constraintNames["C_5"])

        varIdxs = {(r): [startXRB + r*lenB + b for b in cplxMod.B]
                   for r in cplxMod.R}

        varCoeffs = {(r): [1]*len(varIdxs[r]) for r in cplxMod.R}

        cplxMod.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[r],
                                val=varCoeffs[r])
                        for r in cplxMod.R],
                senses=["E"]*len(varIdxs),
                rhs=[1]*len(varIdxs))

        """Ensures that an aircraft can be allocated to at most one fire"""
        cplxMod.constraintNames["C_6"] = [
                "C_6_R" + str(r)
                for r in cplxMod.R]
        cplxMod.constraintIdxStarts["C_6"] = totalConstraints
        totalConstraints = totalConstraints + len(
                cplxMod.constraintNames["C_6"])

        cplxMod.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=[],
                                val=[])
                        for r in cplxMod.R],
                senses=["L"]*len(cplxMod.R),
                rhs=[1]*len(cplxMod.R))

        """Ensures that the maximum number of flying hours are not exceeded"""
        cplxMod.constraintNames["C_9"] = [
                "C_9_K" + str(r)
                for r in cplxMod.R]
        cplxMod.constraintIdxStarts["C_9"] = totalConstraints
        totalConstraints = totalConstraints + len(
                cplxMod.constraintNames["C_9"])

        varIdxs = {(r): [startXRB + r*lenB + b for b in cplxMod.B]
                   for r in cplxMod.R}

        varCoeffs = {(r): [cplxMod.d1_RB[r, b] + r*lenB + b for b in cplxMod.B]
                     for r in cplxMod.R}

        cplxMod.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[r],
                                val=varCoeffs[r])
                        for r in cplxMod.R],
                senses=["L"]*len(cplxMod.R),
                rhs=[cplxMod.Gmax_R[r] - cplxMod.G_R[r]
                     for r in cplxMod.R])

        """ Save the relocation model to the instance """
        self.relocationModel = cplxMod

    def buildMaxCover(self):
        self.relocationModel.objective.set_sense(
                self.relocationModel.objective.sense.maximize)

        """ DECISION VARIABLES """
        totalVars = self.relocationModel.variables.get_num()

        self.relocationModel.decisionVars["Z_CN"] = [
                "Z_CN" + str(n)
                for n in self.relocationModel.N]
        self.relocationModel.decisionVarsIdxStarts["Z_CN"] = totalVars
        totalVars = totalVars + len(self.relocationModel.decisionVars["Z_CN"])

        """ CONSTRAINTS """
        totalConstraints = self.relocationModel.linear_constraints.get_num()

        """Enforces base relocation options based on the control"""
        self.relocationModel.constraintNames["C_11"] = [
                "C_11_R" + str(r) + "_B" + str(b)
                for r in self.relocationModel.R
                for b in self.relocationModel.B]
        self.relocationModel.constraintIdxStarts["C_11"] = totalConstraints
        totalConstraints = totalConstraints + len(
                self.relocationModel.constraintNames["C_11"])

        startXRB = self.relocationModel.decisionVarsIdxStarts["X_RB"]
        lenB = len(self.relocationModel.B)
        varIdxs = {(r, b): [startXRB + r*lenB + b]
                   for r in self.relocationModel.R
                   for b in self.relocationModel.B}

        varCoeffs = {(r, b): [1]
                     for r in self.relocationModel.R
                     for b in self.relocationModel.B}

        self.relocationModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[r, b],
                                val=varCoeffs[r, b])
                        for r in self.relocationModel.R
                        for b in self.relocationModel.B],
                senses=["L"]*(len(varIdxs)),
                rhs=[0]*len(varIdxs))

        """ Cover constraints """
        self.relocationModel.constraintNames["C_A_1"] = [
                "C_A_1_C" + str(c) + "_N" + str(n)
                for c in [0, 1]
                for n in self.relocationModel.N]
        self.relocationModel.constraintIdxStarts["C_A_1"] = totalConstraints
        totalConstraints = totalConstraints + len(
                self.relocationModel.constraintNames["C_A_1"])

        lenN = len(self.relocationModel.N)
        startZCN = self.relocationModel.decisionVarsIdxStarts["Z_CN"]
        varIdxs = {
                (c, n):
                [startZCN + c*lenN + n]
                for c in [0, 1]
                for n in self.relocationModel.N}

        varCoeffs = {(c, n): [1, -1]
                     for c in [0, 1]
                     for n in self.relocationModel.N}

        self.relocationModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[c, n],
                                val=varCoeffs[c, n])
                        for c in [0, 1]
                        for n in self.relocationModel.N],
                senses=["L"]*(len(varIdxs)),
                rhs=[0]*len(varIdxs))

    def buildPMedian(self):
        self.relocationModel.objective.set_sense(
                self.relocationModel.objective.sense.minimize)

        """ CONSTRAINTS """
        totalConstraints = self.relocationModel.linear_constraints.get_num()
        lenB = len(self.relocationModel.B)

        """Restricts the relocation of aircraft by proximity to bases"""
        self.relocationModel.constraintNames["C_11"] = [
                "C_11_R" + str(r) + "_B" + str(b)
                for r in self.relocationModel.R
                for b in self.relocationModel.B]
        self.relocationModel.constraintIdxStarts["C_11"] = totalConstraints
        totalConstraints = totalConstraints + len(
                self.relocationModel.constraintNames["C_11"])

        startXRB = self.relocationModel.decisionVarsIdxStarts["X_RB"]
        varIdxs = {(r, b): [startXRB + r*lenB + b]
                   for r in self.relocationModel.R
                   for b in self.relocationModel.B}

        varCoeffs = {(r, b): [1]
                     for r in self.relocationModel.R
                     for b in self.relocationModel.B}

        self.relocationModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[r, b],
                                val=varCoeffs[r, b])
                        for r in self.relocationModel.R
                        for b in self.relocationModel.B],
                senses=["L"]*(len(varIdxs)),
                rhs=[0]*len(varIdxs))

    def buildAssignment1(self):
        self.relocationModel.objective.set_sense(
                self.relocationModel.objective.sense.minimize)

        lenKP = len(self.relocationModel.KP)

        """ Initial expected damage increase of each fire under each config """
        self.relocationModel.D1_MK = {
                (m, k+1): 0
                for m in self.relocationModel.M
                for k in self.relocationModel.KE}

        """ Init expected damage increase of each patch under each config """
        self.relocationModel.D2_NK = {
                (n, k+1): 0
                for n in self.relocationModel.N
                for k in self.relocationModel.KP}

        """ DECISION VARIABLES """
        totalVars = self.relocationModel.variables.get_num()

        """ Patch configuration covers """
        self.relocationModel.decisionVars["Delta_NK"] = [
                "Delta_NK_Adj_N" + str(n) + "_K" + str(k+1)
                for n in self.relocationModel.N
                for k in self.relocationModel.KP]
        self.relocationModel.decisionVarsIdxStarts["Delta_NK"] = totalVars
        totalVars = totalVars + len(
                self.relocationModel.decisionVars["Delta_NK"])

        self.relocationModel.variables.add(
                ub=[1]*len(self.relocationModel.decisionVars["Delta_NK"]),
                lb=[0]*len(self.relocationModel.decisionVars["Delta_NK"]))

        """ Aircraft-fire assignments and fire configuration covers """
        """
        More complex:
        For each fire (created to date)
        1. Y_MR
        2. Delta_MK
        For fires that are extinguished during simulation, the variable's upper
        bound is set to 0. This effectively eliminates the column
        """
        component1 = [
                ["Y_MR_M" + str(m) + "_R" + str(r)
                 for r in self.relocationModel.R]
                for m in self.relocationModel.M]
        component2 = [
                ["Delta_MK_M" + str(m) + "_K" + str(k+1)
                 for k in self.relocationModel.KE]
                for m in self.relocationModel.M]

        self.relocationModel.decisionVars["Y_MR_Delta_MK"] = []

        for m in self.relocationModel.M:
            self.relocationModel.decisionVars["Y_MR_Delta_MK"].extend(
                    component1[m-1])
            self.relocationModel.decisionVars["Y_MR_Delta_MK"].extend(
                    component2[m-1])

        self.relocationModel.decisionVarsIdxStarts["Y_MR_Delta_MK"] = totalVars
        totalVars = totalVars + len(
                self.relocationModel.decisionVars["Y_MR_Delta_MK"])

        self.relocationModel.variables.add(
                types=([self.relocationModel.variables.type.binary]
                       * len(self.relocationModel.decisionVars[
                               "Y_MR_Delta_MK"])))

        """ CONSTRAINTS """
        totalConstraints = self.relocationModel.linear_constraints.get_num()

        """Probability-adjusted patch covers to account for possibility that
        multiple fires may need to be covered by the aircraft and therefore
        there is a non-zero probability for each configuration possible for a
        patch"""
        self.relocationModel.constraintNames["C_2"] = [
                "C_2_K" + str(k+1) + "_C" + str(c) + "_N" + str(n)
                for k in self.relocationModel.KP
                for c in self.relocationModel.C
                for n in self.relocationModel.N]
        self.relocationModel.constraintIdxStarts["C_2"] = totalConstraints
        totalConstraints = totalConstraints + len(
                self.relocationModel.constraintNames["C_2"])

        lenB = len(self.relocationModel.B)
        startARB = self.relocationModel.decisionVarsIdxStarts["A_RB"]
        startDeltaNK = self.relocationModel.decisionVarsIdxStarts["Delta_NK"]
        varIdxs = {
                (k, c, n):
                [startARB + r*lenB + b
                 for r in self.relocationModel.R
                 for b in self.relocationModel.B]
                + [startDeltaNK + n*lenKP + k]
                for k in range(lenKP)
                for c in self.relocationModel.C
                for n in self.relocationModel.N}

        varCoeffs = {
                (k, c, n):
                [-self.relocationModel.d3_BCN[b, c, n] /
                 self.relocationModel.no_CB[c, b]
                 for r in self.relocationModel.R
                 for b in self.relocationModel.B]
                + [self.relocationModel.Q_KC[self.relocationModel.KP[k], c]]
                for k in range(lenKP)
                for c in self.relocationModel.C
                for n in self.relocationModel.N}

        self.relocationModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[k, c, n],
                                val=varCoeffs[k, c, n])
                        for k in range(len(self.relocationModel.KP))
                        for c in self.relocationModel.C
                        for n in self.relocationModel.N],
                senses=["L"]*(len(varIdxs)),
                rhs=[0]*len(varIdxs))

        """Ensures that the sum of probabilities of applying each configuration
        to a patch is 1"""
        self.relocationModel.constraintNames["C_8"] = [
                "C_8_N" + str(n)
                for n in self.relocationModel.N]
        self.relocationModel.constraintIdxStarts["C_8"] = totalConstraints
        totalConstraints = totalConstraints + len(
                self.relocationModel.constraintNames["C_8"])

        varIdxs = {(n):
                   [startDeltaNK + n*lenKP + k
                    for k in range(len(self.relocationModel.KP))]
                   for n in self.relocationModel.N}

        varCoeffs = {(n): [1]*len(varIdxs[n]) for n in self.relocationModel.N}

        self.relocationModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[n],
                                val=varCoeffs[n])
                        for n in self.relocationModel.N],
                senses=["E"]*len(varIdxs),
                rhs=[1]*len(varIdxs))

    def buildAssignment2(self):
        self.relocationModel.objective.set_sense(
                self.relocationModel.objective.sense.minimize)

        lenKP = len(self.relocationModel.KP)

        """ Initial expected damage increase of each fire under each config """
        self.relocationModel.D1_MK = {
                (m, k+1): 0
                for m in self.relocationModel.M
                for k in self.relocationModel.KE}

        """ Init expected damage increase of each patch under each config """
        self.relocationModel.D2_NK = {
                (n, k+1): 0
                for n in self.relocationModel.N
                for k in self.relocationModel.KP}

        """ DECISION VARIABLES """
        totalVars = self.relocationModel.variables.get_num()

        """ Patch configuration covers """
        self.relocationModel.decisionVars["Delta_NK"] = [
                "Delta_NK_Adj_N" + str(n) + "_K" + str(k+1)
                for n in self.relocationModel.N
                for k in self.relocationModel.KP]
        self.relocationModel.decisionVarsIdxStarts["Delta_NK"] = totalVars
        totalVars = totalVars + len(self.relocationModel.decisionVars[
                "Delta_NK"])

        self.relocationModel.variables.add(
                ub=[1]*len(self.relocationModel.decisionVars["Delta_NK"]),
                lb=[0]*len(self.relocationModel.decisionVars["Delta_NK"]))

        """ Aircraft-fire assignments and fire configuration covers """
        """
        More complex:
        For each fire (created to date)
        1. Y_MR
        2. Delta_MK
        For fires that are extinguished during simulation, the variable's upper
        bound is set to 0. This effectively eliminates the column
        """
        component1 = [
                ["Y_MR_M" + str(m) + "_R" + str(r)
                 for r in self.relocationModel.R]
                for m in self.relocationModel.M]
        component2 = [
                ["Delta_MK_M" + str(m) + "_K" + str(k+1)
                 for k in self.relocationModel.KE]
                for m in self.relocationModel.M]

        self.relocationModel.decisionVars["Y_MR_Delta_MK"] = []

        for m in self.relocationModel.M:
            self.relocationModel.decisionVars["Y_MR_Delta_MK"].extend(
                    component1[m])
            self.relocationModel.decisionVars["Y_MR_Delta_MK"].extend(
                    component2[m])

        self.relocationModel.decisionVarsIdxStarts["Y_MR_Delta_MK"] = totalVars
        totalVars = totalVars + len(
                self.relocationModel.decisionVars["Y_MR_Delta_MK"])

        self.relocationModel.variables.add(
                types=([self.relocationModel.variables.type.binary]
                       * len(self.relocationModel.decisionVars[
                               "Y_MR_Delta_MK"])))

        """ CONSTRAINTS """
        totalConstraints = self.relocationModel.linear_constraints.get_num()

        """Probability-adjusted patch covers to account for possibility that
        multiple fires may need to be covered by the aircraft and therefore
        there is a non-zero probability for each configuration possible for a
        patch"""
        self.relocationModel.constraintNames["C_2"] = [
                "C_2_K" + str(k+1) + "_C" + str(c) + "_N" + str(n)
                for k in self.relocationModel.KP
                for c in self.relocationModel.C
                for n in self.relocationModel.N]
        self.relocationModel.constraintIdxStarts["C_2"] = totalConstraints
        totalConstraints = totalConstraints + len(
                self.relocationModel.constraintNames["C_2"])

        lenB = len(self.relocationModel.B)
        startARB = self.relocationModel.decisionVarsIdxStarts["A_RB"]
        startDeltaNK = self.relocationModel.decisionVarsIdxStarts["Delta_NK"]
        varIdxs = {
                (k, c, n):
                [startARB + r*lenB + b
                 for r in self.relocationModel.R
                 for b in self.relocationModel.B]
                + [startDeltaNK + n*lenKP + k]
                for k in range(lenKP)
                for c in self.relocationModel.C
                for n in self.relocationModel.N}

        varCoeffs = {
                (k, c, n):
                [-self.relocationModel.d3_BCN[b, c, n] /
                 self.relocationModel.no_CB[c, b]
                 for r in self.relocationModel.R
                 for b in self.relocationModel.B]
                + [self.relocationModel.Q_KC[self.relocationModel.KP[k], c]]
                for k in range(lenKP)
                for c in self.relocationModel.C
                for n in self.relocationModel.N}

        self.relocationModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[k, c, n],
                                val=varCoeffs[k, c, n])
                        for k in range(len(self.relocationModel.KP))
                        for c in self.relocationModel.C
                        for n in self.relocationModel.N],
                senses=["L"]*(len(varIdxs)),
                rhs=[0]*len(varIdxs))

        """Ensures that the sum of probabilities of applying each configuration
        to a patch is 1"""
        self.relocationModel.constraintNames["C_8"] = [
                "C_8_N" + str(n)
                for n in self.relocationModel.N]
        self.relocationModel.constraintIdxStarts["C_8"] = totalConstraints
        totalConstraints = totalConstraints + len(
                self.relocationModel.constraintNames["C_8"])

        varIdxs = {(n):
                   [startDeltaNK + n*lenKP + k
                    for k in range(len(self.relocationModel.KP))]
                   for n in self.relocationModel.N}

        varCoeffs = {(n): [1]*len(varIdxs[n]) for n in self.relocationModel.N}

        self.relocationModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[n],
                                val=varCoeffs[n])
                        for n in self.relocationModel.N],
                senses=["E"]*len(varIdxs),
                rhs=[1]*len(varIdxs))

        """Restricts the relocation of aircraft by proximity to bases"""
        self.relocationModel.constraintNames["C_11"] = [
                "C_11_R" + str(r) + "_B" + str(b)
                for r in self.relocationModel.R
                for b in self.relocationModel.B]
        self.relocationModel.constraintIdxStarts["C_11"] = totalConstraints
        totalConstraints = totalConstraints + len(
                self.relocationModel.constraintNames["C_11"])

        startXRB = self.relocationModel.decisionVarsIdxStarts["X_RB"]
        varIdxs = {(r, b): [startXRB + r*lenB + b]
                   for r in self.relocationModel.R
                   for b in self.relocationModel.B}

        varCoeffs = {(r, b): [1]
                     for r in self.relocationModel.R
                     for b in self.relocationModel.B}

        self.relocationModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[r, b],
                                val=varCoeffs[r, b])
                        for r in self.relocationModel.R
                        for b in self.relocationModel.B],
                senses=["L"]*(len(varIdxs)),
                rhs=[0]*len(varIdxs))

    def samplePaths(self):
        region = self.model.getRegion()
        regionSize = region.getX().size
        timeSteps = self.model.getTotalSteps()
        lookahead = self.model.getLookahead()
        noPatches = len(self.model.getRegion().getX())

        simPaths = numpy.empty([self.model.getMCPaths(),
                                self.model.getTotalSteps()
                                + self.model.getLookahead() + 1,
                                noPatches])

        # Generate a sample FFDI path for the region N times
        for path in range(self.model.getMCPaths()):
            rain = numpy.zeros([timeSteps+1+lookahead, regionSize])
            rain[0] = region.getRain()
            precipitation = numpy.zeros([timeSteps+1+lookahead, regionSize])
            precipitation[0] = region.getHumidity()
            temperatureMin = numpy.zeros([timeSteps+1+lookahead, regionSize])
            temperatureMin[0] = region.getTemperatureMin()
            temperatureMax = numpy.zeros([timeSteps+1+lookahead, regionSize])
            temperatureMax[0] = region.getTemperatureMax()
            windNS = numpy.zeros([timeSteps+1+lookahead, regionSize])
            windNS[0] = region.getWindN()
            windEW = numpy.zeros([timeSteps+1+lookahead, regionSize])
            windEW[0] = region.getWindE()
            FFDI = numpy.zeros([timeSteps+1+lookahead, regionSize])
            FFDI[0] = region.getDangerIndex()
            windRegimes = numpy.zeros([timeSteps+1+lookahead])
            windRegimes[0] = region.getWindRegime()

            wg = region.getWeatherGenerator()

            # Simulate the path forward from time zero to the end
            for ii in range(timeSteps+lookahead):
                # Compute weather
                wg.computeWeather(rain, precipitation, temperatureMin,
                                  temperatureMax, windRegimes, windNS, windEW,
                                  FFDI, ii)

            simPaths[path] = FFDI

        root = ("../Experiments/Experiments/" +
                self.model.getInputFile()
                .split("../Experiments/Experiments/")[1]
                .split("/")[0])

        # Now write out the results to a csv file for later use
        with open(root + "/Sample_Paths.csv", 'w', newline='') as sp:
            writer = csv.writer(sp, delimiter=",")

            writer.writerow(
                    ["SAMPLE_FFDI_PATHS"]
                    + [""]*(self.model.getTotalSteps()
                            + self.model.getLookahead()))

            writer.writerow(
                    ["PRESET_PATHS"] + [str(self.model.getMCPaths())]
                    + [""]*(self.model.getTotalSteps()
                            + self.model.getLookahead() - 1))

            for path in range(self.model.getMCPaths()):
                writer.writerow(
                        ["PATH_" + str(path + 1)] + ["HOUR"]
                        + [""]*(self.model.getTotalSteps()
                                + self.model.getLookahead() - 1))

                writer.writerow(
                        ["PATCH"]
                        + [t for t in range(
                                1 + self.model.getTotalSteps()
                                + self.model.getLookahead())])

                for patch in range(noPatches):
                    writer.writerow(
                            [str(patch + 1)]
                            + [simPaths[path][tt, patch]
                               for tt in range(self.model.getTotalSteps() + 1
                                               + self.model.getLookahead())])

    def simulateMPC(self, static=False, rov=False):
        # Computes a Model Predictive Control approach for reallocating
        # aerial resources for fighting fires. The input conditional
        # probabilities are provided as inputs to the program. We just run the
        # simulations to determine the expected end area of the fire for the
        # given input conditions and what the variance around this is.
        region = self.model.getRegion()
        timeSteps = self.model.getTotalSteps()
        lookahead = self.model.getLookahead()
        stepSize = self.model.getStepSize()
        patches = len(region.getPatches())
        bases = len(region.getStations()[0])
        vegetations = self.model.getRegion().getVegetations()
        resources = region.getResources()
        fires = region.getFires()
        noControls = len(self.model.getControls())
        configsE = self.model.getUsefulConfigurationsExisting()
        configsP = self.model.getUsefulConfigurationsPotential()
        sampleFFDIs = self.model.getSamplePaths()
        method = self.model.getControlMethod()

        # Thresholds/parameters for each control
        lambdas = numpy.array([[
                self.model.getControls()[ii].getLambda1(),
                self.model.getControls()[ii].getLambda2()]
            for ii in range(len(self.model.getControls()))],
            dtype=numpy.float32)

        # We need to set up a number of input arrays if we are using GPU-based
        # methods
        if self.model.getAlgo() in [4, 5]:
            mcPaths = self.model.getMCPaths()
            patchVegetations = region.getVegetation().astype(numpy.int32)

            patchAreas = numpy.array(
                [patch.getArea() for patch in patches], dtype=numpy.float32)

            patchCentroids = numpy.array(
                [patch.getCentroid() for patch in patches],
                dtype=numpy.float32)

            baseLocations = numpy.array(
                [base.getLocation() for base in range(bases)],
                dtype=numpy.float32)

            resourceTypes = numpy.array(
                [0 if type(resource).__name__ == 'Tanker'
                 else (1 if type(resource).__name__ == 'Heli'
                 else 2 if type(resource).__name__ == 'Land'
                 else -1)
                 for resource in resources],
                 dtype=numpy.int32)

            resourceSpeeds = numpy.array(
                [resource.getSpeed() for resource in resources],
                dtype=numpy.float32)

            maxHours = numpy.array([resource.getMaxDailyHours()
                                    for resource in resources],
                                   dtype=numpy.float32)

            configurations = numpy.array(
                [self.model.configurations[config]
                 for config in self.model.configurations],
                 dtype=numpy.int32)

            ffdiRanges = numpy.array(
                [vegetation.getFFDIRange() for vegetation in vegetations],
                dtype=numpy.float32)

            rocA2PHMeans = numpy.array(
                [[vegetation.getROCA2PerHourMean()[config]
                  for config, _ in self.model.configurations.items()]
                 for vegetation in vegetations],
                dtype=numpy.float32)

            rocA2PHSDs = numpy.array(
                [[vegetation.getROCA2PerHourSD()[config]
                  for config, _ in self.model.configurations.items()]
                 for vegetation in vegetations],
                dtype=numpy.float32)

            occurrence = numpy.array(
                [[vegetation.getOccurrence()[time]
                  for time in range(timeSteps + lookahead + 1)]
                 for vegetation in vegetations],
                dtype=numpy.float32)

            initSizeM = numpy.array(
                [[vegetation.getInitialSizeMean()[config]
                  for config, _ in self.model.configurations.items()]
                 for vegetation in vegetations],
                dtype=numpy.float32)

            initSizeSD = numpy.array(
                [[vegetation.getInitialSizeSD()[config]
                  for config, _ in self.model.configurations.items()]
                 for vegetation in vegetations],
                dtype=numpy.float32)

            initSuccess = numpy.array(
                [[vegetation.getInitialSuccess()[config]
                  for config, _ in self.model.configurations.items()]
                 for vegetation in vegetations],
                dtype=numpy.float32)

            extSuccess = numpy.array(
                [[vegetation.getExtendedSuccess()[config]
                  for config, _ in self.model.configurations.items()]
                 for vegetation in vegetations],
                dtype=numpy.float32)

            # This is for the configurations components
            thresholds = numpy.array([self.model.fireThreshold,
                                      self.model.baseThreshold],
                                     dtype=numpy.float32)

            tankerDists = numpy.array(
                [[self.geoDist(patchCentroids[patch], baseLocations[base])
                  / resourceSpeeds[0]
                  for base in range(bases)]
                 for patch in range(patches)],
                dtype=numpy.float32)

            heliDists = numpy.array(
                [[self.geoDist(patchCentroids[patch], baseLocations[base])
                  / resourceSpeeds[1]
                  for base in range(bases)]
                 for patch in range(patches)],
                dtype=numpy.float32)

        """ Initial assignment of aircraft to bases (Col 1) and fires (Col 2)
        A value of zero indicates no assignment (only applicable for fires) """
        assignments = self.model.getRegion().getAssignments()

        regionSize = region.getX().size
        samplePaths = (
                len(sampleFFDIs)
                if len(sampleFFDIs) > 0
                else self.model.getRuns())
        samplePaths2 = self.model.getMCPaths()
        lookahead = self.model.getLookahead()
        runs = self.model.getRuns()

        self.finalDamageMaps = [None]*samplePaths
        self.expectedDamages = [None]*samplePaths
        self.realisedAssignments = [None]*samplePaths
        self.realisedFires = [None]*samplePaths
        self.realisedFFDIs = [None]*samplePaths
        self.aircraftHours = [None]*samplePaths
        self.realisedStarts = [None]*samplePaths
        self.initialExtinguished = [None]*samplePaths

        if self.model.getAlgo() in [4, 5]:
            self.dynamicC2G = [None]*samplePaths

        wg = region.getWeatherGenerator()

        for ii in range(samplePaths):
            self.finalDamageMaps[ii] = [None]*runs
            self.expectedDamages[ii] = [None]*runs
            self.realisedAssignments[ii] = [None]*runs
            self.realisedFires[ii] = [None]*runs
            self.realisedFFDIs[ii] = [None]*runs
            self.aircraftHours[ii] = [None]*runs
            self.realisedStarts[ii] = [None]*runs
            self.initialExtinguished[ii] = [None]*runs

            if self.model.getAlgo() in [4, 5]:
                self.dynamicC2G[ii] = [None]*runs

                # Whether bases are within 20 minutes for each aircraft type
                tankerCoversE = numpy.array(
                    [[True if tankerDists[patch, base] <= thresholds[0]
                        else False
                      for base in range(bases)]
                     for patch in range(patches)])

                tankerCoversL = numpy.array(
                    [[True if (tankerDists[patch, base] > thresholds[0]
                               and tankerDists[patch, base] <= thresholds[1])
                        else False
                      for base in range(bases)]
                     for patch in range(patches)])

                heliCoversE = numpy.array(
                    [[True if heliDists[patch, base] <= thresholds[0]
                        else False
                      for base in range(bases)]
                     for patch in range(patches)])

                heliCoversL = numpy.array(
                    [[True if (heliDists[patch, base] > thresholds[0]
                               and heliDists[patch, base] <= thresholds[1])
                        else False
                      for base in range(bases)]
                     for patch in range(patches)])

                # Min no. of aircraft needed for each component of each config
                baseConfigsMax = numpy.array([0, 0, 0, 0], dtype=numpy.int32)
                fireConfigsMax = numpy.array([0, 0, 0, 0], dtype=numpy.int32)

                # Maximum number in each component based on allowed configs
                for c in range(len(configurations)):
                    if c+1 in configsP:
                        if configurations[int(c)][0] > baseConfigsMax[0]:
                            baseConfigsMax[0] += 1
                        if configurations[int(c)][1] > baseConfigsMax[1]:
                            baseConfigsMax[1] += 1
                        if configurations[int(c)][2] > baseConfigsMax[2]:
                            baseConfigsMax[2] += 1
                        if configurations[int(c)][3] > baseConfigsMax[3]:
                            baseConfigsMax[3] += 1

                    if c+1 in configsE:
                        if configurations[int(c)][0] > fireConfigsMax[0]:
                            fireConfigsMax[0] += 1
                        if configurations[int(c)][1] > fireConfigsMax[1]:
                            fireConfigsMax[1] += 1
                        if configurations[int(c)][2] > fireConfigsMax[2]:
                            fireConfigsMax[2] += 1
                        if configurations[int(c)][3] > fireConfigsMax[3]:
                            fireConfigsMax[3] += 1

                # Expected fires for each patch at each time step
                expectedFires = numpy.array([[
                        numpy.interp(
                            sampleFFDIs[ii][patch][t],
                            ffdiRanges[int(patchVegetations[patch])],
                            occurrence[int(patchVegetations[patch])][t]) *
                            patchAreas[patch]
                        for patch in range(patches)]
                       for t in range(timeSteps + lookahead)])

                # Expected visible fires for each component for each patch at each
                # time step. As FFDI is assumed deterministic, this array is used
                # for all paths
                expFiresComp = numpy.array([[
                        numpy.matmul(expectedFires[t], tankerCoversE),
                        numpy.matmul(expectedFires[t], tankerCoversL),
                        numpy.matmul(expectedFires[t], heliCoversE),
                        numpy.matmul(expectedFires[t], heliCoversL)]
                    for t in range(timeSteps + lookahead)])

            for run in range(self.model.getRuns()):
                if self.model.getAlgo() in [4, 5]:
                    self.dynamicC2G[ii][run] = numpy.zeros([
                        self.model.getTotalSteps(), noControls, samplePaths2])

                    aircraftLocations = numpy.zeros([mcPaths, timeSteps + 1,
                                             len(resources), 2],
                                             dtype=numpy.float32)
                    aircraftLocations[:, 0, :] = ([[
                            aircraft.getLocation()
                            for aircraft in resources]]
                            *mcPaths)

                    aircraftAssignments = numpy.zeros([mcPaths, time + 1,
                                                       len(resources), 2],
                                                       dtype=numpy.int32)
                    aircraftAssignments[:, 0, :] = [[
                            [resource.getBase() + 1, 0]
                            for resource in resources
                        ]]*mcPaths

                    fireSizes = numpy.zeros([mcPaths, timeSteps + 1, 500],
                                    dtype=numpy.float32)

                    fireSizes[:, 0, 0:len(fires)] = [[
                            fire.getSize() for fire in fires]]*mcPaths

                    fireLocations = numpy.zeros([mcPaths, timeSteps, 500, 2],
                                        dtype=numpy.float32)
                    fireLocations[:, 0, 0:len(fires), :] = [[
                            fire.getLocation() for fire in fires]]

                    firePatches = numpy.zeros([mcPaths, timeSteps, 500],
                                              dtype=numpy.int32)
                    firePatches[:, 0, 0:len(fires)] = [[
                            fire.getPatchID() for fire in fires]]

                damage = 0
                assignmentsPath = [None]*(timeSteps + 1)
                assignmentsPath[0] = copy.copy(assignments)
                firesPath = copy.copy(fires)
                resourcesPath = copy.copy(resources)
                activeFires = [fire for fire in firesPath]
                self.realisedFires[ii][run] = [None]*(timeSteps + 1)
                self.realisedFires[ii][run][0] = copy.copy(activeFires)
                self.realisedStarts[ii][run] = [None]*(timeSteps + 1)
                self.realisedStarts[ii][run][0] = copy.copy(activeFires)
                self.initialExtinguished[ii][run] = [None]*(timeSteps + 1)
                self.initialExtinguished[ii][run][0] = copy.copy(activeFires)
                self.finalDamageMaps[ii][run] = numpy.empty([timeSteps + 1,
                                                             patches])
                self.finalDamageMaps[ii][run][0] = numpy.zeros([patches])
                self.aircraftHours[ii][run] = numpy.zeros([timeSteps + 1,
                                                           len(resources)])

                rain = numpy.zeros([timeSteps+1+lookahead, regionSize])
                rain[0] = region.getRain()
                precipitation = numpy.zeros([timeSteps+1+lookahead,
                                             regionSize])
                precipitation[0] = region.getHumidity()
                temperatureMin = numpy.zeros([timeSteps+1+lookahead,
                                              regionSize])
                temperatureMin[0] = region.getTemperatureMin()
                temperatureMax = numpy.zeros([timeSteps+1+lookahead,
                                              regionSize])
                temperatureMax[0] = region.getTemperatureMax()
                windNS = numpy.zeros([timeSteps+1+lookahead, regionSize])
                windNS[0] = region.getWindN()
                windEW = numpy.zeros([timeSteps+1+lookahead, regionSize])
                windEW[0] = region.getWindE()
                FFDI = numpy.zeros([timeSteps+1+lookahead, regionSize])
                FFDI[0] = region.getDangerIndex()
                windRegimes = numpy.zeros([timeSteps+1+lookahead])
                windRegimes[0] = region.getWindRegime()
                accumulatedDamage = numpy.zeros([timeSteps+1, patches])
                accumulatedHours = numpy.zeros([timeSteps+1, len(resources)])

                """ If not MPC, we need to know the initial assignments of
                aircraft based on the ENTIRE horizon """
                if static:
                    expectedFFDI = sampleFFDIs[ii]
                    expDamageExist = {
                            (jj, configsE[kk] - 1):
                            self.expectedDamageExisting(
                                    activeFires[jj], expectedFFDI,
                                    configsE[kk], 0,
                                    self.model.getTotalSteps() + lookahead)
                            for jj in range(len(activeFires))
                            for kk in range(len(configsE))}

                    expDamagePoten = {
                            (jj, configsP[kk] - 1):
                            self.expectedDamagePotential(
                                    jj, expectedFFDI, configsP[kk], 0,
                                    self.model.getTotalSteps() + lookahead)
                            for jj in range(patches)
                            for kk in range(len(configsP))}

                    [patchConfigs, fireConfigs] = (
                            self.assignAircraft(
                                    assignmentsPath, expDamageExist,
                                    expDamagePoten, activeFires, resourcesPath,
                                    expectedFFDI, 0, 1, method, static))

                    self.fixBaseAssignments(assignmentsPath[0])

                for tt in range(timeSteps):
                    if len(sampleFFDIs) == 0:
                        rainTemp = numpy.zeros([
                                timeSteps + lookahead + 1, regionSize])
                        rainTemp[tt] = rain[tt]
                        precipitationTemp = numpy.zeros([
                                timeSteps + lookahead + 1, regionSize])
                        precipitationTemp[tt] = precipitation[tt]
                        temperatureMinTemp = numpy.zeros([
                                timeSteps + lookahead + 1, regionSize])
                        temperatureMinTemp[tt] = temperatureMin[tt]
                        temperatureMaxTemp = numpy.zeros([
                                timeSteps + lookahead + 1, regionSize])
                        temperatureMaxTemp[tt] = temperatureMax[tt]
                        windNSTemp = numpy.zeros([
                                timeSteps + lookahead + 1, regionSize])
                        windNSTemp[tt] = windNS[tt]
                        windEWTemp = numpy.zeros([
                                timeSteps + lookahead + 1, regionSize])
                        windEWTemp[tt] = windEW[tt]
                        FFDITemp = numpy.zeros([
                                timeSteps + lookahead + 1, regionSize])
                        FFDITemp[tt] = FFDI[tt]
                        windRegimesTemp = numpy.zeros([timeSteps
                                                       + lookahead + 1])
                        windRegimesTemp[tt] = windRegimes[tt]

                        FFDISamples = numpy.zeros([
                                samplePaths2, lookahead, regionSize])

                        # This part is hard as it requires many simulations to
                        # achieve convergence in the expected FFDIs across the
                        # region
                        for pp in range(samplePaths2):
                            for ll in range(tt + 1, tt + lookahead + 1):
                                wg.computeWeather(
                                    rainTemp, precipitationTemp,
                                    temperatureMinTemp, temperatureMaxTemp,
                                    windRegimesTemp, windNSTemp, windEWTemp,
                                    FFDITemp, ll)

                                FFDISamples[pp, ll, :] = FFDITemp[ll]

                        # Compute the expected FFDI at each time step for each
                        # patch
                        expectedFFDI = FFDISamples.sum(0)/len(samplePaths2)

                    else:
                        expectedFFDI = sampleFFDIs[ii]

                    """ Compute the new assignments. If static, only fire
                    assignments are computed here. Otherwise, we also compute
                    relocations """
                    """ The expected damages are for the rolling horizon, not
                    the full horizon """
                    expDamageExist = {
                            (jj, configsE[kk] - 1):
                            self.expectedDamageExisting(
                                    activeFires[jj], expectedFFDI,
                                    configsE[kk], tt, lookahead)
                            for jj in range(len(activeFires))
                            for kk in range(len(configsE))}

                    expDamagePoten = {
                            (jj, configsP[kk] - 1):
                            self.expectedDamagePotential(
                                    jj, expectedFFDI, configsP[kk], tt,
                                    lookahead)
                            for jj in range(patches)
                            for kk in range(len(configsP))}

                    # Assign aircraft using LP
                    # If this is for the static case, only fire assignments are
                    # considered
                    if self.model.getAlgo() in [2, 4, 5]:
                        # This is the ROV case so we need to use the
                        # regressions provided earlier
                        bestControl = 0
                        bestC2G = math.inf

                        if self.model.getAlgo() == 2:
                            # ROV
                            if tt == timeSteps - 1:
                                pass
                            else:
                                [s1, s2, s3] = self.computeState(
                                    assignmentsPath, resourcesPath,
                                    expDamageExist, expDamagePoten,
                                    activeFires, configsP, configsE,
                                    expectedFFDI, accumulatedHours, method,
                                    tt + 1)

                                if tt > 0:
                                    for c in range(noControls):
                                        # Use the current states to interpolate
                                        # the data representing the
                                        # expectations
                                        poly_reg = PolynomialFeatures(degree=2)
                                        predict_ = poly_reg.fit_transform(
                                                numpy.array([s1, s2, s3])
                                                .reshape(1, -1))
                                        currC2G = (self.regModel[ii][tt, c]
                                                   .predict(predict_))

                                        if currC2G < bestC2G:
                                            bestC2G = currC2G
                                            bestControl = c

                                else:
                                    bestControl = self.controls[ii][0,0]

                        elif self.model.getAlgo() in [4, 5]:
                            # Dynamic MPC approaches

                            if self.model.getAlgo() == 4:
                                # Dynamic MPC (fixed control)
                                for c in range(noControls):
                                    # Use the current states to interpolate the
                                    # data representing the expectations
                                    SimulationNumba.simulateMPCDyn(
                                        mcPaths, sampleFFDIs[ii],
                                        patchVegetations, patchAreas,
                                        patchCentroids, baseLocations,
                                        resourceTypes, resourceSpeeds,
                                        maxHours, configurations, configsE,
                                        configsP, thresholds, ffdiRanges,
                                        rocA2PHMeans, rocA2PHSDs, occurrence,
                                        initSizeM, initSizeSD, initSuccess,
                                        extSuccess, tankerDists, heliDists,
                                        fireConfigsMax, baseConfigsMax,
                                        expFiresComp, timeSteps, lookahead,
                                        stepSize, accumulatedDamage,
                                        accumulatedHours, fires, fireSizes,
                                        fireLocations, firePatches,
                                        aircraftLocations, aircraftAssignments,
                                        noControls, self.dynamicC2G[ii][run],
                                        lambdas, method, noControls, False, c)

                                    currC2G = self.dynamicC2G[ii][run][:, c].mean()

                                    if currC2G < bestC2G:
                                        bestC2G = currC2G
                                        bestControl = c

                            elif self.model.getAlgo() == 5:
                                # Dynamic MPC (random control)
                                for c in range(noControls):
                                    # Use the current states to interpolate the
                                    # data representing the expectations
                                    SimulationNumba.simulateMPCDyn(
                                        mcPaths, sampleFFDIs[ii],
                                        patchVegetations, patchAreas,
                                        patchCentroids, baseLocations,
                                        resourceTypes, resourceSpeeds,
                                        maxHours, configurations, configsE,
                                        configsP, thresholds, ffdiRanges,
                                        rocA2PHMeans, rocA2PHSDs, occurrence,
                                        initSizeM, initSizeSD, initSuccess,
                                        extSuccess, tankerDists, heliDists,
                                        fireConfigsMax, baseConfigsMax,
                                        expFiresComp, timeSteps, lookahead,
                                        stepSize, accumulatedDamage,
                                        accumulatedHours, fires, fireSizes,
                                        fireLocations, firePatches,
                                        aircraftLocations, aircraftAssignments,
                                        noControls, self.dynamicC2G[ii][run],
                                        lambdas, method, noControls, True, c)

                                    currC2G = self.dynamicC2G[ii][run][:, c].mean()

                                    if currC2G < bestC2G:
                                        bestC2G = currC2G
                                        bestControl = c

                    else:
                        bestControl = 0

#                    print(self.relocationModel.decisionVarsIdxStarts)
#                    print(self.relocationModel.constraintIdxStarts)
                    if bestControl < noControls:
                        [patchConfigs, fireConfigs] = (
                                self.assignAircraft(
                                        assignmentsPath, expDamageExist,
                                        expDamagePoten, activeFires,
                                        resourcesPath, expectedFFDI, tt + 1,
                                        bestControl, method,
                                        staticAfter=static))
                    else:
                        print("Control Idx too high")
                        print(bestControl)
                        sys.exit()

#                    [s1, s2, s3] = self.computeState(
#                            assignmentsPath, resourcesPath, expDamageExist,
#                            expDamagePoten, activeFires, configsP,
#                            configsE, expectedFFDI, accumulatedHours,
#                            method, tt + 1)
                    # Save the active fires to the path history
                    self.realisedFires[ii][run][tt + 1] = copy.copy(
                            activeFires)

                    # Simulate the fire growth, firefighting success and the
                    # new positions of each resources
                    damage += self.simulateSinglePeriod(
                            assignmentsPath, resourcesPath, firesPath,
                            activeFires, accumulatedDamage, accumulatedHours,
                            patchConfigs, fireConfigs, FFDI[tt], tt)

                    self.aircraftHours[ii][run][tt + 1] = numpy.array([
                            resourcesPath[r].getFlyingHours()
                            for r in range(len(
                                    self.model.getRegion().getResources()))])

                    # Simulate the realised weather for the next time step
                    if len(sampleFFDIs) == 0:
                        wg.computeWeather(
                                rain, precipitation, temperatureMin,
                                temperatureMax, windRegimes, windNS, windEW,
                                FFDI, tt)
                    else:
                        FFDI[tt + 1] = sampleFFDIs[ii][:, tt + 1]

                # Store the output results
                self.finalDamageMaps[ii][run] = accumulatedDamage
                self.expectedDamages[ii][run] = damage
                self.realisedAssignments[ii][run] = assignmentsPath
                self.realisedFFDIs[ii][run] = FFDI
                self.aircraftHours[ii][run] = accumulatedHours

        """ Prepare for write out """
        maxFFDI = 0.0
        minFFDI = math.inf
        maxDamage = 0.0
        minDamage = math.inf

        for ii in range(len(self.realisedFFDIs)):
            for jj in range(self.model.getRuns()):
                if self.realisedFFDIs[ii][jj].max() > maxFFDI:
                    maxFFDI = self.realisedFFDIs[ii][jj].max()

                if self.realisedFFDIs[ii][jj].min() < minFFDI:
                    minFFDI = self.realisedFFDIs[ii][jj].min()

        for ii in range(len(self.finalDamageMaps)):
            for jj in range(self.model.getRuns()):
                if self.finalDamageMaps[ii][jj].max() > maxDamage:
                    maxDamage = self.finalDamageMaps[ii][jj].max()

                if self.finalDamageMaps[ii][jj].min() < minDamage:
                    minDamage = self.finalDamageMaps[ii][jj].min()

        """Save the results for this sample"""
        for ii in range(samplePaths):
            for run in range(self.model.getRuns()):
                self.writeOutResults(maxFFDI, minFFDI, maxDamage, minDamage,
                                     ii, run)

        self.writeOutSummary()

    def fixBaseAssignments(self, assignments):
        """ Non-assignments """
        self.relocationModel.variables.set_upper_bounds([(
                (self.relocationModel.decisionVarsIdxStarts["X_RB"] +
                 r*len(self.relocationModel.B) + b,
                 1 if assignments[r, 0] == b + 1 else 0))
                for r in self.relocationModel.R
                for b in self.relocationModel.B])

        """ Fixed assignments """
        self.relocationModel.variables.set_lower_bounds([
                (self.relocationModel.decisionVarsIdxStarts["X_RB"] +
                 r*len(self.relocationModel.B) + b,
                 1 if assignments[r, 0] == b + 1 else 0)
                for r in self.relocationModel.R
                for b in self.relocationModel.B])

    def assignAircraft(self, assignmentsPath, expDamageExist,
                       expDamagePoten, activeFires, resourcesPath, ffdiPath,
                       timeStep, control=0, method=1, static=False, dummy=0,
                       staticAfter=False):

        """First compute the parts common to all relocation programs"""
        """ First copy the relocation model """
        tempModel = cplex.Cplex(self.relocationModel)
        self.copyNonCplexComponents(tempModel, self.relocationModel)
        bases = self.model.getRegion().getStations()[0]

        """ Set the fire details for the sets and decision variables """
        tempModel.M = [ii for ii in range(len(activeFires))]

        """ Cumulative flying hours """
        tempModel.G_R = {
                r: resourcesPath[r].getFlyingHours()
                for r in tempModel.R}

        """ Travel times between resource R and base B """
        tempModel.d1_RB = {
                (r, b): self.geoDist(bases[b].getLocation(),
                                     resourcesPath[r].getLocation()) /
                        resourcesPath[r].getSpeed()
                for r in tempModel.R
                for b in tempModel.B}

        """ Travel times between resource R and fire M """
        tempModel.d2_RM = {
                (r, m): self.geoDist(activeFires[m].getLocation(),
                                     resourcesPath[r].getLocation()) /
                 resourcesPath[r].getSpeed()
                for r in tempModel.R
                for m in tempModel.M}

        """ Expected number of fires for patch N over horizon """
        look = (self.model.getLookahead() + self.model.getTotalSteps()
                if static
                else self.model.getLookahead())

        tempModel.no_N = {
                n:
                sum([numpy.interp(ffdiPath[n, t],
                                  self.model.getRegion().getPatches()[n].
                                  getVegetation().getFFDIRange(),
                                  self.model.getRegion().getPatches()[n].
                                  getVegetation().getOccurrence()[
                                          timeStep + t]) *
                     self.model.getRegion().getPatches()[n].getArea()
                     for t in range(look)])
                for n in tempModel.N}

        switch = {
            1: self.assignMaxCover,
            2: self.assignPMedian,
            3: self.assignAssignment2,
            4: self.assignAssignment1
        }

        lpModel = self.model.getNestedOptMethod()
        prog = switch.get(lpModel)
        return prog(assignmentsPath, expDamageExist, expDamagePoten,
                    activeFires, resourcesPath, ffdiPath, timeStep, control,
                    static, tempModel, method, dummy, staticAfter)

    def assignMaxCover(self, assignmentsPath, expDamageExist, expDamagePoten,
                       activeFires, resourcesPath, ffdiPath, timeStep, control,
                       static, tempModel, method, dummy=0):

        """ First copy the relocation model """
        bases = self.model.getRegion().getStations()[0]
        patches = self.model.getRegion().getPatches()

        """////////////////////////////////////////////////////////////////////
        /////////////////////////// DECISION VARIABLES ////////////////////////
        ////////////////////////////////////////////////////////////////////"""

        """ Set lengths """
        lenR = len(tempModel.R)
        lenB = len(tempModel.B)
        lenM = len(tempModel.M)

        """ Thresholds """
        if method == 1:
            maxB = tempModel.lambdas[control][1]
            maxF = tempModel.lambdas[control][0]
            fw = 1
            bw = 1
        elif method == 2:
            fw = tempModel.lambdas[control][0]
            bw = 1 - tempModel.lambdas[control][1]
            maxB = tempModel.lambdas[control][1]
            maxF = math.inf

        """ Aircraft-fire assignments and fire configuration covers """
        """
        For each fire (created to date)
        1. Y_MR
        2. Z_CM
        For fires that are extinguished during simulation, the variable's upper
        bound is set to 0. This effectively eliminates the column
        """
        component1 = [
                ["Y_MR_M" + str(m) + "_R" + str(r)
                 for r in tempModel.R]
                for m in tempModel.M]
        component2 = [
                ["Z_CM" + str(m)]
                for m in tempModel.M]

        tempModel.decisionVars["Y_MR_Z_CM"] = []

        for m in tempModel.M:
            tempModel.decisionVars["Y_MR_Z_CM"].extend(component1[m-1])
            tempModel.decisionVars["Y_MR_Z_CM"].extend(component2[m-1])

        totalVars = tempModel.variables.get_num()
        tempModel.decisionVarsIdxStarts["Y_MR_Z_CM"] = totalVars
        totalVars = totalVars + len(tempModel.decisionVars["Y_MR_Z_CM"])

        tempModel.variables.add(
                types=([tempModel.variables.type.binary]
                       * len(tempModel.decisionVars["Y_MR_Z_CM"])))

        """////////////////////////////////////////////////////////////////////
        ///////////////////////////// PARAMETERS //////////////////////////////
        ////////////////////////////////////////////////////////////////////"""
        """Expected damage of existing fires if uncontrolled burn"""
        tempModel.D1_M = expDamageExist[0]

        """Expected damage of potential fires if uncontrolled burn"""
        tempModel.D2_N = expDamagePoten[0]

        """ Expected number of fires for patch N over horizon """
        look = (self.model.getLookahead() + self.model.getTotalSteps()
                if static
                else self.model.getLookahead())

        tempModel.no_N = {
                n:
                sum([numpy.interp(ffdiPath[n, t],
                                  self.model.getRegion().getPatches()[n].
                                  getVegetation().getFFDIRange(),
                                  self.model.getRegion().getPatches()[n].
                                  getVegetation().getOccurrence()[
                                          timeStep + t]) *
                     self.model.getRegion().getPatches()[n].getArea()
                     for t in range(look)])
                for n in tempModel.N}

        """ Whether base B satisfies the maximum relocation distance for
        resource R for the designated control """
        closestBase = numpy.zeros(len(tempModel.R))

        for r in tempModel.R:
            closestDist = math.inf

            for b in tempModel.B:
                if tempModel.d1_RB[r, b] < closestDist:
                    closestBase[r] = b
                    closestDist = tempModel.d1_RB[r, b]

        tempModel.d1_RB_B2 = {
                (r, b): (1 if ((tempModel.d1_RB[r, b]) <= maxB
                               or b == closestBase[r])
                         else 0)
                for r in tempModel.R
                for b in tempModel.B}

        """ Whether fire M satisfies the maximum relocation distance for
        resources R for the designated control """
        tempModel.d2_MR_B1 = {
                (m, r):
                (1 if (tempModel.d2_RM[r, m]) <= maxF
                    else 0)
                for m in tempModel.M
                for r in tempModel.R}

        """ Travel times between base B and patch N for component C"""
        tempModel.d4_BNC = {
                (b, n, c): (self.geoDist(patches[n].getCentroid(),
                                         bases[b].getLocation()) /
                 (self.model.getResourceTypes()[0].getSpeed() if c in [1, 3]
                  else self.model.getResourceTypes()[1].getSpeed()))
                for b in tempModel.B
                for n in tempModel.N
                for c in [1, 2]}

        """ Expected number of fires visible by base B for aircraft type C """
        tempModel.no_CB = {
                (c, b):
                max(1,
                    sum([tempModel.no_N[n]
                         if (((c == 1 or c == 3)
                              and tempModel.d4_BNC[b, n, c] <=
                                  tempModel.thresholds[0])
                             or (c == 2 or c == 4
                                 and tempModel.d4_BNC[b, n, c] >
                                 tempModel.thresholds[0]
                                 and tempModel.d4_BNC[b, n, c] <=
                                 tempModel.thresholds[1]))
                         else 0
                         for n in tempModel.N]))
                for c in [1, 2, 3, 4]
                for b in tempModel.B}

        """////////////////////////////////////////////////////////////////////
        ///////////////////// OBJECTIVE VALUE COEFFICIENTS ////////////////////
        ////////////////////////////////////////////////////////////////////"""

        startZN = tempModel.decisionVarsIdxStarts["Z_N"]
        startZM = tempModel.decisionVarsIdxStarts["Y_MR_Z_CM"]
        startXRB = tempModel.decisionVarsIdxStarts["X_RB"]

        if len(tempModel.M) > 0:
            tempModel.objective.set_linear(list(zip(
                    [startZM + m*(lenR + 1) + lenR
                     for m in tempModel.M],
                    [fw * tempModel.D1_M[m]
                     for m in tempModel.M])))

        tempModel.objective.set_linear(list(zip(
                [startZN + n
                 for n in tempModel.N],
                [bw * tempModel.D2_N[n]
                 for n in tempModel.N])))

        """////////////////////////////////////////////////////////////////////
        //////////////////////////// CONSTRAINTS //////////////////////////////
        ////////////////////////////////////////////////////////////////////"""
        totalConstraints = tempModel.linear_constraints.get_num()

        """ Limits the fires that each aircraft is allowed to attend """
        tempModel.constraintNames["C_10"] = [
                "C_10_M" + str(m) + "_R" + str(r)
                for m in tempModel.M
                for r in tempModel.R]
        tempModel.constraintIdxStarts["C_10"] = totalConstraints
        totalConstraints = totalConstraints + len(
                tempModel.constraintNames["C_10"])

        varIdxs = {(m, r): [startZM + m*(lenR + 1) + r]
                   for m in tempModel.M
                   for r in tempModel.R}

        varCoeffs = {(m, r): [1]*len(varIdxs[m, r])
                     for m in tempModel.M
                     for r in tempModel.R}

        tempModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[m, r],
                                val=varCoeffs[m, r])
                        for m in tempModel.M
                        for r in tempModel.R],
                senses = ["L"]*len(varIdxs),
                rhs=[tempModel.d2_MR_B1[m, r]
                        for m in tempModel.M
                        for r in tempModel.R])

        """ CONSTRAINT 4 """
        startC4 = tempModel.constraintIdxStarts["C_4"]

        if len(tempModel.M) > 0:
            coefficients = [
                    (startC4 + r, startZM + m*(lenR + 1) + r, 1)
                    for r in tempModel.R
                    for m in tempModel.M]

            tempModel.linear_constraints.set_coefficients(coefficients)

        """ CONSTRAINT 6 """
        startC6 = tempModel.constraintIdxStarts["C_6"]

        if len(tempModel.M) > 0:
            coefficients = [
                    (startC6 + r,
                     startZM + m*(lenR + 1) + r,
                     1)
                    for r in tempModel.R
                    for m in tempModel.M]

            tempModel.linear_constraints.set_coefficients(coefficients)

        """ CONSTRAINT 9 """
        startC9 = tempModel.constraintIdxStarts["C_9"]
        coefficients = []
        if len(tempModel.M) > 0:
            coefficients.extend([
                    (startC9 + r,
                     startZM + m*(lenR + 1) + r,
                     tempModel.d2_RM[r, m])
                    for r in tempModel.R
                    for m in tempModel.M])

        coefficients.extend([
                (startC9 + r,
                 startXRB + r*lenB + b,
                 tempModel.d1_RB[r, b])
                for r in tempModel.R
                for b in tempModel.B])

        tempModel.linear_constraints.set_coefficients(coefficients)

        tempModel.linear_constraints.set_rhs([
                (startC9 + r,
                 tempModel.Gmax_R[r] - tempModel.G_R[r])
                for r in tempModel.R])

        """ CONSTRAINT 11 """
        startC11 = tempModel.constraintIdxStarts["C_11"]
        tempModel.linear_constraints.set_rhs([
                (startC11 + r*lenB + b, tempModel.d1_RB_B2[r, b])
                for r in tempModel.R
                for b in tempModel.B])

        """ SOLVE THE MODEL """
        tempModel.solve()

        """ UPDATE THE RESOURCE ASSIGNMENTS IN THE SYSTEM """
        assignments = numpy.zeros([len(tempModel.R), 2])

        x_RB = [[round(tempModel.solution.get_values(
                       tempModel.decisionVarsIdxStarts["X_RB"]
                       + r*lenB + b))
                 for b in tempModel.B]
                for r in tempModel.R]

        for r in tempModel.R:
            assignments[r, 0] = x_RB[r].index(1) + 1

        y_RM = [[round(tempModel.solution.get_values(
                         tempModel.decisionVarsIdxStarts["Y_MR_Z_CM"]
                         + m*(lenR + 1) + r))
                 for m in tempModel.M]
                for r in tempModel.R]

        for r in tempModel.R:
            for m in tempModel.M:
                if y_RM[r][m] == 1:
                    assignments[r, 1] = m + 1
                    break

        """ Update the attack configurations for each patch and active fire """
        configsN = [[max([sum([tempModel.d3_BNC[b, n, c] *
                               tempModel.get_values(
                                       tempModel.decisionVarsIdxStarts["A_RB"]
                                       + r*lenB + b) / tempModel.no_CB[c, b]
                               for r in tempModel.R
                               for b in tempModel.B]) / tempModel.Q_KC[k, c]
                          for c in tempModel.C])
                     for k in range(len(tempModel.KP))]
                    for n in tempModel.N]

        for n in tempModel.N:
            damageSorted = numpy.argsort(expDamagePoten[n])
            remain = 1.0

            for k in damageSorted:
                configsN[k, n] = min(configsN[k, n], remain)
                remain = max(0, remain - configsN[k, n])

        configsM = [[math.floor(max([sum([tempModel.d2_RCM[r, c, m] *
                               tempModel.get_values(
                                       tempModel.decisionVarsIdxStarts[
                                               "Y_MR_Z_CM"] + m*(lenM + 1) + r)
                               for r in tempModel.R]) / tempModel.Q_KC[k, c]
                          for c in tempModel.C]) + 1e-6)
                     for k in range(len(tempModel.KP))]
                    for m in tempModel.M]

        fireConfigs = [configsM[m].index(1) for m in tempModel.M]

        assignmentsPath[timeStep] = assignments.astype(int)

        return [configsN, fireConfigs]

    def assignPMedian(self, assignmentsPath, expDamageExist, expDamagePoten,
                      activeFires, resourcesPath, ffdiPath, timeStep, control,
                      static, tempModel, method, dummy=0):

        """////////////////////////////////////////////////////////////////////
        /////////////////////////// DECISION VARIABLES ////////////////////////
        ////////////////////////////////////////////////////////////////////"""

        """ Set lengths """
        lenR = len(tempModel.R)
        lenB = len(tempModel.B)
        lenM = len(tempModel.M)

        """ Thresholds """
        if method == 1:
            maxB = tempModel.lambdas[control][1]
            maxF = tempModel.lambdas[control][0]
            fw = 1
            bw = 1
        elif method == 2:
            fw = tempModel.lambdas[control][0]
            bw = 1 - tempModel.lambdas[control][1]
            maxB = tempModel.lambdas[control][1]
            maxF = math.inf

        """ Set the fire details for the sets and decision variables """
        tempModel.M = [ii for ii in range(len(activeFires))]

        """ Aircraft-fire assignments """
        totalVars = tempModel.variables.get_num()
        tempModel.decisionVarsIdxStarts["Y_MR"] = totalVars
        totalVars = totalVars + len(tempModel.decisionVars["Y_MR"])

        tempModel.variables.add(
                types=([tempModel.variables.type.binary]
                       * len(tempModel.decisionVars["Y_MR"])))

        """////////////////////////////////////////////////////////////////////
        ///////////////////////////// PARAMETERS //////////////////////////////
        ////////////////////////////////////////////////////////////////////"""
        """Expected damage of existing fires if uncontrolled burn"""
        tempModel.D1_M = expDamageExist[0]

        """Expected damage of potential fires if uncontrolled burn"""
        tempModel.D2_N = expDamagePoten[0]

        """ Whether base B satisfies the maximum relocation distance for
        resource R for the designated control """
        closestBase = numpy.zeros(len(tempModel.R))

        for r in tempModel.R:
            closestDist = math.inf

            for b in tempModel.B:
                if tempModel.d1_RB[r, b] < closestDist:
                    closestBase[r] = b
                    closestDist = tempModel.d1_RB[r, b]

        tempModel.d1_RB_B2 = {
                (r, b): (1 if ((tempModel.d1_RB[r, b]) <= maxB
                               or b == closestBase[r])
                         else 0)
                for r in tempModel.R
                for b in tempModel.B}

        """ Whether fire M satisfies the maximum relocation distance for
        resources R for the designated control """
        tempModel.d2_MR_B1 = {
                (m, r):
                (1 if (tempModel.d2_RM[r, m]) <= maxF
                    else 0)
                for m in tempModel.M
                for r in tempModel.R}

        """////////////////////////////////////////////////////////////////////
        ///////////////////// OBJECTIVE VALUE COEFFICIENTS ////////////////////
        ////////////////////////////////////////////////////////////////////"""

        startYMR = tempModel.decisionVarsIdxStarts["Y_MR"]
        startXRB = tempModel.decisionVarsIdxStarts["X_RB"]

        if len(tempModel.M) > 0:
            tempModel.objective.set_linear(list(zip(
                    [startYMR + m*lenR + r
                     for m in tempModel.M
                     for r in tempModel.R],
                    [fw * tempModel.D1_M[m]*tempModel.d2_RM[r, m]
                     for m in tempModel.M
                     for r in tempModel.R])))

        tempModel.objective.set_linear(list(zip(
                [startXRB + r*lenB + b
                 for r in tempModel.R
                 for b in tempModel.B
                 for n in tempModel.N],
                [bw * tempModel.D2_N[n]*tempModel.d1_RB[r, b]
                 for r in tempModel.R
                 for b in tempModel.B
                 for n in tempModel.N])))

        """////////////////////////////////////////////////////////////////////
        //////////////////////////// CONSTRAINTS //////////////////////////////
        ////////////////////////////////////////////////////////////////////"""
        totalConstraints = tempModel.linear_constraints.get_num()

        """ Limits the fires that each aircraft is allowed to attend """
        tempModel.constraintNames["C_10"] = [
                "C_10_M" + str(m) + "_R" + str(r)
                for m in tempModel.M
                for r in tempModel.R]
        tempModel.constraintIdxStarts["C_10"] = totalConstraints
        totalConstraints = totalConstraints + len(
                tempModel.constraintNames["C_10"])

        varIdxs = {(m, r): [startYMR + m*lenR + r]
                   for m in tempModel.M
                   for r in tempModel.R}

        varCoeffs = {(m, r): [1]*len(varIdxs[m, r])
                     for m in tempModel.M
                     for r in tempModel.R}

        tempModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[m, r],
                                val=varCoeffs[m, r])
                        for m in tempModel.M
                        for r in tempModel.R],
                senses = ["L"]*len(varIdxs),
                rhs=[tempModel.d2_MR_B1[m, r]
                        for m in tempModel.M
                        for r in tempModel.R])

        """ CONSTRAINT 4 """
        startC4 = tempModel.constraintIdxStarts["C_4"]

        if len(tempModel.M) > 0:
            coefficients = [
                    (startC4 + r, startYMR + m*lenR + r, 1)
                    for r in tempModel.R
                    for m in tempModel.M]

            tempModel.linear_constraints.set_coefficients(coefficients)

        """ CONSTRAINT 6 """
        startC6 = tempModel.constraintIdxStarts["C_6"]

        if len(tempModel.M) > 0:
            coefficients = [
                    (startC6 + r,
                     startYMR + m*lenR + r,
                     1)
                    for r in tempModel.R
                    for m in tempModel.M]

            tempModel.linear_constraints.set_coefficients(coefficients)

        """ CONSTRAINT 9 """
        startC9 = tempModel.constraintIdxStarts["C_9"]
        coefficients = []
        if len(tempModel.M) > 0:
            coefficients.extend([
                    (startC9 + r,
                     startYMR + m*lenR + r,
                     tempModel.d2_RM[r, m])
                    for r in tempModel.R
                    for m in tempModel.M])

        coefficients.extend([
                (startC9 + r,
                 startXRB + r*lenB + b,
                 tempModel.d1_RB[r, b])
                for r in tempModel.R
                for b in tempModel.B])

        tempModel.linear_constraints.set_coefficients(coefficients)

        tempModel.linear_constraints.set_rhs([
                (startC9 + r,
                 tempModel.Gmax_R[r] - tempModel.G_R[r])
                for r in tempModel.R])

        """ CONSTRAINT 11 """
        startC11 = tempModel.constraintIdxStarts["C_11"]
        tempModel.linear_constraints.set_rhs([
                (startC11 + r*lenB + b, tempModel.d1_RB_B2[r, b])
                for r in tempModel.R
                for b in tempModel.B])

        """ SOLVE THE MODEL """
        tempModel.solve()

        """ UPDATE THE RESOURCE ASSIGNMENTS IN THE SYSTEM """
        assignments = numpy.zeros([len(tempModel.R), 2])

        x_RB = [[round(tempModel.solution.get_values(
                       tempModel.decisionVarsIdxStarts["X_RB"]
                       + r*lenB + b))
                 for b in tempModel.B]
                for r in tempModel.R]

        for r in tempModel.R:
            assignments[r, 0] = x_RB[r].index(1) + 1

        y_RM = [[round(tempModel.solution.get_values(
                         tempModel.decisionVarsIdxStarts["Y_MR_Z_CM"]
                         + m*(lenR + 1) + r))
                 for m in tempModel.M]
                for r in tempModel.R]

        for r in tempModel.R:
            for m in tempModel.M:
                if y_RM[r][m] == 1:
                    assignments[r, 1] = m + 1
                    break

        """ Update the attack configurations for each patch and active fire """
        configsN = [[max([sum([tempModel.d3_BNC[b, n, c] *
                               tempModel.get_values(
                                       tempModel.decisionVarsIdxStarts["A_RB"]
                                       + r*lenB + b) / tempModel.no_CB[c, b]
                               for r in tempModel.R
                               for b in tempModel.B]) / tempModel.Q_KC[k, c]
                          for c in tempModel.C])
                     for k in range(len(tempModel.KP))]
                    for n in tempModel.N]

        for n in tempModel.N:
            damageSorted = numpy.argsort(expDamagePoten[n])
            remain = 1.0

            for k in damageSorted:
                configsN[k, n] = min(configsN[k, n], remain)
                remain = max(0, remain - configsN[k, n])

        configsM = [[math.floor(max([sum([tempModel.d2_RCM[r, c, m] *
                               tempModel.get_values(
                                       tempModel.decisionVarsIdxStarts[
                                               "Y_MR_Z_CM"] + m*(lenM + 1) + r)
                               for r in tempModel.R]) / tempModel.Q_KC[k, c]
                          for c in tempModel.C]) + 1e-6)
                     for k in range(len(tempModel.KP))]
                    for m in tempModel.M]

        fireConfigs = [configsM[m].index(1) for m in tempModel.M]

        assignmentsPath[timeStep] = assignments.astype(int)

        return [configsN, fireConfigs]

    def assignAssignment1(self, assignmentsPath, expDamageExist,
                          expDamagePoten, activeFires, resourcesPath, ffdiPath,
                          timeStep, control, static, tempModel, method,
                          dummy=0):
        """ Assigns aircraft by weighting:
        1. Preference for EXISTING and POTENTIAL fires
        2. Fire DAMAGE and RELOCATION COST """

        """ First copy the relocation model """
        bases = self.model.getRegion().getStations()[0]
        patches = self.model.getRegion().getPatches()

        """////////////////////////////////////////////////////////////////////
        /////////////////////////// DECISION VARIABLES ////////////////////////
        ////////////////////////////////////////////////////////////////////"""

        """ Set lengths """
        lenR = len(tempModel.R)
        lenB = len(tempModel.B)
        lenN = len(tempModel.N)
        lenKP = len(tempModel.KP)
        lenKE = len(tempModel.KE)
        lenC = len(tempModel.C)

        """ Set the fire details for the sets and decision variables """
        tempModel.M = [ii for ii in range(len(activeFires))]

        """ Aircraft-fire assignments and fire configuration covers """
        """
        For each fire (created to date)
        1. Y_MR
        2. Delta_MK
        For fires that are extinguished during simulation, the variable's upper
        bound is set to 0. This effectively eliminates the column
        """
        component1 = [
                ["Y_MR_M" + str(m) + "_R" + str(r)
                 for r in tempModel.R]
                for m in tempModel.M]
        component2 = [
                ["Delta_MK_M" + str(m) + "_K" + str(k+1)
                 for k in tempModel.KE]
                for m in tempModel.M]

        tempModel.decisionVars["Y_MR_Delta_MK"] = []

        for m in tempModel.M:
            tempModel.decisionVars["Y_MR_Delta_MK"].extend(component1[m-1])
            tempModel.decisionVars["Y_MR_Delta_MK"].extend(component2[m-1])

        totalVars = tempModel.variables.get_num()
        tempModel.decisionVarsIdxStarts["Y_MR_Delta_MK"] = totalVars
        totalVars = totalVars + len(tempModel.decisionVars["Y_MR_Delta_MK"])

        tempModel.variables.add(
                types=([tempModel.variables.type.binary]
                       * len(tempModel.decisionVars["Y_MR_Delta_MK"])))

        """////////////////////////////////////////////////////////////////////
        ///////////////////////////// PARAMETERS //////////////////////////////
        ////////////////////////////////////////////////////////////////////"""

        """ Now set the parameters and respective coefficients """
        """ Expected damage increase of each fire under each config """
        tempModel.D1_MK = expDamageExist

        """ Expected damage increase of each patch under each config """
        tempModel.D2_NK = expDamagePoten

        """ Travel times between base B and patch N for component C"""
        tempModel.d4_BNC = {
                (b, n, c):
                (self.geoDist(patches[n].getCentroid(),
                              bases[b].getLocation()) /
                 (self.model.getResourceTypes()[0].getSpeed() if c in [1, 3]
                  else self.model.getResourceTypes()[1].getSpeed()))
                for b in tempModel.B
                for n in tempModel.N
                for c in [1, 2, 3, 4]}

        """ Expected number of fires for patch N over horizon """
        look = (self.model.getLookahead() + self.model.getTotalSteps()
                if static
                else self.model.getLookahead())

        tempModel.no_N = {
                n:
                sum([numpy.interp(ffdiPath[n, t],
                                  self.model.getRegion().getPatches()[n].
                                  getVegetation().getFFDIRange(),
                                  self.model.getRegion().getPatches()[n].
                                  getVegetation().getOccurrence()[
                                          timeStep + t]) *
                     self.model.getRegion().getPatches()[n].getArea()
                     for t in range(look)])
                for n in tempModel.N}

        """ Whether resource R satisfies component C for fire M """
        tempModel.d2_RCM = {
                (r, c, m):
                (1
                    if (((c == 1 or c == 3) and tempModel.d2_RM[r, m] <=
                            tempModel.thresholds[0])
                        or (c == 2 or c == 4 and
                            tempModel.d2_RM[r, m] > tempModel.thresholds[0] and
                            tempModel.d2_RM[r, m] <= tempModel.thresholds[1]))
                    else 0)
                for r in tempModel.R
                for c in tempModel.C
                for m in tempModel.M}

        """ Whether base B satisfies component C for patch N """
        tempModel.d3_BCN = {
                (b, c, n):
                (1
                    if (((c == 1 or c == 3) and tempModel.d3_BN[b, n] <=
                            tempModel.thresholds[0])
                        or (c == 2 or c == 4 and
                            tempModel.d3_BN[b, n] > tempModel.thresholds[0] and
                            tempModel.d3_BN[b, n] <= tempModel.thresholds[1]))
                    else 0)
                for b in tempModel.B
                for c in tempModel.C
                for n in tempModel.N}

        """ Expected number of fires visible by base B for component C """
        tempModel.no_CB = {
                (c, b):
                max(1,
                    sum([tempModel.no_N[n]
                         if (((c == 1 or c == 3)
                              and tempModel.d4_BNC[b, n, c] <=
                                  tempModel.thresholds[0])
                             or (c == 2 or c == 4
                                 and tempModel.d4_BNC[b, n, c] >
                                     tempModel.thresholds[0]
                                 and tempModel.d4_BNC[b, n, c] <=
                                     tempModel.thresholds[1]))
                         else 0
                         for n in tempModel.N]))
                for c in tempModel.C
                for b in tempModel.B}

        """////////////////////////////////////////////////////////////////////
        ///////////////////// OBJECTIVE VALUE COEFFICIENTS ////////////////////
        ////////////////////////////////////////////////////////////////////"""
        startYMR = tempModel.decisionVarsIdxStarts["Y_MR_Delta_MK"]
        startXRB = tempModel.decisionVarsIdxStarts["X_RB"]
        startDeltaNK = tempModel.decisionVarsIdxStarts["Delta_NK"]
        lambdaVals = tempModel.lambdas[control+1]

        if len(tempModel.M) > 0:
            tempModel.objective.set_linear(list(zip(
                    [startYMR + m*(lenR + lenKE) + lenR + k
                     for m in tempModel.M
                     for k in range(len(tempModel.KE))],
                    [tempModel.D1_MK[m, tempModel.KE[k]]*lambdaVals[0]
                     *lambdaVals[1]
                     for m in tempModel.M
                     for k in range(len(tempModel.KE))])))

        tempModel.objective.set_linear(list(zip(
                [startDeltaNK + n*lenKP + k
                 for n in tempModel.N
                 for k in range(len(tempModel.KP))],
                [tempModel.D2_NK[n, tempModel.KP[k]]*lambdaVals[1]
                 *(1 - lambdaVals[0])
                 for n in tempModel.N
                 for k in range(len(tempModel.KP))])))

        tempModel.objective.set_linear(list(zip(
                [startXRB + r*lenB + b
                 for r in tempModel.R
                 for b in tempModel.B],
                [(1 - lambdaVals[1])*tempModel.d1_RB[r, b]
                 for r in tempModel.R
                 for b in tempModel.B])))

        """////////////////////////////////////////////////////////////////////
        //////////////////////////// CONSTRAINTS //////////////////////////////
        ////////////////////////////////////////////////////////////////////"""

        """Makes sure that a particular aircraft configuration at fire m can
        only be met if the correct number of aircraft to satisfy each of the
        components in c in configuration k are available."""
        totalConstraints = tempModel.linear_constraints.get_num()
        tempModel.constraintNames["C_1"] = [
                "C_1_K" + str(k+1) + "_C" + str(c) + "_M" + str(m)
                for k in tempModel.KE
                for c in tempModel.C
                for m in tempModel.M]
        tempModel.constraintIdxStarts["C_1"] = totalConstraints
        totalConstraints = totalConstraints + len(
                tempModel.constraintNames["C_1"])

        varIdxs = {
                (k, c, m):
                [startYMR + m*(lenR + lenKE) + lenR + k]
                + [startYMR + m*(lenR + lenKE) + r for r in tempModel.R]
                for k in range(len(tempModel.KE))
                for c in tempModel.C
                for m in tempModel.M}

        varCoeffs = {
                (k, c, m):
                [tempModel.Q_KC[tempModel.K[k], c]]
                + [-tempModel.d2_RCM[r, c, m] for r in tempModel.R]
                for k in range(len(tempModel.KE))
                for c in tempModel.C
                for m in tempModel.M}

        tempModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[k, c, m],
                                val=varCoeffs[k, c, m])
                        for k in range(len(tempModel.KE))
                        for c in tempModel.C
                        for m in tempModel.M],
                senses=["L"]*len(varIdxs),
                rhs=[0]*len(varIdxs))

        """Ensures that a fire can be assigned only one configuration"""
        tempModel.constraintNames["C_7"] = [
                "C_7_M" + str(m)
                for m in tempModel.M]
        tempModel.constraintIdxStarts["C_7"] = totalConstraints
        totalConstraints = totalConstraints + len(
                tempModel.constraintNames["C_7"])

        varIdxs = {(m): [startYMR + m*(lenR + lenKE) + lenR + k
                         for k in range(len(tempModel.KE))]
                   for m in tempModel.M}

        varCoeffs = {(m): [1]*len(varIdxs[m])
                     for m in tempModel.M}

        tempModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[m],
                                val=varCoeffs[m])
                        for m in tempModel.M],
                senses=["E"]*len(varIdxs),
                rhs=[1]*len(varIdxs))

        """ MODIFY THE COEFFICIENTS AND COMPONENTS OF OTHER CONSTRAINTS """
        """ CONSTRAINT 2 """
        startARB = tempModel.decisionVarsIdxStarts["A_RB"]
        startC2 = tempModel.constraintIdxStarts["C_2"]
        coefficients = [
                (startC2 + k*lenC*lenN + c*lenN + n,
                 startARB + r*lenB + b,
                 -tempModel.d3_BCN[b, c, n]/tempModel.no_CB[c, b])
                for r in tempModel.R
                for b in tempModel.B
                for k in range(len(tempModel.KP))
                for c in tempModel.C
                for n in tempModel.N]

        tempModel.linear_constraints.set_coefficients(coefficients)

        """ CONSTRAINT 4 """
        startC4 = tempModel.constraintIdxStarts["C_4"]

        if len(tempModel.M) > 0:
            coefficients = [
                    (startC4 + r, startYMR + m*(lenR + lenKE) + r, 1)
                    for r in tempModel.R
                    for m in tempModel.M]

            tempModel.linear_constraints.set_coefficients(coefficients)

        """ CONSTRAINT 6 """
        startC6 = tempModel.constraintIdxStarts["C_6"]

        if len(tempModel.M) > 0:
            coefficients = [
                    (startC6 + r,
                     startYMR + m*(lenR + lenKE) + r,
                     1)
                    for r in tempModel.R
                    for m in tempModel.M]

            tempModel.linear_constraints.set_coefficients(coefficients)

        """ CONSTRAINT 9 """
        startC9 = tempModel.constraintIdxStarts["C_9"]
        coefficients = []
        if len(tempModel.M) > 0:
            coefficients.extend([
                    (startC9 + r,
                     startYMR + m*(lenR + lenKE) + r,
                     tempModel.d2_RM[r, m])
                    for r in tempModel.R
                    for m in tempModel.M])

        coefficients.extend([
                (startC9 + r,
                 startXRB + r*lenB + b,
                 tempModel.d1_RB[r, b])
                for r in tempModel.R
                for b in tempModel.B])

        tempModel.linear_constraints.set_coefficients(coefficients)

        tempModel.linear_constraints.set_rhs([
                (startC9 + r,
                 tempModel.Gmax_R[r] - tempModel.G_R[r])
                for r in tempModel.R])

        """ SOLVE THE MODEL """
        tempModel.solve()

        """ UPDATE THE RESOURCE ASSIGNMENTS IN THE SYSTEM """
        assignments = numpy.zeros([len(tempModel.R), 2])

        x_RB = [[round(tempModel.solution.get_values(
                       tempModel.decisionVarsIdxStarts["X_RB"]
                       + r*lenB + b))
                 for b in tempModel.B]
                for r in tempModel.R]

        for r in tempModel.R:
            assignments[r, 0] = x_RB[r].index(1) + 1

        y_RM = [[round(tempModel.solution.get_values(
                         tempModel.decisionVarsIdxStarts["Y_MR_Delta_MK"]
                         + m*(lenR + lenKE) + r))
                 for m in tempModel.M]
                for r in tempModel.R]

        for r in tempModel.R:
            for m in tempModel.M:
                if y_RM[r][m] == 1:
                    assignments[r, 1] = m + 1
                    break

        """ Update the attack configurations for each patch and active fire """
        configsN = [[(tempModel.solution.get_values(
                      tempModel.decisionVarsIdxStarts["Delta_NK"]
                      + n*lenKP + k))
                     for k in range(len(tempModel.KP))]
                    for n in tempModel.N]

        configsM = [[round(tempModel.solution.get_values(
                           tempModel.decisionVarsIdxStarts["Y_MR_Delta_MK"]
                           + m*(lenR + lenKE) + lenR + k))
                     for k in range(len(tempModel.KE))]
                    for m in tempModel.M]

        fireConfigs = [configsM[m].index(1) for m in tempModel.M]

        assignmentsPath[timeStep] = assignments.astype(int)

        return [configsN, fireConfigs]

    def assignAssignment2(self, assignmentsPath, expDamageExist,
                          expDamagePoten, activeFires, resourcesPath, ffdiPath,
                          timeStep, control, static, tempModel, method,
                          dummy=0, staticAfter=False):

        """ Main assignment method with thresholds or thresholds and weight """

        """ First copy the relocation model """
        bases = self.model.getRegion().getStations()[0]
        patches = self.model.getRegion().getPatches()

        """////////////////////////////////////////////////////////////////////
        /////////////////////////// DECISION VARIABLES ////////////////////////
        ////////////////////////////////////////////////////////////////////"""

        """ Set lengths """
        lenR = len(tempModel.R)
        lenB = len(tempModel.B)
        lenN = len(tempModel.N)
        lenKP = len(tempModel.KP)
        lenKE = len(tempModel.KE)
        lenC = len(tempModel.C)

        """ Thresholds """
        if dummy == 0:
            # Chosen assignments
            if method == 1:
                maxB = tempModel.lambdas[control+1][1]
                maxF = tempModel.lambdas[control+1][0]
                fw = 1
                bw = 1
            elif method == 2:
                fw = tempModel.lambdas[control+1][0]
                bw = 1 - tempModel.lambdas[control+1][0]
                maxB = tempModel.lambdas[control+1][1]
                maxF = tempModel.lambdas[control+1][1]

        elif dummy == 1:
            # Assign only to cover existing fires (state vals 3 + 4)
            if method == 1:
                fw = max([tempModel.lambdas[c][0]
                          for c in tempModel.lambdas.keys()])
                fw = max(fw, 0.95)
            else:
                fw = 0.95

            bw = 1 - fw
            maxF = math.inf
            maxB = math.inf

        elif dummy == 2:
            # Assign only to cover potential fires (state val 5)
            fw = 0
            bw = 1
            maxF = math.inf
            maxB = math.inf

        elif dummy == 3:
            # We leave the aircraft at their current base assignments and
            # determine patch covers from there. This requires fixing
            # base assignment variables to what they already are through the
            # addition of a few more constraints
            fw = 0
            bw = 1
            maxF = math.inf
            maxB = math.inf

            # Set assignments
            lenB - len(tempModel.B)
            start = tempModel.decisionVarsIdxStarts["X_RB"]

            for r in range(len(tempModel.R)):
                for b in range(len(tempModel.B)):
                    tempModel.variables.set_lower_bounds(
                        start + r*lenB + b, 0.0)
                    tempModel.variables.set_upper_bounds(
                        start + r*lenB + b, 0.0)

                    if assignmentsPath[timeStep][r][0] == b + 1:
                        tempModel.variables.set_lower_bounds(
                            start + r*lenB + b, 1.0)
                        tempModel.variables.set_upper_bounds(
                            start + r*lenB + b, 1.0)
                    else:
                        tempModel.variables.set_lower_bounds(
                            start + r*lenB + b, 0.0)
                        tempModel.variables.set_upper_bounds(
                            start + r*lenB + b, 0.0)

        """ Set the fire details for the sets and decision variables """
        tempModel.M = [ii for ii in range(len(activeFires))]

        """ Aircraft-fire assignments and fire configuration covers """
        """
        For each fire (created to date)
        1. Y_MR
        2. Delta_MK
        For fires that are extinguished during simulation, the variable's upper
        bound is set to 0. This effectively eliminates the column
        """
        component1 = [
                ["Y_MR_M" + str(m) + "_R" + str(r)
                 for r in tempModel.R]
                for m in tempModel.M]
        component2 = [
                ["Delta_MK_M" + str(m) + "_K" + str(k+1)
                 for k in tempModel.KE]
                for m in tempModel.M]

        tempModel.decisionVars["Y_MR_Delta_MK"] = []

        for m in tempModel.M:
            tempModel.decisionVars["Y_MR_Delta_MK"].extend(component1[m-1])
            tempModel.decisionVars["Y_MR_Delta_MK"].extend(component2[m-1])

        totalVars = tempModel.variables.get_num()
        tempModel.decisionVarsIdxStarts["Y_MR_Delta_MK"] = totalVars
        totalVars = totalVars + len(tempModel.decisionVars["Y_MR_Delta_MK"])

        tempModel.variables.add(
                types=([tempModel.variables.type.binary]
                       * len(tempModel.decisionVars["Y_MR_Delta_MK"])))

        """////////////////////////////////////////////////////////////////////
        ///////////////////////////// PARAMETERS //////////////////////////////
        ////////////////////////////////////////////////////////////////////"""

        """ Now set the parameters and respective coefficients """
        """ Expected damage increase of each fire under each config """
        tempModel.D1_MK = expDamageExist

        """ Expected damage increase of each patch under each config """
        tempModel.D2_NK = expDamagePoten

        """ Travel times between base B and patch N for component C"""
        tempModel.d4_BNC = {
                (b, n, c):
                (self.geoDist(patches[n].getCentroid(),
                              bases[b].getLocation()) /
                 (self.model.getResourceTypes()[0].getSpeed() if c in [1, 3]
                  else self.model.getResourceTypes()[1].getSpeed()))
                for b in tempModel.B
                for n in tempModel.N
                for c in [1, 2, 3, 4]}

        """ Expected number of fires for patch N over horizon """
        look = (self.model.getLookahead() + self.model.getTotalSteps()
                if static and timeStep == 0
                else self.model.getLookahead())

        tempModel.no_N = {
                n:
                sum([numpy.interp(ffdiPath[n, t],
                                  self.model.getRegion().getPatches()[n].
                                  getVegetation().getFFDIRange(),
                                  self.model.getRegion().getPatches()[n].
                                  getVegetation().getOccurrence()[
                                          timeStep + t]) *
                     self.model.getRegion().getPatches()[n].getArea()
                     for t in range(look)])
                for n in tempModel.N}

        """ Whether resource R satisfies component C for fire M """
        tempModel.d2_RCM = {
                (r, c, m):
                (1
                    if (((c == 1 or c == 3) and tempModel.d2_RM[r, m] <=
                            tempModel.thresholds[0])
                        or (c == 2 or c == 4 and
                            tempModel.d2_RM[r, m] > tempModel.thresholds[0] and
                            tempModel.d2_RM[r, m] <= tempModel.thresholds[1]))
                    else 0)
                for r in tempModel.R
                for c in tempModel.C
                for m in tempModel.M}

        """ Whether base B satisfies the maximum relocation distance for
        resource R for the designated control """
        closestBase = numpy.zeros(len(tempModel.R))

        for r in tempModel.R:
            closestDist = math.inf

            for b in tempModel.B:
                if tempModel.d1_RB[r, b] < closestDist:
                    closestBase[r] = b
                    closestDist = tempModel.d1_RB[r, b]

        if staticAfter:
            print("got here cunt")
            tempModel.d1_RB_B2 = {
                    (r, b): (1 if (assignmentsPath[0][r][0] == b + 1)
                             else 0)
                    for r in tempModel.R
                    for b in tempModel.B}
        else:
            tempModel.d1_RB_B2 = {
                    (r, b): (1 if ((tempModel.d1_RB[r, b]) <= maxB
                                   or b == closestBase[r])
                             else 0)
                    for r in tempModel.R
                    for b in tempModel.B}

        """ Whether fire M satisfies the maximum relocation distance for
        resources R for the designated control """
        tempModel.d2_MR_B1 = {
                (m, r): (1 if (tempModel.d2_RM[r, m]) <= maxF else 0)
                for m in tempModel.M
                for r in tempModel.R}

        """ Expected number of fires visible by base B for component C """
        tempModel.no_CB = {
                (c, b):
                max(1,
                    sum([tempModel.no_N[n]
                         if (((c == 1 or c == 3)
                              and tempModel.d4_BNC[b, n, c] <=
                                  tempModel.thresholds[0])
                             or (c == 2 or c == 4
                                 and tempModel.d4_BNC[b, n, c] >
                                     tempModel.thresholds[0]
                                 and tempModel.d4_BNC[b, n, c] <=
                                     tempModel.thresholds[1]))
                         else 0
                         for n in tempModel.N]))
                for c in tempModel.C
                for b in tempModel.B}

        """////////////////////////////////////////////////////////////////////
        ///////////////////// OBJECTIVE VALUE COEFFICIENTS ////////////////////
        ////////////////////////////////////////////////////////////////////"""
        startYMR = tempModel.decisionVarsIdxStarts["Y_MR_Delta_MK"]
        startXRB = tempModel.decisionVarsIdxStarts["X_RB"]
        startDeltaNK = tempModel.decisionVarsIdxStarts["Delta_NK"]

        if len(tempModel.M) > 0:
            tempModel.objective.set_linear(list(zip(
                    [startYMR + m*(lenR + lenKE) + lenR + k
                     for m in tempModel.M
                     for k in range(len(tempModel.KE))],
                    [tempModel.D1_MK[m, tempModel.KE[k]]*fw
                     for m in tempModel.M
                     for k in range(len(tempModel.KE))])))

        tempModel.objective.set_linear(list(zip(
                [startDeltaNK + n*lenKP + k
                 for n in tempModel.N
                 for k in range(len(tempModel.KP))],
                [tempModel.D2_NK[n, tempModel.KP[k]]*bw
                 for n in tempModel.N
                 for k in range(len(tempModel.KP))])))

        """////////////////////////////////////////////////////////////////////
        //////////////////////////// CONSTRAINTS //////////////////////////////
        ////////////////////////////////////////////////////////////////////"""

        """Makes sure that a particular aircraft configuration at fire m can
        only be met if the correct number of aircraft to satisfy each of the
        components in c in configuration k are available."""
        totalConstraints = tempModel.linear_constraints.get_num()
        tempModel.constraintNames["C_1"] = [
                "C_1_K" + str(k+1) + "_C" + str(c) + "_M" + str(m)
                for k in tempModel.KE
                for c in tempModel.C
                for m in tempModel.M]
        tempModel.constraintIdxStarts["C_1"] = totalConstraints
        totalConstraints = totalConstraints + len(
                tempModel.constraintNames["C_1"])

        varIdxs = {
                (k, c, m):
                [startYMR + m*(lenR + lenKE) + lenR + k]
                + [startYMR + m*(lenR + lenKE) + r for r in tempModel.R]
                for k in range(len(tempModel.KE))
                for c in tempModel.C
                for m in tempModel.M}

        varCoeffs = {
                (k, c, m):
                [tempModel.Q_KC[tempModel.K[k], c]]
                + [-tempModel.d2_RCM[r, c, m] for r in tempModel.R]
                for k in range(len(tempModel.KE))
                for c in tempModel.C
                for m in tempModel.M}

        tempModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[k, c, m],
                                val=varCoeffs[k, c, m])
                        for k in range(len(tempModel.KE))
                        for c in tempModel.C
                        for m in tempModel.M],
                senses=["L"]*len(varIdxs),
                rhs=[0]*len(varIdxs))

        """Ensures that a fire can be assigned only one configuration"""
        tempModel.constraintNames["C_7"] = [
                "C_7_M" + str(m)
                for m in tempModel.M]
        tempModel.constraintIdxStarts["C_7"] = totalConstraints
        totalConstraints = totalConstraints + len(
                tempModel.constraintNames["C_7"])

        varIdxs = {(m): [startYMR + m*(lenR + lenKE) + lenR + k
                         for k in range(len(tempModel.KE))]
                   for m in tempModel.M}

        varCoeffs = {(m): [1]*len(varIdxs[m])
                     for m in tempModel.M}

        tempModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[m],
                                val=varCoeffs[m])
                        for m in tempModel.M],
                senses=["E"]*len(varIdxs),
                rhs=[1]*len(varIdxs))

        """ Limits the fires that each aircraft is allowed to attend """
        tempModel.constraintNames["C_10"] = [
                "C_10_M" + str(m) + "_R" + str(r)
                for m in tempModel.M
                for r in tempModel.R]
        tempModel.constraintIdxStarts["C_10"] = totalConstraints
        totalConstraints = totalConstraints + len(
                tempModel.constraintNames["C_10"])

        varIdxs = {(m, r): [startYMR + m*(lenR + lenKE) + r]
                   for m in tempModel.M
                   for r in tempModel.R}

        varCoeffs = {(m, r): [1]*len(varIdxs[m, r])
                     for m in tempModel.M
                     for r in tempModel.R}

        tempModel.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[m, r],
                                val=varCoeffs[m, r])
                        for m in tempModel.M
                        for r in tempModel.R],
                senses = ["L"]*len(varIdxs),
                rhs=[tempModel.d2_MR_B1[m, r]
                        for m in tempModel.M
                        for r in tempModel.R])

        """ MODIFY THE COEFFICIENTS AND COMPONENTS OF OTHER CONSTRAINTS """
        """ CONSTRAINT 2 """
        startARB = tempModel.decisionVarsIdxStarts["A_RB"]
        startC2 = tempModel.constraintIdxStarts["C_2"]
        coefficients = [
                (startC2 + k*lenC*lenN + c*lenN + n,
                 startARB + r*lenB + b,
                 -tempModel.d3_BCN[b, c, n]/tempModel.no_CB[c, b])
                for r in tempModel.R
                for b in tempModel.B
                for k in range(len(tempModel.KP))
                for c in tempModel.C
                for n in tempModel.N]

        tempModel.linear_constraints.set_coefficients(coefficients)

        """ CONSTRAINT 4 """
        startC4 = tempModel.constraintIdxStarts["C_4"]

        if len(tempModel.M) > 0:
            coefficients = [
                    (startC4 + r, startYMR + m*(lenR + lenKE) + r, 1)
                    for r in tempModel.R
                    for m in tempModel.M]

            tempModel.linear_constraints.set_coefficients(coefficients)

        """ CONSTRAINT 6 """
        startC6 = tempModel.constraintIdxStarts["C_6"]

        if len(tempModel.M) > 0:
            coefficients = [
                    (startC6 + r,
                     startYMR + m*(lenR + lenKE) + r,
                     1)
                    for r in tempModel.R
                    for m in tempModel.M]

            tempModel.linear_constraints.set_coefficients(coefficients)

        """ CONSTRAINT 9 """
        startC9 = tempModel.constraintIdxStarts["C_9"]
        coefficients = []
        if len(tempModel.M) > 0:
            coefficients.extend([
                    (startC9 + r,
                     startYMR + m*(lenR + lenKE) + r,
                     tempModel.d2_RM[r, m])
                    for r in tempModel.R
                    for m in tempModel.M])

        coefficients.extend([
                (startC9 + r,
                 startXRB + r*lenB + b,
                 tempModel.d1_RB[r, b])
                for r in tempModel.R
                for b in tempModel.B])

        tempModel.linear_constraints.set_coefficients(coefficients)

        tempModel.linear_constraints.set_rhs([
                (startC9 + r,
                 tempModel.Gmax_R[r] - tempModel.G_R[r])
                for r in tempModel.R])

        """ CONSTRAINT 11 """
        startC11 = tempModel.constraintIdxStarts["C_11"]
        tempModel.linear_constraints.set_rhs([
                (startC11 + r*lenB + b, tempModel.d1_RB_B2[r, b])
                for r in tempModel.R
                for b in tempModel.B])

#        self.relocationModel.write('LP.lp')
        """ SOLVE THE MODEL """
        tempModel.solve()

        """ UPDATE THE RESOURCE ASSIGNMENTS IN THE SYSTEM """
        assignments = numpy.zeros([len(tempModel.R), 2])

        x_RB = [[round(tempModel.solution.get_values(
                       tempModel.decisionVarsIdxStarts["X_RB"]
                       + r*lenB + b))
                 for b in tempModel.B]
                for r in tempModel.R]

        for r in tempModel.R:
            assignments[r, 0] = x_RB[r].index(1) + 1

        y_RM = [[round(tempModel.solution.get_values(
                         tempModel.decisionVarsIdxStarts["Y_MR_Delta_MK"]
                         + m*(lenR + lenKE) + r))
                 for m in tempModel.M]
                for r in tempModel.R]

        for r in tempModel.R:
            for m in tempModel.M:
                if y_RM[r][m] == 1:
                    assignments[r, 1] = m + 1
                    break

        """ Update the attack configurations for each patch and active fire """
        configsN = [[(tempModel.solution.get_values(
                      tempModel.decisionVarsIdxStarts["Delta_NK"]
                      + n*lenKP + k))
                     for k in range(len(tempModel.KP))]
                    for n in tempModel.N]

        configsM = [[round(tempModel.solution.get_values(
                           tempModel.decisionVarsIdxStarts["Y_MR_Delta_MK"]
                           + m*(lenR + lenKE) + lenR + k))
                     for k in range(len(tempModel.KE))]
                    for m in tempModel.M]

        fireConfigs = [configsM[m].index(1) for m in tempModel.M]

        assignmentsPath[timeStep] = assignments.astype(int)

        return [configsN, fireConfigs]

    def simulateSinglePeriod(self, assignmentsPath, resourcesPath,
                             firesPath, activeFires, accumulatedDamage,
                             accumulatedHours, patchConfigs, fireConfigs, ffdi,
                             tt):
        """ This routine updates the state of the system given decisions """
        damage = 0
        patches = self.model.getRegion().getPatches()

        """ Fight existing fires """
        inactiveFires = []

        """ First, compute the hours flown by each aircraft this period.
        If relocating, then it is the travel time between bases.
        If fighting fires, it is the full hour."""
        for r, resource in enumerate(resourcesPath):
            baseAssignment = assignmentsPath[tt + 1][r][0] - 1
            fireAssignment = assignmentsPath[tt + 1][r][1]

            if fireAssignment > 0:
                newLoc = (activeFires[fireAssignment - 1].getLocation())
            else:
                newLoc = (self.model.getRegion().getStations()[0][
                          baseAssignment].getLocation())

            oldLoc = resource.getLocation()

            if fireAssignment == 0:
                travTime = min(
                        numpy.linalg.norm(newLoc - oldLoc)/resource.getSpeed(),
                        self.model.getStepSize())
            else:
                travTime = self.model.getStepSize()

            resource.setLocation(newLoc)
            accumulatedHours[tt+1, r] = accumulatedHours[tt, r] + travTime

        """ Next, carry over the accumulated damage from the previous
        period"""
        accumulatedDamage[tt + 1, :] = accumulatedDamage[tt, :]

        """ Fight existing fires """
        for fire in range(len(activeFires)):
            sizeOld = activeFires[fire].getSize()

            extinguished = activeFires[fire].growFire(
                    self.model,
                    ffdi[activeFires[fire].getPatchID()],
                    fireConfigs[fire] + 1,
                    random=True)

            sizeCurr = max(activeFires[fire].getSize(), sizeOld)

            if extinguished:
                # Extinguish fire and remove from list of active fires
                inactiveFires.append(fire)

            """ Update damage map for area burned for existing fires """
            accumulatedDamage[tt + 1, activeFires[fire].getPatchID()] += (
                    sizeCurr - sizeOld)
            damage += sizeCurr - sizeOld

        """ Remove fires that are now contained """
        newFires = []
        for fire in range(len(activeFires)):
            if fire not in inactiveFires:
                newFires.append(activeFires[fire])

        """ Update existing fires """
        activeFires.clear()

        for fire in newFires:
            activeFires.append(fire)

        """ New fires """
        for patch in range(len(self.model.getRegion().getPatches())):
            nfPatch = patches[patch].newFires(self.model,
                                              ffdi[patch],
                                              tt,
                                              patchConfigs[patch])

            for fire in nfPatch:
                damage += fire.getSize()
                accumulatedDamage[tt + 1, patch] += fire.getSize()

                if not(fire.getInitialSuccess()):
                    activeFires.append(fire)

        """ Return incremental damage for region """
        return damage

    def copyNonCplexComponents(self, tempModel, copyModel):
        """ This routine only copies components that are immutable """
        tempModel.R = copyModel.R
        tempModel.B = copyModel.B
        tempModel.N = copyModel.N
        tempModel.T = copyModel.T
        tempModel.K = copyModel.K
        tempModel.KE = copyModel.KE
        tempModel.KP = copyModel.KP
        tempModel.C = copyModel.C
        tempModel.M_All = copyModel.M_All
        tempModel.lambdas = copyModel.lambdas
        tempModel.Gmax_R = copyModel.Gmax_R
        tempModel.Q_KC = copyModel.Q_KC
        tempModel.decisionVars = copy.copy(copyModel.decisionVars)
        tempModel.decisionVarsIdxStarts = copy.copy(
                copyModel.decisionVarsIdxStarts)
        tempModel.constraintIdxStarts = copy.copy(
                copyModel.constraintIdxStarts)
        tempModel.constraintNames = copy.copy(copyModel.constraintNames)
        tempModel.d3_BN = copyModel.d3_BN
        tempModel.d3_BCN = copyModel.d3_BCN
        tempModel.thresholds = copyModel.thresholds

    def geoDist(self, x1d, x2d):
        x1 = [x1d[0] * math.pi/180, x1d[1] * math.pi/180]
        x2 = [x2d[0] * math.pi/180, x2d[1] * math.pi/180]
        a = (math.sin(0.5 * (x2[1] - x1[1])) ** 2
             + math.cos(x1[0]) * math.cos(x2[0])
             * math.sin(0.5 * (x2[0] - x1[0])) ** 2)
        c = math.sqrt(a)
        dist = 2 * 6371 * math.asin(c)
        return dist

    def expectedDamageExisting(self, fire, ffdiPath, configID, time, look):
        damage = 0
        fireTemp = copy.copy(fire)
        patch = fire.getPatchID()
        vegetation = self.model.getRegion().getPatches()[patch].getVegetation()
        size = fireTemp.getSize()

        for tt in range(look):
            ffdi = ffdiPath[patch, tt + time]

            success = numpy.interp(ffdi,
                                   vegetation.getFFDIRange(),
                                   vegetation.getExtendedSuccess()[configID])

            sizeOld = size
            fireTemp.growFire(self.model, ffdi, configID)
            size = fireTemp.getSize()
            growth = size - sizeOld

            damage += growth * (1 - success) ** (tt)

        return damage

    def expectedDamagePotential(self, patchID, ffdiPath, configID, time, look):
        damage = 0
        patch = self.model.getRegion().getPatches()[patchID]
        vegetation = patch.getVegetation()

        for tt in range(look):
            # Only look at the expected damage of fires started at this time
            # period to the end of the rolling horizon
            ffdi = ffdiPath[patchID, tt + time]

            occ = numpy.interp(ffdi,
                               vegetation.getFFDIRange(),
                               vegetation.getOccurrence()[time + tt + 1])

            sizeM = numpy.interp(ffdi,
                                 vegetation.getFFDIRange(),
                                 vegetation.getInitialSizeMean()[configID])

            sizeSD = numpy.interp(ffdi,
                                  vegetation.getFFDIRange(),
                                  vegetation.getInitialSizeSD()[configID])

            success = numpy.interp(ffdi,
                                   vegetation.getFFDIRange(),
                                   vegetation.getInitialSuccess()[configID])

            size = math.exp(sizeM + sizeSD ** 2 / 2)

            damage += occ * size

            for t2 in range(tt, look):
                ffdi = ffdiPath[patchID, t2 + time]
                success2 = numpy.interp(ffdi,
                                        vegetation.getFFDIRange(),
                                        vegetation.getExtendedSuccess()[
                                            configID])

                grMean = numpy.interp(
                        ffdi,
                        vegetation.getFFDIRange(),
                        vegetation.getROCA2PerHourMean()[configID])

                grSD = numpy.interp(
                        ffdi,
                        vegetation.getFFDIRange(),
                        vegetation.getROCA2PerHourSD()[configID])

                radCurr = (math.sqrt(size*10000/math.pi))
                radNew = radCurr + math.exp(grMean + grSD ** 2 / 2)
                sizeOld = size
                size = (math.pi * radNew**2)/10000
                growth = size - sizeOld

                damage += ((occ * growth * (1 - success2) ** (t2 - tt)) *
                           (1 - success))

            return damage * patch.getArea()

    """////////////////////////////////////////////////////////////////////////
    ////////////////////////// ROV Routines for later /////////////////////////
    ////////////////////////////////////////////////////////////////////////"""

    def simulateROV(self):
        res = self.model.getROVResolution()
        region = self.model.getRegion()
        totalSteps = self.model.getTotalSteps()
        lookahead = self.model.getLookahead()
        stepSize = self.model.getStepSize()
        patches = region.getPatches()
        bases = region.getStations()[0]
        vegetations = self.model.getRegion().getVegetations()
        resources = region.getResources()
        fires = region.getFires()
        configsE = self.model.getUsefulConfigurationsExisting().astype(numpy.int32)
        configsP = self.model.getUsefulConfigurationsPotential().astype(numpy.int32)
        sampleFFDIs = numpy.array([self.model.getSamplePaths()[ii]
                                   for ii in range(len(
                                       self.model.getSamplePaths()))],
                                  dtype=numpy.float32)

        discount = self.model.getDiscountFactor()
        method = self.model.getControlMethod()
        # Thresholds/parameters for each control
        lambdas = numpy.array([[
                self.model.getControls()[ii].getLambda1(),
                self.model.getControls()[ii].getLambda2()]
            for ii in range(len(self.model.getControls()))],
            dtype=numpy.float32)

        noControls = len(lambdas)

        samplePaths = (
                len(sampleFFDIs)
                if len(sampleFFDIs) > 0
                else self.model.getRuns())
        mcPaths = self.model.getROVPaths()

        """ Input data for ROV """
        noBases = len(bases)
        noPatches = len(patches)
        noResources = len(resources)
        patchVegetations = region.getVegetation().astype(numpy.int32)
        patchAreas = numpy.array([patch.getArea() for patch in patches],
                                 dtype=numpy.float32)
        patchCentroids = numpy.array([patch.getCentroid()
                                      for patch in patches],
                                     dtype=numpy.float32)
        baseLocations = numpy.array([base.getLocation()
                                     for base in bases],
                                    dtype=numpy.float32)
        resourceTypes = numpy.array([0 if type(resource).__name__ == 'Tanker'
                                        else (1 if type(resource).__name__ == 'Heli'
                                        else 2 if type(resource).__name__ == 'Land'
                                        else -1)
                                     for resource in resources],
                                    dtype=numpy.int32)
        resourceSpeeds = numpy.array([resource.getSpeed()
                                      for resource in resources],
                                     dtype=numpy.float32)
        maxHours = numpy.array([resource.getMaxDailyHours()
                                for resource in resources],
                               dtype=numpy.float32)
        configurations = numpy.array([self.model.configurations[config]
                                      for config in self.model.configurations],
                                     dtype=numpy.int32)
        ffdiRanges = numpy.array([vegetation.getFFDIRange()
                                  for vegetation in vegetations],
                                 dtype=numpy.float32)
        rocA2PHMeans = numpy.array([[vegetation.getROCA2PerHourMean()[config]
                                     for config, _ in
                                         self.model.configurations.items()]
                                    for vegetation in vegetations],
                                   dtype=numpy.float32)
        rocA2PHSDs = numpy.array([[vegetation.getROCA2PerHourSD()[config]
                                   for config, _ in
                                       self.model.configurations.items()]
                                  for vegetation in vegetations],
                                 dtype=numpy.float32)
        occurrence = numpy.array([[vegetation.getOccurrence()[time]
                                   for time in range(totalSteps + lookahead
                                                     + 1)]
                                  for vegetation in vegetations],
                                 dtype=numpy.float32)
        initSizeM = numpy.array([[vegetation.getInitialSizeMean()[config]
                                  for config, _ in
                                  self.model.configurations.items()]
                                 for vegetation in vegetations],
                                dtype=numpy.float32)
        initSizeSD = numpy.array([[vegetation.getInitialSizeSD()[config]
                                   for config, _ in
                                   self.model.configurations.items()]
                                  for vegetation in vegetations],
                                 dtype=numpy.float32)
        initSuccess = numpy.array([[vegetation.getInitialSuccess()[config]
                                    for config, _ in
                                    self.model.configurations.items()]
                                   for vegetation in vegetations],
                                  dtype=numpy.float32)
        extSuccess = numpy.array([[vegetation.getExtendedSuccess()[config]
                                   for config, _ in
                                   self.model.configurations.items()]
                                  for vegetation in vegetations],
                                 dtype=numpy.float32)
        # This is for the configurations components
        thresholds = numpy.array([self.model.fireThreshold,
                                  self.model.baseThreshold],
                                 dtype=numpy.float32)

        """ MC Path Outputs (Policy Maps)"""
        self.statePaths = [None]*samplePaths
        self.regressionsX = [None]*samplePaths
        self.regressionsY = [None]*samplePaths
        self.costs2Go = [None]*samplePaths
        self.mapStates = [None]*samplePaths
        self.mapC2G = [None]*samplePaths
        self.accumulatedDamages = [None]*samplePaths
        self.controls = [None]*samplePaths
        self.rSquared = [None]*samplePaths
        self.tStatistic = [None]*samplePaths
        self.pVal = [None]*samplePaths
        self.regModel = [None]*samplePaths

        for ii in range(samplePaths):
            randCont = numpy.random.randint(6, size=[mcPaths, totalSteps + 1])

            if res > 1:
                regressionX = numpy.zeros([totalSteps, noControls, res, 3],
                                          dtype=numpy.float32)
                regressionY = numpy.zeros([totalSteps, noControls, res, res, res],
                                          dtype=numpy.float32)
            else:
                regressionX = numpy.zeros([totalSteps, noControls, 10, 1],
                                          dtype=numpy.float32)
                regressionY = numpy.zeros([totalSteps, noControls, 1, 1, 1])

            regModels = {}
#            regModels = [[None] * noControls] * totalSteps
            rSquareds = numpy.zeros([totalSteps, noControls])
            tStatistics = numpy.zeros([totalSteps, noControls, 3])
            pVals = numpy.zeros([totalSteps, noControls, 3])
            states = numpy.zeros([mcPaths, totalSteps + 1, 10],
                                 dtype=numpy.float32)
            costs2go = numpy.zeros([mcPaths, totalSteps + 1],
                                   dtype=numpy.float32)
            mapStates = {}
            mapC2G = {}

            """ Set Up Data Stream for ROV """
            accumulatedDamages = numpy.zeros([mcPaths, totalSteps + 1,
                                              noPatches], dtype=numpy.float32)
            accumulatedHours = numpy.zeros([mcPaths, totalSteps + 1,
                                            noResources], dtype=numpy.float32)
            aircraftLocations = numpy.zeros([mcPaths, totalSteps + 1,
                                             noResources, 2],
                                             dtype=numpy.float32)
            aircraftLocations[:, 0, :] = ([[
                    aircraft.getLocation()
                    for aircraft in self.model.getRegion().getResources()]]
                    *mcPaths)
            aircraftAssignments = numpy.zeros([mcPaths, totalSteps + 1,
                                               noResources, 2],
                                               dtype=numpy.int32)
            aircraftAssignments[:, 0, :] = [[
                    [resource.getBase() + 1, 0]
                    for resource in self.model.getRegion().getResources()
                ]]*mcPaths
            initialExtinguished = numpy.zeros([mcPaths, totalSteps + 1],
                                  dtype=numpy.int32)
            fireStarts = numpy.zeros([mcPaths, totalSteps + 1],
                                  dtype=numpy.int32)
            noFires = numpy.zeros([mcPaths, totalSteps + 1],
                                  dtype=numpy.int32)
            fireSizes = numpy.zeros([mcPaths, totalSteps + 1, 500],
                                    dtype=numpy.float32)
            fireLocations = numpy.zeros([mcPaths, totalSteps + 1, 500, 2],
                                        dtype=numpy.float32)
            firePatches = numpy.zeros([mcPaths, totalSteps + 1, 500],
                                      dtype=numpy.int32)

            if len(fires) > 0:
                noFires[:, 0] = numpy.array([len(fires)]*mcPaths)
                fireSizes[:, 0, 0:len(fires)] = [[
                        fire.getSize() for fire in fires]]*mcPaths
                fireLocations[:, 0, 0:len(fires), :] = [[
                        fire.getLocation() for fire in fires]]
                firePatches[:, 0, 0:len(fires)] = [[
                        fire.getPatchID() for fire in fires]]

            tankerDists = numpy.array(
                [[self.geoDist(patchCentroids[patch], baseLocations[base])
                  / resourceSpeeds[0]
                  for base in range(noBases)]
                 for patch in range(noPatches)],
                dtype=numpy.float32)

            heliDists = numpy.array(
                [[self.geoDist(patchCentroids[patch], baseLocations[base])
                  / resourceSpeeds[1]
                  for base in range(noBases)]
                 for patch in range(noPatches)],
                dtype=numpy.float32)

            # Whether bases are within 20 minutes for each aircraft type
            tankerCoversE = numpy.array(
                [[True if tankerDists[patch, base] <= thresholds[0] else False
                  for base in range(noBases)]
                 for patch in range(noPatches)])

            tankerCoversL = numpy.array(
                    [[True if (tankerDists[patch, base] > thresholds[0]
                               and tankerDists[patch, base] <= thresholds[1])
                        else False
                      for base in range(noBases)]
                     for patch in range(noPatches)])

            heliCoversE = numpy.array(
                [[True if heliDists[patch, base] <= thresholds[0] else False
                  for base in range(noBases)]
                 for patch in range(noPatches)])

            heliCoversL = numpy.array(
                    [[True if (heliDists[patch, base] > thresholds[0]
                               and heliDists[patch, base] <= thresholds[1])
                        else False
                      for base in range(noBases)]
                     for patch in range(noPatches)])

            # Min number of aircraft needed for each component of each config
            baseConfigsMax = numpy.array([0, 0, 0, 0], dtype=numpy.int32)
            fireConfigsMax = numpy.array([0, 0, 0, 0], dtype=numpy.int32)

            # Maximum number in each component based on allowed configs
            for c in range(len(configurations)):
                if c+1 in configsP:
                    if configurations[int(c)][0] > baseConfigsMax[0]:
                        baseConfigsMax[0] += 1
                    if configurations[int(c)][1] > baseConfigsMax[1]:
                        baseConfigsMax[1] += 1
                    if configurations[int(c)][2] > baseConfigsMax[2]:
                        baseConfigsMax[2] += 1
                    if configurations[int(c)][3] > baseConfigsMax[3]:
                        baseConfigsMax[3] += 1

                if c+1 in configsE:
                    if configurations[int(c)][0] > fireConfigsMax[0]:
                        fireConfigsMax[0] += 1
                    if configurations[int(c)][1] > fireConfigsMax[1]:
                        fireConfigsMax[1] += 1
                    if configurations[int(c)][2] > fireConfigsMax[2]:
                        fireConfigsMax[2] += 1
                    if configurations[int(c)][3] > fireConfigsMax[3]:
                        fireConfigsMax[3] += 1

            # Expected fires for each patch at each time step
            expectedFires = numpy.array([[
                    numpy.interp(sampleFFDIs[ii][patch][t],
                                 ffdiRanges[int(patchVegetations[patch])],
                                 occurrence[int(patchVegetations[patch])][t]) *
                                 patchAreas[patch]
                    for patch in range(noPatches)]
                   for t in range(totalSteps + lookahead)])

            # Expected visible fires for each component for each patch at each
            # time step. As FFDI is assumed deterministic, this array is used
            # for all paths
            expFiresComp = numpy.array([[
                    numpy.matmul(expectedFires[t], tankerCoversE),
                    numpy.matmul(expectedFires[t], tankerCoversL),
                    numpy.matmul(expectedFires[t], heliCoversE),
                    numpy.matmul(expectedFires[t], heliCoversL)]
                for t in range(totalSteps + lookahead)])

            """ Call the ROV code to produce the maps """
            print("MC Paths")
#            numpy.set_printoptions(threshold=numpy.nan)
#            print(sampleFFDIs[ii])
#            print(sampleFFDIs[ii][:, 0:(totalSteps + lookahead + 1)])
#
#            sys.exit()
            t0 = time.clock()
            SimulationNumba.simulateROV(
                    mcPaths, sampleFFDIs[ii], patchVegetations, patchAreas,
                    patchCentroids, baseLocations, resourceTypes,
                    resourceSpeeds, maxHours, configurations, configsE,
                    configsP, thresholds, ffdiRanges, rocA2PHMeans, rocA2PHSDs,
                    occurrence, initSizeM, initSizeSD, initSuccess, extSuccess,
                    tankerDists, heliDists, fireConfigsMax, baseConfigsMax,
                    expFiresComp, totalSteps, lookahead, stepSize,
                    accumulatedDamages, accumulatedHours, noFires,
                    initialExtinguished, fireStarts, fireSizes, fireLocations,
                    firePatches, aircraftLocations, aircraftAssignments,
                    randCont, regressionX, regressionY, regModels, rSquareds,
                    tStatistics, pVals, states, costs2go, lambdas, method,
                    noControls, mapStates, mapC2G, discount)
            t1 = time.clock()
            print('Time:   ' + str(t1-t0))

            """ Aggregate accumulated damages """
            accumulatedDamages = accumulatedDamages.sum(2)
            self.regressionsX[ii] = regressionX
            self.regressionsY[ii] = regressionY
            self.statePaths[ii] = states
            self.accumulatedDamages[ii] = accumulatedDamages
            self.costs2Go[ii] = costs2go
            self.mapStates[ii] = mapStates
            self.mapC2G[ii] = mapC2G
            self.controls[ii] = randCont
            self.rSquared[ii] = rSquareds
            self.tStatistic[ii] = tStatistics
            self.pVal[ii] = pVals
            self.regModel[ii] = regModels
            self.writeOutROV(ii)
            self.writeOutStatistics(ii)

        """////////////////////// MODEL VALIDATION /////////////////////"""
        """ Just run MPC code but with the ROV regressions as input """
        self.simulateMPC(static=False, rov=True)

        self.writeOutSummary()

    def forwardPaths(self):
        os.system("taskset -p 0xff %d" % os.getpid())
        # We don't need to store the precipitation, wind, and temperature
        # matrices over time. We only need the resulting danger index

        paths = [None]*self.model.getROVPaths()
        for pathNo in range(self.model.getROVPaths()):
            paths[pathNo] = self.initialForwardPath()

        return paths

    def randomControls(self):
        randControls = (numpy.random.choice(
                range(len(self.model.getControls())),
                self.model.getROVPaths()*self.model.getTotalSteps())
            .reshape(self.model.getROVPaths(),
                     self.model.getTotalSteps()))

        return randControls

    def endogenousPaths(self, ep, rc):
        # We store the actual fires and their sizes

        paths = []

        [initialFires, initialAss, initialLocs, cumHours, resourceTypes] = (
            self.initialiseAssignments())

        for path in range(self.model.getROVPaths()):
            [fires, assignments, currLocs, cumHours, cumDamage] = (
                self.initialEndogenousPath(ep[path], rc[path], initialFires,
                                           initialAss, initialLocs, cumHours,
                                           resourceTypes))
            paths.append([fires, assignments, currLocs, cumHours,
                          cumDamage])

        return paths

    def initialForwardPath(self):
        region = self.model.getRegion()
        regionSize = region.getX().size
        timeSteps = self.model.getTotalSteps()
        lookahead = self.model.getLookahead()

        rain = numpy.zeros([timeSteps+1+lookahead, regionSize])
        rain[0] = region.getRain()
        precipitation = numpy.zeros([timeSteps+1+lookahead, regionSize])
        precipitation[0] = region.getHumidity()
        temperatureMin = numpy.zeros([timeSteps+1+lookahead, regionSize])
        temperatureMin[0] = region.getTemperatureMin()
        temperatureMax = numpy.zeros([timeSteps+1+lookahead, regionSize])
        temperatureMax[0] = region.getTemperatureMax()
        windNS = numpy.zeros([timeSteps+1+lookahead, regionSize])
        windNS[0] = region.getWindN()
        windEW = numpy.zeros([timeSteps+1+lookahead, regionSize])
        windEW[0] = region.getWindE()
        FFDI = numpy.zeros([timeSteps+1+lookahead, regionSize])
        FFDI[0] = region.getDangerIndex()
        windRegimes = numpy.zeros([timeSteps+1+lookahead])
        windRegimes[0] = region.getWindRegime()

        wg = region.getWeatherGenerator()

        # Simulate the path forward from time zero to the end
        for ii in range(timeSteps+lookahead):
            # Compute weather
            wg.computeWeather(rain, precipitation, temperatureMin,
                              temperatureMax, windRegimes, windNS, windEW,
                              FFDI, ii)

        return FFDI

    def initialEndogenousPath(self, ep, rc, initialFires, initialAss,
                              initialLocs, cumHours0, resourceTypes):
        timeSteps = self.model.getTotalSteps()
        lookahead = self.model.getLookahead()

        # Keep a whole map of fire in the region. We do not keep track of all
        # the randomly-generated numbers used for fire growth and success (or
        # even first starts for that matter) as it is expected that the fires
        # will have different successes and may even be different fires in the
        # first place. This will save a lot of memory.
        # The assignments list is 1 element longer than the other values. This
        # is because the first element is where the aircraft were assigned in
        # the time step leading up to the start of the simulation.
        assignments = [None]*(timeSteps+1)
        currLocs = [None]*(timeSteps+1)
        cumHours = [None]*(timeSteps+1)
        cumDamage = numpy.zeros(timeSteps+1)
        # Initial cumulative damage is zero (we don't care what happened prior
        # to our study period).
        cumDamage[0] = 0.0
        # fireSeverityMap[0] = initialMap
        assignments[0] = initialAss
        currLocs[0] = initialLocs
        cumHours[0] = cumHours0

        fires = [None]*(self.model.getTotalSteps()+1)
        fires[0] = initialFires

        for ii in range(timeSteps):
            control = rc[ii]
            locationProgram = self.model.getNestedOptMethod()

            # NESTED OPTIMISATION #############################################
            # Optimise aircraft locations given selected control and state
            assignments[ii+1] = self.optimalLocations(
                control,
                fires[ii],
                assignments[ii],
                currLocs[ii],
                cumHours[ii],
                resourceTypes,
                ep[ii:(ii + lookahead)],
                locationProgram)

            # Given the locations found for this control, update the fire
            # severities for the next time period. We use the probabilities. We
            # also compute the new damage incurred and the hours used.
            [cumHours[ii+1], fires[ii+1], currLocs[ii+1], damage] = (
                self.fireSeverity(fires[ii], assignments[ii+1], currLocs[ii],
                                  cumHours[ii], resourceTypes, ep[ii]))
            cumDamage[ii+1] = cumDamage[ii] + damage

        return [fires, assignments, currLocs, cumHours, cumDamage]

    def initialiseAssignments(self):
        # Find where all of the fires currently are
        initialFires = self.model.getRegion().getFires()

        # Create a matrix of all existing aircraft (tankers then helicopters)
        totalTankers = 0
        totalHelis = 0

        airStripIdx = 0
        aircraftPositions = []
        initialLocations = []
        resourceTypes = []
        cumHours = []
        resources = self.model.getResourceTypes()

        for airStrip in self.model.getRegion().getStations()[0]:
            currentTankers = airStrip.getAirTankers()
            currentHelis = airStrip.getHelicopters()

            totalTankers = totalTankers + len(currentTankers)
            totalHelis = totalHelis + len(currentHelis)

            for tanker in range(len(currentTankers)):
                aircraftPositions.append(airStripIdx)
                resourceTypes.append(0)
                initialLocations.append([airStrip.getLocation()[0],
                                         airStrip.getLocation()[1]])
                cumHours.append(resources[0].getMaxDailyHours())

            for heli in range(len(currentHelis)):
                aircraftPositions.append(airStripIdx)
                resourceTypes.append(1)
                initialLocations.append([airStrip.getLocation()[0],
                                         airStrip.getLocation()[1]])
                cumHours.append(resources[1].getMaxDailyHours())

            airStripIdx = airStripIdx + 1

        initialBaseAss = numpy.array(aircraftPositions)
        resourceTypes = numpy.array(resourceTypes)
        initialLocations = numpy.array(initialLocations)

        # First column of the assignment is the base assignment. The second is
        # the assignment to fires. Initially, no aircraft are assigned to fires
        # until the location program is run.
        # A fire assignment of 0 implies that the aircraft is not assigned to
        # any fire
        initialFireAss = numpy.zeros(initialBaseAss.size)
        initialAss = numpy.array([initialBaseAss, initialFireAss]).transpose()

        return [initialFires, initialAss, initialLocations, cumHours,
                resourceTypes]

    def comparator(self, ffdi, tt):
        comparators = numpy.zeros(self.region.getX().size(), 1)
        # Serial
        for ii in range(self.region.getX().size):
            # Linear interpolation. Assume ffdis evenly spaced
            veg = self.region.getVegetation()[ii]
            ffdiRange = self.region.getVegetations[veg].getFFDIRange()
            occurrenceProbs = self.region.getVegetations[veg].getOccurrence()
            ffdis = ffdiRange.size
            ffdiMinIdx = math.floor((ffdi[tt][ii] - ffdiRange[0])*(ffdis-1) /
                                    (ffdiRange[ffdis] - ffdiRange[0]))
            ffdiMaxIdx = ffdiMinIdx + 1

            if ffdiMinIdx < 0:
                ffdiMinIdx = 0
                ffdiMaxIdx = 1
            elif ffdiMaxIdx >= ffdis:
                ffdiMinIdx = ffdis - 2
                ffdiMaxIdx = ffdis - 1

            xd = ((ffdi[tt][ii] - ffdiRange[ffdiMinIdx]) /
                  (ffdiRange[ffdiMaxIdx] - ffdiRange[ffdiMinIdx]))

            comparators[ii] = (xd * occurrenceProbs[ffdiMinIdx] +
                               (1 - xd) * occurrenceProbs[ffdiMaxIdx])

        return comparators

    def expectedDamages(self, expDE, expDP, ffdi, fires, config):
        # FFDI contains the FFDI in every patch at each time step in the
        # lookahead. Here, we find the index of the FFDI bin to which each of
        # these data points belongs.
        lookahead = self.model.getLookahead()
        ffdiBins = self.model.getRegion().getFFDIRange()
        patches = self.model.getPatches()

        expDP_im = numpy.zeros([len(patches), config.shape[0]])
        expDE_lm = numpy.zeros([len(fires), config.shape[0]])

        for ii in len(patches):
            ffdiPath = [numpy.where(ffdiBins <= ffdi[tt][ii])[0][-1]
                        for tt in range(lookahead)]

            for mm in config.shape[0]:
                index = ffdiPath.append(mm)
                expDP_im[ii, mm] = expDP[index]

        for ll in len(fires):
            ffdiPath = [numpy.where(ffdiBins <=
                        ffdi[tt][tuple(fires[ll].getLocation())])[0][-1]
                        for tt in range(lookahead)]

            for mm in config.shape[0]:
                index = ffdiPath.append(mm)
                expDE_lm[ll, mm] = expDE[index]

    def writeOutResults(self, maxFFDI, minFFDI, maxDamage, minDamage, sample,
                        run):
        plot = self.model.plot
        root = ("../Experiments/Experiments/" +
                self.model.getInputFile().split(
                        "../Experiments/Experiments/")[1].split("/")[0])

        """ Output folder """
        outfolder = (root + "/Outputs/Scenario_" + str(self.id) + "/Sample_" +
                     str(sample))

        subOutfolder = outfolder + "/Run_" + str(run) + "/"

        if not os.path.exists(os.path.dirname(subOutfolder)):
            try:
                os.makedirs(os.path.dirname(subOutfolder))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        """ FFDI Data """
        outputfile = subOutfolder + "FFDI.csv"
        with open(outputfile, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            # Compute columns
            columns = int(self.model.getTotalSteps() + 1)

            writer.writerow(
                    ['FFDI_PATHS']
                    + ['']*(columns-1))

            writer.writerow(
                    ['PATHS', str(len(self.realisedFFDIs[sample][run]))]
                    + ['']*(columns-2))

            writer.writerow(
                    ['PATH_' + str(run + 1), 'HOUR']
                    + ['']*(columns-2))

            writer.writerow(
                    ['PATCH']
                    + [str(h + 1)
                       for h in range(self.model.getTotalSteps() + 1)])

            writer.writerows([
                    [str(patch + 1)]
                    + self.realisedFFDIs[sample][run][
                            0:(self.model.getTotalSteps() + 1),
                            patch].tolist()
                    for patch in range(len(
                            self.model.getRegion().getPatches()))])

        if plot:
            """ FFDI Map """
            outputGraphs = pdf.PdfPages(subOutfolder + "FFDI_Map.pdf")

            rawPatches = self.model.getRegion().getPatches()
            xMin = rawPatches[0].getVertices().bounds[0]
            yMin = rawPatches[0].getVertices().bounds[1]
            xMax = rawPatches[0].getVertices().bounds[2]
            yMax = rawPatches[0].getVertices().bounds[3]

            for patch in rawPatches:
                xMinP = patch.getVertices().bounds[0]
                yMinP = patch.getVertices().bounds[1]
                xMaxP = patch.getVertices().bounds[2]
                yMaxP = patch.getVertices().bounds[3]

                xMax = xMaxP if xMaxP > xMax else xMax
                yMax = yMaxP if yMaxP > yMax else yMax
                xMin = xMinP if xMinP < xMin else xMin
                yMin = yMinP if yMinP < yMin else yMin

            xSpan = xMax - xMin

            for tt in range(self.model.getTotalSteps() + 1):
                fig = plt.figure(figsize=(5, 5*(yMax - yMin)/(xMax - xMin)))
                ax = fig.add_subplot(111)
                ax.set_xlim(xMin, xMax)
                ax.set_ylim(yMin, yMax)
                ax.set_aspect('equal')

                cmap = clrmp.get_cmap('Oranges')

                for rp, patch in enumerate(rawPatches):
                    colorFloat = (self.realisedFFDIs[sample][run][tt, rp] -
                                  minFFDI)/maxFFDI
                    self.model.shape.loc[[patch.getShapefileIndex() - 1],
                                         'geometry'].plot(
                            ax=ax, color=cmap(colorFloat), edgecolor='black')

                basePolys = []
                # Annotate with bases
                for base in range(
                        len(self.model.getRegion().getStations()[0])):
                    polygon = mpp.Polygon(
                            [(self.model.getRegion().getStations()[0][base]
                              .getLocation()[0] - 1.05 * xSpan / 40,
                             self.model.getRegion().getStations()[0][base]
                              .getLocation()[1] - 0.95 * xSpan / 40),
                             (self.model.getRegion().getStations()[0][base]
                              .getLocation()[0] - 0.95 * xSpan / 40,
                             self.model.getRegion().getStations()[0][base]
                              .getLocation()[1] - 1.05 * xSpan / 40),
                             (self.model.getRegion().getStations()[0][base]
                              .getLocation()[0] + 1.95 * xSpan / 40,
                             self.model.getRegion().getStations()[0][base]
                              .getLocation()[1] + 0.95 * xSpan / 40),
                             (self.model.getRegion().getStations()[0][base]
                              .getLocation()[0] + 0.95 * xSpan / 40,
                             self.model.getRegion().getStations()[0][base]
                              .getLocation()[1] + 1.05 * xSpan / 40)],
                            closed=True)
                    basePolys.append(polygon)

                p = mpc.PatchCollection(basePolys)
                p.set_array(numpy.ones(len(
                        self.model.getRegion().getStations()[0])))
                ax.add_collection(p)

                for base in basePolys:
                    ax.add_patch(mpp.Polygon(base.get_xy(),
                                 closed=True,
                                 ec='b',
                                 lw=1,
                                 fill='b'))

                fig.canvas.draw()
                outputGraphs.savefig(fig)

            outputGraphs.close()

        if self.model.getAlgo() > 0:
            outputfile = subOutfolder + "Resource_States.csv"

            """ Aircraft Hours and Positions """
            with open(outputfile, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')

                columns = (self.model.getTotalSteps()+1)*5

                row = ['RESOURCE_ID', 'TYPE', 'ACCUMULATED_HOURS']

                row.extend(['']*(self.model.getTotalSteps()))
                row.extend([a + str(ii)
                            for ii in range(self.model.getTotalSteps()
                                            + 1)
                            for a in ['X_POSITION_T', 'Y_POSITION_T']])
                row.append('BASE_IDS')
                row.extend(['']*(self.model.getTotalSteps()))
                row.append('FIRE_IDS')
                row.extend(['']*(self.model.getTotalSteps()))
                writer.writerow(row)

                resources = self.model.getRegion().getResources()
                helis = self.model.getRegion().getHelicopters()
                tankers = self.model.getRegion().getAirTankers()

                for ii in range(len(resources)):
                    row = [
                        str(ii+1),
                        'Fixed' if resources[ii] in tankers else 'Heli']
                    row.extend(self.aircraftHours[sample][run][:, ii])

                    row.extend([
                            (self.model.getRegion().getStations()[0][
                             self.realisedAssignments[sample][run][tt][ii][0]
                             - 1].getLocation()[pp])
                            for tt in range(self.model.getTotalSteps() + 1)
                            for pp in [0, 1]])
                    row.extend([
                            self.realisedAssignments[sample][run][tt][ii][0]
                            for tt in range(self.model.getTotalSteps() + 1)])
                    row.extend([
                            self.realisedAssignments[sample][run][tt][ii][1]
                            for tt in range(self.model.getTotalSteps() + 1)])

                    writer.writerow(row)

            """ Accumulated Damage Per Patch """
            outputfile = subOutfolder + "Accumulated_Patch_Damage.csv"

            with open(outputfile, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')

                writer.writerow(
                        ['PATCH']
                        + ['TIME_' + str(t+1)
                           for t in range(self.model.getTotalSteps() + 1)])

                writer.writerows([
                        ['T_' + str(patch + 1)] +
                        self.finalDamageMaps[sample][run][:, patch].tolist()
                        for patch in range(len(
                                self.model.getRegion().getPatches()))])

            """ Active fires """
            outputfile = subOutfolder + "Fire_States.csv"

            with open(outputfile, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')

                for tt in range(self.model.getTotalSteps() + 1):
                    writer.writerow(
                            ['Time_Step', str(tt)]
                            + ['']*3)

                    writer.writerow(
                            ['Fires',
                             str(len(self.realisedFires[sample][run][tt]))]
                            + ['']*3)

                    writer.writerow(
                            ['Fire_ID', 'Start_Size', 'End_Size',
                             'X_Pos', 'Y_Pos'])

                    for fire in range(len(
                            self.realisedFires[sample][run][tt])):
                        writer.writerow(
                                [str(fire+1),
                                 self.realisedFires[sample][run][tt][fire]
                                 .getInitialSize(),
                                 self.realisedFires[sample][run][tt][fire]
                                 .getFinalSize(),
                                 self.realisedFires[sample][run][tt][fire]
                                 .getLocation()[0],
                                 self.realisedFires[sample][run][tt][fire]
                                 .getLocation()[1]])

                    writer.writerow(['']*5)

            if self.model.getAlgo() in [4, 5]:
                """ Monte Carlo path values for dynamic MPC """
                outputfile = subOutfolder + "DynamicMPCC2G.csv"

                with open(outputfile, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')

                    writer.writerow(
                        ['PATH']
                        + ['T' + str(tt) + '_C' + str(cc+1)
                           for tt in range(self.model.getTotalSteps())
                           for cc in range(len(self.model.getControls()))])

                    for pp in range(self.model.getMCPaths()):
                        writer.writerow(
                            [str(pp+1)]
                            + [self.dynamicC2G[sample][run][pp][tt][cc]
                               for tt in range(self.model.getTotalSteps())
                               for cc in range(len(self.model.getControls()))])

            if plot:
                """ Accumulated Damage Maps with Active Fires """
                outputGraphs = pdf.PdfPages(subOutfolder + "Damage_Map.pdf")
                for tt in range(self.model.getTotalSteps() + 1):
                    fig = plt.figure(figsize=(5, 5*(yMax - yMin) /
                                              (xMax - xMin)))
                    ax = fig.add_subplot(111)
                    ax.set_xlim(xMin, xMax)
                    ax.set_ylim(yMin, yMax)
                    ax.set_aspect('equal')

                    cmap = clrmp.get_cmap('Oranges')

                    # Damage heat map
                    for rp, patch in enumerate(rawPatches):
                        colorFloat = (self.finalDamageMaps[sample][run][tt, rp]
                                      - minDamage)/maxDamage
                        self.model.shape.loc[[patch.getShapefileIndex() - 1],
                                             'geometry'].plot(
                                              ax=ax,
                                              color=cmap(colorFloat),
                                              edgecolor='black')

                    basePolys = []
                    # Annotate with bases
                    for base in range(len(
                            self.model.getRegion().getStations()[0])):
                        polygon = mpp.Polygon(
                                [(self.model.getRegion().getStations()[0][base]
                                    .getLocation()[0] - 1.05 * xSpan / 40,
                                 self.model.getRegion().getStations()[0][base]
                                 .getLocation()[1] - 0.95 * xSpan / 40),
                                 (self.model.getRegion().getStations()[0][base]
                                 .getLocation()[0] - 0.95 * xSpan / 40,
                                 self.model.getRegion().getStations()[0][base]
                                 .getLocation()[1] - 1.05 * xSpan / 40),
                                 (self.model.getRegion().getStations()[0][base]
                                 .getLocation()[0] + 1.95 * xSpan / 40,
                                 self.model.getRegion().getStations()[0][base]
                                 .getLocation()[1] + 0.95 * xSpan / 40),
                                 (self.model.getRegion().getStations()[0][base]
                                 .getLocation()[0] + 0.95 * xSpan / 40,
                                 self.model.getRegion().getStations()[0][base]
                                 .getLocation()[1] + 1.05 * xSpan / 40)],
                                closed=True)
                        basePolys.append(polygon)

                    p = mpc.PatchCollection(basePolys)
                    p.set_array(numpy.ones(len(
                            self.model.getRegion().getStations()[0])))
                    ax.add_collection(p)

                    for base in basePolys:
                        ax.add_patch(mpp.Polygon(base.get_xy(),
                                     closed=True,
                                     ec='b',
                                     lw=1,
                                     fill='b'))

                    # Annotate with active fires
                    # (only fires that survive to end of period)
                    for fire in self.realisedFires[sample][run][tt]:
                        circle = mpp.Circle(
                                (fire.getLocation()[0], fire.getLocation()[1]),
                                math.sqrt(fire.getFinalSize()/(math.pi))*9/150,
                                edgecolor="red",
                                facecolor="red",
                                alpha=0.5)
                        ax.add_patch(circle)

                    fig.canvas.draw()

                    # Base assignment annotations
                    for b, base in enumerate(
                            self.model.getRegion().getStations()[0]):
                        tankers = sum([
                                1
                                for ii in range(len(
                                        self.model.getRegion().getResources()))
                                if (self.model.getRegion().getResources()[ii]
                                    in self.model.getRegion().getAirTankers()
                                    and self.realisedAssignments[sample][run][
                                            tt][ii, 0] == b + 1)])
                        helis = sum([
                                1
                                for ii in range(len(
                                        self.model.getRegion().getResources()))
                                if (self.model.getRegion().getResources()[ii]
                                    in self.model.getRegion().getHelicopters()
                                    and self.realisedAssignments[sample][run][
                                            tt][ii, 0] == b + 1)])

                        plt.text(base.getLocation()[0] + 1.2*xSpan/40,
                                 base.getLocation()[1],
                                 "T=" + str(tankers) + ", H=" + str(helis),
                                 color="blue")

                    # Fire assignment annotations
                    for f, fire in enumerate(
                            self.realisedFires[sample][run][tt]):
                        tankers = sum([
                                1
                                for ii in range(len(
                                        self.model.getRegion().getResources()))
                                if (self.model.getRegion().getResources()[ii]
                                    in self.model.getRegion().getAirTankers()
                                    and self.realisedAssignments[sample][run][
                                            tt][ii, 1] == f + 1)])
                        helis = sum([
                                1
                                for ii in range(len(
                                        self.model.getRegion().getResources()))
                                if (self.model.getRegion().getResources()[ii]
                                    in self.model.getRegion().getHelicopters()
                                    and self.realisedAssignments[sample][run][
                                            tt][ii, 1] == f + 1)])

                        config = ""
                        if tankers > 0:
                            config += "T=" + str(tankers)
                        if helis > 0 and tankers > 0:
                            config += ", H=" + str(helis)
                        elif helis > 0:
                            config = "H=" + str(helis)

                        if tankers > 0 or helis > 0:
                            plt.text(fire.getLocation()[0],
                                     fire.getLocation()[1],
                                     config,
                                     color="red")

                    outputGraphs.savefig(fig)

                outputGraphs.close()

    def writeOutSummary(self):
        root = ("../Experiments/Experiments/" +
                self.model.getInputFile().split(
                        "../Experiments/Experiments/")[1].split("/")[0])

        """ Output folder """
        outfolder = (root + "/Outputs/Scenario_" + str(self.id))

        if not os.path.exists(os.path.dirname(outfolder)):
            try:
                os.makedirs(os.path.dirname(outfolder))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        """ FFDI Data """
        outputfile = outfolder + "_Summary.csv"
        with open(outputfile, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            # Compute columns
            columns = int(self.model.getRuns() + 1)

            writer.writerow(
                    ['SUMMARY_DAMAGES_HECTARES']
                    + ['']*(columns-1))

            writer.writerow(
                    ['SAMPLE_PATH']
                    + ['']*(columns-1))

            writer.writerows([
                    [str(path + 1)]
                    + [self.expectedDamages[path][run]
                       for run in range(self.model.getRuns())]
                    for path in range(len(
                            self.model.getSamplePaths()))])

    def writeOutROV(self, sample):
        res = self.model.getROVResolution()
        root = ("../Experiments/Experiments/" +
                self.model.getInputFile().split(
                        "../Experiments/Experiments/")[1].split("/")[0])

        """ Output folder """
        outfolder = (root + "/Outputs/Scenario_" + str(self.id) + "/Sample_" +
                     str(sample) + "/")

        if not os.path.exists(os.path.dirname(outfolder)):
            try:
                os.makedirs(os.path.dirname(outfolder))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        """ Monte Carlo Paths (with Reduced States) """
        outputfile = outfolder + "/ROV_Paths.csv"

        with open(outputfile, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            rowLists = ([
                    ['T_' + str(tt) + '_S_' + str(ss+1) for ss in range(10)]
                    + ['T_' + str(tt) + '_C2G'] + ['T_' + str(tt) + '_AD']
                for tt in range(self.model.getTotalSteps())])

            row = (['PATH'] + [listVal[ii]
                for listVal in rowLists
                for ii in range(len(listVal))])

            writer.writerow(row)

            for ii in range(self.model.getROVPaths()):
                row = [str(ii + 1)]

                for tt in range(self.model.getTotalSteps()):
                    row.extend([
                        self.statePaths[sample][ii][tt][ss]
                        for ss in range(10)])
                    row.append(self.costs2Go[sample][ii][tt])
                    row.append(self.accumulatedDamages[sample][ii][tt])

                writer.writerow(row)

        """ ROV Regressions """
        outputfile = outfolder + "/ROV_Regressions.csv"

        if self.regressionsY[0].shape[3] >1:
            """ We used Kernel regression """
            with open(outputfile, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')

                rowLists = ([[
                        ['T' + str(tt+1) + '_C' + str(cc+1) + '_S' + str(ss+1)
                         for ss in range(3)]
                        + ['T' + str(tt) + '_C' + str(cc) + '_C2G']
                        for cc in range(len(self.model.getControls()))]
                    for tt in range(self.model.getTotalSteps())])

                row = (rowListsL2[jj]
                    for rowListsL1 in rowLists
                    for rowListsL2 in rowListsL1
                    for jj in range(len(rowListsL2)))

                writer.writerow(row)

                # Three predictors
                for r1 in range(res):
                    for r2 in range(res):
                        for r3 in range(res):
                            row = []

                            for tt in range(self.model.getTotalSteps()):
                                for cc in range(len(self.model.getControls())):
                                    row.append(self.regressionsX[
                                            sample][tt, cc, r1, 0])
                                    row.append(self.regressionsX[
                                            sample][tt, cc, r2, 1])
                                    row.append(self.regressionsX[
                                            sample][tt, cc, r3, 2])

                                    row.append(self.regressionsY[
                                            sample][tt, cc, r1, r2, r3])

                            writer.writerow(row)

        else:
            """ We used multiple global quadratic regression """
            with open(outputfile, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')

                row = ['Time+Control', '1', 'x1', 'x2', 'x3', 'x1^2', 'x1x2', 'x1x3',
                       'x2^2', 'x2x3', 'x3^2']

                for tt in range(self.model.getTotalSteps()):
                    for cc in range(len(self.model.getControls())):
                        row = ['T_' + str(tt + 1) + '_C_' + str(cc + 1)]
                        row.extend(self.regressionsX[sample][tt, cc, :, 0])

                        writer.writerow(row)

        """ Controls """
        outputfile = outfolder + "/Controls.csv"

        with open(outputfile, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            row = (['PATH']
                   + ['T_' + str(ii)
                      for ii in range(self.model.getTotalSteps())])

            writer.writerow(row)

            for ii in range(self.model.getROVPaths()):
                row = [str(ii+1)]

                for tt in range(self.model.getTotalSteps()):
                    row.extend([self.controls[sample][ii][tt]])

                writer.writerow(row)

        """ Regression Data Points, Grouped by Control """
        noControls = len(self.model.getControls())

        for c in range(noControls):
            outputfile = outfolder + "/Regression_Raw_data_" + str(c) + ".csv"

            with open(outputfile, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')

                rowLists = ([
                        ['T_' + str(tt) + '_S_' + str(ss) for ss in range(3)]
                        + ['T_' + str(tt) + '_C2G']
                    for tt in range(self.model.getTotalSteps())])

                row = ['SAMPLE']

                for listVal in rowLists:
                    row.extend(listVal)

                writer.writerow(row)

                samples = [len(self.mapStates[sample][tt, c])
                           for tt in range(self.model.getTotalSteps() - 1)]

                maxSamples = max(samples)

                for ii in range(maxSamples):
                    row = [str(ii)]

                    for tt in range(self.model.getTotalSteps() - 1):

                        if ii < samples[tt]:
                            row.extend(self.mapStates[sample][tt, c][ii])
                            row.append(self.mapC2G[sample][tt, c][ii])
                        else:
                            row.extend(['', '', '', ''])

                    writer.writerow(row)

        """ Plot the control maps at different times """
        """ Plot for three different 2D views. One page per time step """
        outputGraphs = pdf.PdfPages(outfolder + "Control_Maps.pdf")
#        cm = plt.get_cmap('gist_rainbow')
#        cNorm = colors.Normalize(vmin=0, vmax=noControls-1)
#        scalarMap = clrmp.ScalarMappable(norm=cNorm, cmap=cm)

        for tt in range(1, self.model.getTotalSteps()):
            """ First time step not used """
            fig = plt.figure()
            fig.set_size_inches(11.69, 16.53)

            """ State 1 and state 2 """
            ax1 = fig.add_subplot(311)
#            ax1.set_prop_cycle(color=[scalarMap.to_rgba(ii)
#                                      for ii in range(noControls)])

            for c in range(noControls):
                ax1.scatter(self.mapStates[sample][tt, c][:, 0],
                            self.mapStates[sample][tt, c][:, 1],
                            s=5,
                            label='c_' + str(c))

            ax1.set_title('State_1 vs State_2')
            ax1.set_xlabel('State_1')
            ax1.set_ylabel('State_2')
            ax1.legend(loc='upper left')

            """ State 1 and state 3 """
            ax2 = fig.add_subplot(312)

            for c in range(noControls):
                ax2.scatter(self.mapStates[sample][tt, c][:, 0],
                            self.mapStates[sample][tt, c][:, 2],
                            s=5,
                            label='c_' + str(c))

            ax2.set_title('State_1 vs State_3')
            ax2.set_xlabel('State_1')
            ax2.set_ylabel('State_3')
            ax2.legend(loc='upper left')

            """ State 2 and state 3 """
            ax3 = fig.add_subplot(313)

            for c in range(noControls):
                ax3.scatter(self.mapStates[sample][tt, c][:, 1],
                            self.mapStates[sample][tt, c][:, 2],
                            s=5,
                            label='c_' + str(c))

            ax3.set_title('State_2 vs State_3')
            ax3.set_xlabel('State_2')
            ax3.set_ylabel('State_3')
            ax3.legend(loc='upper left')

            fig.tight_layout()
            outputGraphs.savefig(fig)

        outputGraphs.close()

        """ Plot an projection of a regular grid onto a map for a clean
        image. We need to slice in the unobserved dimension to be able to
        project onto a 2D plane. """
        outputGraphs = pdf.PdfPages(outfolder + "Control_Maps_Clean.pdf")

        for tt in range(1, self.model.getTotalSteps() - 1):
            """ First time step not used. Neither is last as the C2G there is
            simply computed using the MIP heuristic. """
            fig, axes = plt.subplots(nrows=5, ncols=3)
            fig.set_size_inches(11.69, 16.53)

            """ State 1 and State 2. 5 Different levels of State 3 """
            maxX = max([self.mapStates[sample][tt, c][:, 0].max()
                        for c in range(noControls)])
            minX = min([self.mapStates[sample][tt, c][:, 0].min()
                        for c in range(noControls)])
            maxY = max([self.mapStates[sample][tt, c][:, 1].max()
                        for c in range(noControls)])
            minY = min([self.mapStates[sample][tt, c][:, 1].min()
                        for c in range(noControls)])
            maxZ = min([self.mapStates[sample][tt, c][:, 2].max()
                        for c in range(noControls)])
            minZ = min([self.mapStates[sample][tt, c][:, 2].min()
                        for c in range(noControls)])

            """ State 1 and State 2 """
            for l in range(1,6):
                poly_reg = PolynomialFeatures(degree = 2)

                x1_grid, x2_grid = numpy.meshgrid(
                    numpy.linspace(minX, maxX, 200),
                    numpy.linspace(minY, maxY, 200))

                predict = numpy.array([numpy.ravel(x1_grid),
                                       numpy.ravel(x2_grid),
                                       numpy.ones(40000) * (maxZ - minZ)
                                       / 6.0 * l]).transpose()

                predict_ = poly_reg.fit_transform(predict)

                yPredict = numpy.zeros([x1_grid.size, noControls])

                for c in range(noControls):
                    yPredict[:, c] = self.regModel[sample][tt, c].predict(
                            predict_)

                selectedC = yPredict.argmax(1).reshape(200, 200)

                if (maxX > minX) and (maxY > minY):
                    axes[l-1, 0].imshow(selectedC,
                                      extent=[minX, maxX, minY, maxY],
                                      aspect='auto')
                    axes[l-1, 0].invert_yaxis()

            """ State 1 and State 3 """
            for l in range(1,6):
                poly_reg = PolynomialFeatures(degree = 2)

                x1_grid, x3_grid = numpy.meshgrid(
                    numpy.linspace(minX, maxX, 200),
                    numpy.linspace(minZ, maxZ, 200))

                predict = numpy.array([numpy.ravel(x1_grid),
                                       numpy.ones(40000) * (maxY - minY)
                                       / 6.0 * l,
                                       numpy.ravel(x3_grid)]).transpose()

                predict_ = poly_reg.fit_transform(predict)

                yPredict = numpy.zeros([x1_grid.size, noControls])

                for c in range(noControls):
                    yPredict[:, c] = self.regModel[sample][tt, c].predict(
                            predict_)

                selectedC = yPredict.argmax(1).reshape(200, 200)

                if (maxX > minX) and (maxZ > minZ):
                    axes[l-1, 1].imshow(selectedC,
                                      extent=[minX, maxX, minZ, maxZ],
                                      aspect='auto')
                    axes[l-1, 1].invert_yaxis()

            """ State 2 and State 3 """
            for l in range(1,6):
                poly_reg = PolynomialFeatures(degree = 2)

                x2_grid, x3_grid = numpy.meshgrid(
                    numpy.linspace(minY, maxY, 200),
                    numpy.linspace(minZ, maxZ, 200))

                predict = numpy.array([numpy.ones(40000) * (maxX - minX)
                                       / 6.0 * l,
                                       numpy.ravel(x2_grid),
                                       numpy.ravel(x3_grid)]).transpose()

                predict_ = poly_reg.fit_transform(predict)

                yPredict = numpy.zeros([x2_grid.size, noControls])

                for c in range(noControls):
                    yPredict[:, c] = self.regModel[sample][tt, c].predict(
                            predict_)

                selectedC = yPredict.argmax(1).reshape(200, 200)

                if (maxY > minY) and (maxZ > minZ):
                    axes[l-1, 2].imshow(selectedC,
                                      extent=[minY, maxY, minZ, maxZ],
                                      aspect='auto')
                    axes[l-1, 2].invert_yaxis()

            fig.tight_layout()
            outputGraphs.savefig(fig)

        outputGraphs.close()

        """ Plot the raw data (and regressions) at different times for
        different controls """
        """ Plot for the C2G vs state for all three states. One page per time
        step. One row per control. One column per state. """
        outputGraphs = pdf.PdfPages(outfolder + "States_vs_C2G.pdf")

        for tt in range(1, self.model.getTotalSteps()):
            fig, axes = plt.subplots(nrows=noControls, ncols=3, sharey=True)
            fig.set_size_inches(11.69, 16.53)

            for c in range(noControls):
                for s in range(3):
                    axes[c, s].scatter(
                        self.mapStates[sample][tt, c][:, s],
                        self.mapC2G[sample][tt, c],
                        s=7,
                        label=('Time_' + str(tt) + '_State_' + str(s)
                               + ' vs. C2G'))

            fig.tight_layout()
            outputGraphs.savefig(fig)

        outputGraphs.close()

    def writeOutStatistics(self, sample):
        root = ("../Experiments/Experiments/" +
                self.model.getInputFile().split(
                        "../Experiments/Experiments/")[1].split("/")[0])

        """ Output folder """
        outfolder = (root + "/Outputs/Scenario_" + str(self.id) + "/Sample_" +
                     str(sample) + "/")

        if not os.path.exists(os.path.dirname(outfolder)):
            try:
                os.makedirs(os.path.dirname(outfolder))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        """ Statistics """
        outputfile = outfolder + "/Statistics.csv"

        noControls = len(self.model.getControls())

        with open(outputfile, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')

            row = (['TIME', 'CONTROL', 'Adj_R_Sq', 'F_stat', 'F_stat_p_val',
                    'AIC', 'BIC']
                   + ['Coeffs', '' ,'' ,'' ,'', '', '', '', '', '']
                   + ['Coeffs_T_Stat', '' ,'' ,'' ,'', '', '', '', '', '']
                   + ['Coeffs_P_Values', '' ,'' ,'' ,'', '', '', '', '', '']
                   + ['Coeffs_STDErr', '' ,'' ,'' ,'', '', '', '', '', '']
                   + ['Coeffs_P<2.5%', '' ,'' ,'' ,'', '', '', '', '', '']
                   + ['Coeffs_P>97.5%', '' ,'' ,'' ,'', '', '', '', '', ''])

            writer.writerow(row)
            row = (['', '', '', '', '', '', '']
                   + ['Const', 'x1', 'x2', 'x3', 'x1x1', 'x1x2', 'x1x3',
                      'x2x2', 'x2x3', 'x3x3']
                   + ['Const', 'x1', 'x2', 'x3', 'x1x1', 'x1x2', 'x1x3',
                      'x2x2', 'x2x3', 'x3x3']
                   + ['Const', 'x1', 'x2', 'x3', 'x1x1', 'x1x2', 'x1x3',
                      'x2x2', 'x2x3', 'x3x3']
                   + ['Const', 'x1', 'x2', 'x3', 'x1x1', 'x1x2', 'x1x3',
                      'x2x2', 'x2x3', 'x3x3']
                   + ['Const', 'x1', 'x2', 'x3', 'x1x1', 'x1x2', 'x1x3',
                      'x2x2', 'x2x3', 'x3x3']
                   + ['Const', 'x1', 'x2', 'x3', 'x1x1', 'x1x2', 'x1x3',
                      'x2x2', 'x2x3', 'x3x3'])

            writer.writerow(row)

            for tt in range(1, self.model.getTotalSteps() - 1):
                """ The last period does not have a C2G function. It just uses
                the MIP values """
                for cc in range(noControls):
                    row = [str(tt + 1), str(cc + 1)]

                    for cc in range(noControls):
                        row.append(self.regModel[sample][tt, cc].rsquared_adj)
                        row.append(self.regModel[sample][tt, cc].fvalue)
                        row.append(self.regModel[sample][tt, cc].f_pvalue)
                        row.append(self.regModel[sample][tt, cc].aic)
                        row.append(self.regModel[sample][tt, cc].bic)
                        row.extend(self.regModel[sample][tt, cc].params)
                        row.extend(self.regModel[sample][tt, cc].tvalues)
                        row.extend(self.regModel[sample][tt, cc].pvalues)
                        row.extend(self.regModel[sample][tt, cc].HC0_se)
                        row.extend(self.regModel[sample][tt, cc].conf_int()[
                            :10, 0])
                        row.extend(self.regModel[sample][tt, cc].conf_int()[
                            :10, 1])

                        writer.writerow(row)

    def currPos2Fire(self, currLocs, fires):
        # Computes the distance between each aircraft and each fire

        xFire = numpy.array([fires[fire].getLocation()[0]
                             for fire in range(len(fires))])

        yFire = numpy.array([fires[fire].getLocation()[1]
                             for fire in range(len(fires))])

        xAircraft = numpy.array([currLocs[aircraft][0]
                                 for aircraft in currLocs.shape[0]])

        yAircraft = numpy.array([currLocs[aircraft][1]
                                 for aircraft in currLocs.shape[0]])

        noFires = xFire.shape
        noAircraft = xAircraft.shape

        X1 = numpy.tile(xFire.transpose(), (noAircraft[0], 1))
        X2 = numpy.tile(xAircraft, (noFires[0], 1)).transpose()
        Y1 = numpy.tile(yFire.transpose(), (noAircraft[0], 1))
        Y2 = numpy.tile(yAircraft, (noFires[0], 1)).transpose()

        dists2Fire = numpy.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)

        return dists2Fire

    def aggregateWeatherData(self, ffdi, fires):

        patches = self.model.getRegion().getPatches()
        severityAgg = numpy.zeros(patches)
        lookahead = self.model.getLookahead()
        ffdiAgg = [None]*lookahead

        for t in range(lookahead):
            iterator = 0
            ffdiAgg[t] = numpy.zeros(patches)

            for patch in self.model.getRegion().getPatches():
                ffdiAgg[t][iterator] = (sum([ffdi[ii-1]
                                             for ii in patch.getIndices()]) /
                                        len(patch.getIndices()))
                iterator = iterator + 1

        for fire in fires:
            fireX = fire.getLocation()[0]
            fireY = fire.getLocation()[1]

            patchesX = numpy.array([patches[ii].getCentroid()[0]
                                    for ii in range(len(patches))])
            patchesY = numpy.array([patches[ii].getCentroid()[1]
                                    for ii in range(len(patches))])

            dists = numpy.sqrt((patchesX - fireX)**2 + (patchesY - fireY)**2)
            severityAgg[numpy.argsort(dists)[0]] = fire.getSize()

        return [ffdiAgg, severityAgg]

    def computeState(self, assignmentsPath, resourcesPath, expDamageExist,
                     expDamagePoten, activeFires, configsP, configsE,
                     expectedFFDI, aircraftHours, method, time):

        """ STATES - USED """
        """ 1. Expected potential damage (no relocation, no existing) """
        """ Or weighted cover without relocation """
        [patchConfigs, fireConfigs] = self.assignAircraft(
                assignmentsPath, expDamageExist, expDamagePoten, activeFires,
                resourcesPath, expectedFFDI, time, 1, method, False, 1)

        s1 = 0

        for patch in range(len(self.model.getRegion().getPatches())):
            for config in range(len(patchConfigs[0])):
                s1 += (patchConfigs[patch][config]
                       * expDamagePoten[patch, configsP[config]-1])

        """ 2. Expected existing damage (no suppression) """
        s2 = 0

        for fire in range(len(activeFires)):
            s2 += expDamageExist[fire, 0]

        """ 3. Expected potential  (relocate to existing, no existing) """
        """ Relocate to existing. Remainder cover potential. """
        """ Or weighted cover with relocation to existing """
        """ Or weighted cover of potential with relocation to existing """
        [patchConfigs, fireConfigs] = self.assignAircraft(
                assignmentsPath, expDamageExist, expDamagePoten, activeFires,
                resourcesPath, expectedFFDI, time, 1, method, False, 2)

        """ STATES - UNUSED/INFORMATION """
        """ 4. Expected existing (relocate to existing) (NOT USED YET) """
        s4 = 0

        for fire in range(len(activeFires)):
            s4 += expDamageExist[fire, fireConfigs[fire]]

        # Now check the expected damage from potential fires if expected fires
        # are ignored.
        [patchConfigs, fireConfigs] = self.assignAircraft(
                assignmentsPath, expDamageExist, expDamagePoten, activeFires,
                resourcesPath, expectedFFDI, time, 1, method, False, 3)

#        # Do a dummy single period simulation
#        _ = self.simulateSinglePeriod(
#                assignmentsPath, resourcesPath, firesPath,
#                activeFires, accumulatedDamage, accumulatedHours,
#                patchConfigs, fireConfigs, FFDI[time], time)

        s3 = 0

        for patch in range(len(self.model.getRegion().getPatches())):
            for config in range(len(patchConfigs[0])):
                s3 += (patchConfigs[patch][config]
                       * expDamagePoten[patch, configsP[config]-1])

#        # Aircraft hours state
#        s1 = sum([self.relocationModel.maxHours[r] - aircraftHours[time, r]
#                  for r in self.relocationModel.R])
#
#        # Short term expected damage (Potential)
#        s2 = sum([sum([
#                patchConfigs[time][0][n, k] * expDamagePoten[n, k]
#                for k in self.relocationModel.KP])
#            for n in len(self.relocationModel.N)])
#
#        # Short term expected damage (Existing)
#        s3 = sum([expDamageExist[m, fireConfigs[m]]
#                  for m in self.relocationModel.M])

        return [s1, s2, s3]

    def interpolateC2G(self, regressionX, regressionY, stateVals, control,
                       time):
        lower = numpy.zero(3, dtype=numpy.float32)
        upper = numpy.zero(3, dtype=numpy.float32)
        coeffs = numpy.zero(8, dtype=numpy.float32)
        lowerInd = numpy.zero(3, dtype=numpy.int32)

        # Indices for each state dimension value
        for dim in range(3):
            lower[dim] = regressionX[time][control][dim][0]
            upper[dim] = regressionX[time][control][dim][-1]
            lowerInd[dim] = int(len(regressionX[time][control][dim])
                                * stateVals[0] / (upper[dim] - lower[dim]))

            if lowerInd[dim] < 0:
                lowerInd[dim] = 0
            elif lowerInd[dim] >= len(regressionX[time][control][dim]):
                lowerInd[dim] = len(regressionX[time][control][dim]) - 2

        # Now that we have all the index requirements, let's interpolate
        # Uppermost dimension X value
        x0 = regressionX[time][0][lowerInd[0]]
        x1 = regressionX[time][0][lowerInd[0] + 1]

        if abs(x1 - x0) < sys.float_info.epsilon:
            xd = 0.0
        else:
            xd = (stateVals[0] - x0) / (x1 - x0)

        # Assign y values to the coefficient matrix
        coeffs[0] = (
            (1 - xd)*regressionY[time][control][lowerInd[0]][lowerInd[1]][
            lowerInd[2]]
            + xd*regressionY[time][control][lowerInd[0]+1][lowerInd[1]][
            lowerInd[2]])
        coeffs[1] = (
            (1 - xd)*regressionY[time][control][lowerInd[0]][lowerInd[1]][
            lowerInd[2]+1]
            + xd*regressionY[time][control][lowerInd[0]+1][lowerInd[1]][
            lowerInd[2]+1])
        coeffs[2] = (
            (1 - xd)*regressionY[time][control][lowerInd[0]][lowerInd[1]+1][
            lowerInd[2]]
            + xd*regressionY[time][control][lowerInd[0]+1][lowerInd[1]+1][
            lowerInd[2]])
        coeffs[3] = (
            (1 - xd)*regressionY[time][control][lowerInd[0]][lowerInd[1]][
            lowerInd[2]]
            + xd*regressionY[time][control][lowerInd[0]+1][lowerInd[1]+1][
            lowerInd[2]+1])

        # Now progress down
        x0 = regressionX[time][control][1][lowerInd[0]]
        x1 = regressionX[time][control][1][lowerInd[0] + 1]

        if abs(x1 - x0) < sys.float_info.epsilon:
            xd = 0.0
        else:
            xd = (stateVals[1] - x0) / (x1 - x0)

        coeffs[0] = (1 - xd)*coeffs[0] + xd*coeffs[2]
        coeffs[1] = (1 - xd)*coeffs[1] + xd*coeffs[3]

        # Final dimension
        x0 = regressionX[2][lowerInd[0]]
        x1 = regressionX[2][lowerInd[0] + 1]

        if abs(x1 - x0) < sys.float_info.epsilon:
            xd = 0.0
        else:
            xd = (stateVals[2] - x0) / (x1 - x0)

        return (1 - xd)*coeffs[0] + xd*coeffs[1]

    def fillEmptyFFDIData(self):
        # Get distances between all patches

        # For all 'None' entries:

        # Get average FFDI of all other patches, weighted by distance from this
        pass

    @staticmethod
    def computeFFDI(temp, rh, wind, df):
        return 2*numpy.exp(-0.45 + 0.987*numpy.log(df) -
                           0.0345*rh + 0.0338*temp + 0.0234*wind)
