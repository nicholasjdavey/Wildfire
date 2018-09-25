# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:32:32 2017

@author: davey
"""

import numpy
import cplex
import math
import copy
import os
import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as clrmp
import matplotlib.backends.backend_pdf as pdf
#import multiprocessing as mp

from Fire import Fire


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
        self.realisedFFDIs = []
        self.aircraftHours = []
        self.id = Simulation.simulations
        Simulation.simulations = Simulation.simulations + 1

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

    def getRealisedFFDIs(self):
        return self.realisedFFDIs

    def setRealisedFFDIs(self, rf):
        self.realisedFFDIs = rf

    def getAircraftHours(self):
        return self.aircraftHours

    def setAircraftHours(self, ah):
        self.aircraftHours = ah

    def simulate(self):

        switch = {
            0: self.samplePaths,
            1: self.simulateMPC,
            2: self.simulateROV,
            3: self.simulateStatic
            }

        algo = self.model.getAlgo()

        if algo > 0:
            self.buildLPModel()

        prog = switch.get(algo)
        prog()

    """////////////////////////////////////////////////////////////////////////
    /////////////////////// CPLEX Models Used in Program //////////////////////
    ////////////////////////////////////////////////////////////////////////"""

    def buildLPModel(self):
        switch = {
#            1: self.buildMaxCover,
#            2: self.buildPMedian,
            3: self.buildAssignment1
#            4: self.buildAssignment2
        }

        lpModel = self.model.getNestedOptMethod()
        prog = switch.get(lpModel)
        prog()

    def buildAssignment1(self):
        """ Builds an assignment model to reuse for computing relocations. It
        does not contain any fires. Rather, it is copied each time it is needed
        and fires are added. The copied model is solved to give the required
        assignments"""

        cplxMod = cplex.Cplex()
        cplxMod.objective.set_sense(cplxMod.objective.sense.minimize)

        """ INDEXED SETS """
        cplxMod.R = [ii for ii in
                     range(len(self.model.getRegion().getResources()))]
        cplxMod.B = [ii for ii in
                     range(len(self.model.getRegion().getStations()))]
        cplxMod.N = [ii for ii in
                     range(len(self.model.getRegion().getPatches()))]
        cplxMod.T = [ii for ii in
                     range(self.model.getTotalSteps())]
        cplxMod.K = [ii for ii in
                     range(len(self.model.getUsefulConfigurations()))]
        cplxMod.C = [1, 2, 3, 4]
        cplxMod.M = [ii for ii in
                     range(len(self.model.getRegion().getFires()))]
        cplxMod.M_All = [ii for ii in cplxMod.M]

        """ PARAMETERS """
        """ Control Parameters """
        cplxMod.lambdas = {
                ii+1:
                (self.model.getControls()[ii].getLambda1(),
                 self.model.getControls()[ii].getLambda2())
                for ii in range(len(self.model.getControls()))}

        """ Initial expected damage increase of each fire under each config """
        cplxMod.D1_MK = {
                (m, k): 0
                for m in cplxMod.M
                for k in cplxMod.K}

        """ Initial expected damage increase of each patch under each config """
        cplxMod.D2_NK = {
                (n, k): 0
                for n in cplxMod.N
                for k in cplxMod.K}

        """ Cumulative flying hours """
        cplxMod.G_R = {
                r: 0
                for r in cplxMod.R}

        """ Max flying hours """
        cplxMod.Gmax_R = {
                r: self.model.getResources()[r].getMaxDailyHours()
                for r in cplxMod.R}

        """ Travel times between aircraft and bases """
        cplxMod.d1_RB = {
                (r, b): 0
                for r in cplxMod.R
                for b in cplxMod.B}

        """ Distance between resource R and fire M """
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

        """ Whether resource R satisfies component C for patch N """
        cplxMod.d3_RCN = {
                (r, c, n): 0
                for r in cplxMod.R
                for c in cplxMod.C
                for n in cplxMod.N}

        """ Number of aircraft required for component C of configuration C """
        cplxMod.Q_KC = {
                (k, c): self.model.getConfigurations()[k-1, c-1]
                for k in cplxMod.K
                for c in cplxMod.C}

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
                       *len(cplxMod.decisionVarsIdxStarts["X_RB"])))

        """ Patch configuration covers """
        cplxMod.decisionVars["Delta_NK"] = [
                "Delta_NK_N" + str(n) + "_K" + str(k)
                for n in cplxMod.N
                for k in cplxMod.K]
        cplxMod.decisionVarsIdxStarts["Delta_NK"] = totalVars
        totalVars = totalVars + len(cplxMod.decisionVars["Delta_NK"])

        cplxMod.variables.add(
                types=([cplxMod.variables.type.binary],
                       len(cplxMod.decisionVars["Delta_NK"])))

        """ Risk-adjusted patch configruation covers """
        cplxMod.decisionVars["Delta_NK_Adj"] = [
                "Delta_NK_Adj_N" + str(n) + "_K" + str(k)
                for n in cplxMod.N
                for k in cplxMod.K]
        cplxMod.decisionVarsIdxStarts["Delta_NK_Adj"] = totalVars
        totalVars = totalVars + len(cplxMod.decisionVars["Delta_NK_Adj"])

        cplxMod.variables.add(
                lb=[0]*len(cplxMod.decisionVars["Delta_NK_Adj"]))

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
                ["Y_MRB_M" + str(m) + "_R" + str(r) + "_B" + str(b)
                 for r in cplxMod.R
                 for b in cplxMod.B]
                for m in cplxMod.M]
        component2 = [
                ["Delta_MK_M" + str(m) + "_K" + str(k)
                 for k in cplxMod.K]
                for m in cplxMod.M]

        cplxMod.decisionVars["Y_MRB_Delta_MK"] = []

        for m in cplxMod.M:
            cplxMod.decisionVars["Y_MRB_Delta_MK"].extend(component1[m-1])
            cplxMod.decisionVars["Y_MRB_Delta_MK"].extend(component2[m-1])

        cplxMod.decisionVarsIdxStarts["Y_MRB_Delta_MK"] = totalVars
        totalVars = totalVars + len(cplxMod.decisionVars["Y_MRB_Delta_MK"])

        cplxMod.variables.add(
                types=([cplxMod.variables.type.binary]
                       *len(cplxMod.decisionVarsIdxStarts["Y_MRB_Delta_MK"])))

        """ CONSTRAINTS """
        cplxMod.constraintNames = {}
        cplxMod.constraintIdxStarts = {}
        totalConstraints = 0

        """Makes sure that a particular aircraft configuration at patch n can
        only be met if the correct number of aircraft to satisfy each of the
        components in c in configuration k are available."""
        cplxMod.constraintNames["C_2"] = [
                "C_2_K" + str(k) + "_C" + str(c) + "_N" + str(n)
                for k in cplxMod.K
                for c in cplxMod.C
                for n in cplxMod.N]
        cplxMod.constraintIdxStarts["C_2"] = totalConstraints
        totalConstraints = totalConstraints + len(cplxMod.constraintNames["C_2"])

        varIdxs = {}
        varCoeffs = {}
        for k in cplxMod.K:
            for c in cplxMod.C:
                for n in cplxMod.N:
                    varIdxs[k, c, n] = (
                            [cplxMod.decisionVarsIdxStarts["X_RB"] +
                             r*len(cplxMod.B) + b
                             for r in cplxMod.R
                             for b in cplxMod.B])
                    varCoeffs[k, c, n] = (
                            [-cplxMod.d3_RNC[r, n, c]
                             for r in cplxMod.R
                             for b in cplxMod.B])

                    varIdxs[k, c, n].extend(
                            [cplxMod.decisionVarsIdxStarts["Y_MRB_Delta_MK"]
                             + m*(len(cplxMod.R)*len(cplxMod.B) +
                                  len(cplxMod.K))
                             + r*len(cplxMod.B) + b
                             for r in cplxMod.R
                             for b in cplxMod.B
                             for m in cplxMod.M])
                    varCoeffs[k, c, n].extend(
                            [cplxMod.d3_RNC[r, n, c]
                             for r in cplxMod.R
                             for b in cplxMod.B
                             for m in cplxMod.M])

                    varIdxs[k, c, n].append(
                            [cplxMod.decisionVarsIdxStarts["Delta_NK"]
                             + n*len(cplxMod.K) + k])
                    varCoeffs[k, c, n].append(cplxMod.Q_KC[k, c])

        cplxMod.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[k, c, n],
                                val=varCoeffs[k, c, n])
                        for k in cplxMod.K
                        for c in cplxMod.C
                        for n in cplxMod.N],
                senses=["L"]*(len(varIdxs)),
                rhs=[0]*len(varIdxs))

        """Probability-adjusted patch covers to account for possibility that
        multiple fires may need to be covered by the aircraft"""
        cplxMod.constraintNames["C_3"] = [
                "C_3_K" + str(k) + "_C" + str(c) + "_N" + str(n)
                for k in cplxMod.K
                for c in cplxMod.C
                for n in cplxMod.N]
        cplxMod.constraintIdxStarts["C_3"] = totalConstraints
        totalConstraints = totalConstraints + len(cplxMod.constraintNames["C_3"])

        varIdxs = {}
        varCoeffs = {}
        for k in cplxMod.K:
            for c in cplxMod.C:
                for n in cplxMod.N:
                    varIdxs[k, c, n] = (
                            [cplxMod.decisionVarsIdxStarts["X_RB"] +
                             r*len(cplxMod.B) + b
                             for r in cplxMod.R
                             for b in cplxMod.B])
                    varCoeffs[k, c, n] = (
                            [-cplxMod.d3_RNC[r, n, c]/cplxMod.no_CB[c, b]
                             for r in cplxMod.R
                             for b in cplxMod.B])

                    varIdxs[k, c, n].append(
                            [cplxMod.decisionVarsIdxStarts["Y_MRB_Delta_MK"]
                             + m*(len(cplxMod.R)*len(cplxMod.B) +
                                  len(cplxMod.K))
                             + r*len(cplxMod.B) + b
                             for r in cplxMod.R
                             for b in cplxMod.B
                             for m in cplxMod.M])
                    varCoeffs[k, c, n].extend(
                            [cplxMod.d3_RNC[r, n, c]/cplxMod.no_CB[c, b]
                             for r in cplxMod.R
                             for b in cplxMod.B
                             for m in cplxMod.M])

                    varIdxs[k, c, n].append(
                            [cplxMod.decisionVarsIdxStarts["Delta_NK_Adj"]
                             + n*len(cplxMod.K) + k])
                    varCoeffs[k, c, n].append(cplxMod.Q_KC[k, c])

        cplxMod.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[k, c, n],
                                val=varCoeffs[k, c, n])
                        for k in cplxMod.K
                        for c in cplxMod.C
                        for n in cplxMod.N],
                senses=["L"]*(len(varIdxs)),
                rhs=[0]*len(varIdxs))

        """Ensures that a configuration is not applied to a patch unless it is
        indeed possible"""
        cplxMod.constraintNames["C_4"] = [
                "C_4_K" + str(k) + "_N" + str(n)
                for k in cplxMod.K
                for n in cplxMod.N]
        cplxMod.constraintIdxStarts["C_4"] = totalConstraints
        totalConstraints = totalConstraints + len(cplxMod.constraintNames["C_4"])

        varIdxs = {}
        varCoeffs = {}
        for k in cplxMod.K:
            for n in cplxMod.N:
                varIdxs[k, n] = (
                        [cplxMod.decisionVarsIdxStarts["Delta_NK"]
                         + n*len(cplxMod.K) + k])
                varCoeffs[k, n] = ([1])

                varIdxs[k, n].extend(
                        [cplxMod.decisionVarsIdxStarts["Delta_NK_Adj"]
                         + n*len(cplxMod.K) + k])
                varCoeffs[k, n].extend([-1])

        cplxMod.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[k, n],
                                val=varCoeffs[k, n])
                        for k in cplxMod.K
                        for n in cplxMod.N],
                senses=["L"]*len(varIdxs),
                rhs=[0]*len(varIdxs))

        """Ensures that an aircraft is assigned to only one base"""
        cplxMod.constraintNames["C_6"] = [
                "C_6_R" + str(r)
                for r in cplxMod.R]
        cplxMod.constraintIdxStarts["C_6"] = totalConstraints
        totalConstraints = totalConstraints + len(cplxMod.constraintNames["C_6"])

        varIdxs = {}
        varCoeffs = {}
        for r in cplxMod.R:
            varIdxs[r] = (
                    [cplxMod.decisionVarsIdxStarts["X_RB"]
                    + b
                    for b in cplxMod.B])
            varCoeffs[r] = [1]*len(varIdxs[r])

        cplxMod.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[r],
                                val=varCoeffs[r])
                        for r in cplxMod.R],
                senses=["E"]*len(varIdxs),
                rhs=[1]*len(varIdxs))

        """Ensures that an aircraft can be allocated to at most one fire"""
        cplxMod.constraintNames["C_7"] = [
                "C_7_R" + str(r) + "_B" + str(b)
                for r in cplxMod.R
                for b in cplxMod.B]
        cplxMod.constraintIdxStarts["C_7"] = totalConstraints
        totalConstraints = totalConstraints + len(cplxMod.constraintNames["C_7"])

        cplxMod.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=[],
                                val=[])
                        for r in cplxMod.R
                        for b in cplxMod.B],
                senses=["L"]*len(cplxMod.R)*len(cplxMod.B),
                rhs=[1]*len(cplxMod.R)*len(cplxMod.B))

        """Ensures that a patch can be assigned only one configuration"""
        cplxMod.constraintNames["C_9"] = [
                "C_9_N" + str(n)
                for n in cplxMod.N]
        cplxMod.constraintIdxStarts["C_9"] = totalConstraints
        totalConstraints = totalConstraints + len(cplxMod.constraintNames["C_9"])

        varIdxs = {}
        varCoeffs = {}
        for n in cplxMod.N:
            varIdxs[n] = (
                    [cplxMod.decisionVarsIdxStarts["Delta_NK"]
                    + k
                    for k in cplxMod.K])
            varCoeffs[n] = [1]*len(varIdxs[n])

        cplxMod.linear_constraints.add(
                lin_expr=[
                        cplex.SparsePair(
                                ind=varIdxs[n],
                                val=varCoeffs[n])
                        for n in cplxMod.N],
                senses=["E"]*len(varIdxs),
                rhs=[1]*len(varIdxs))

        """Ensures that the maximum number of flying hours are not exceeded"""
        cplxMod.constraintNames["C_10"] = [
                "C_10_K" + str(k)
                for k in cplxMod.K]
        cplxMod.constraintIdxStarts["C_10"] = totalConstraints
        totalConstraints = totalConstraints + len(cplxMod.constraintNames["C_10"])

        varIdxs = {}
        varCoeffs = {}
        for r in cplxMod.R:
            varIdxs[r] = (
                    [cplxMod.decisionVarsIdxStarts["Y_MRB_Delta_MK"]
                     + m*(len(cplxMod.R)*len(cplxMod.B) + len(cplxMod.K))
                     + r*len(cplxMod.B) + b
                     for b in cplxMod.B
                     for m in cplxMod.M])
            varCoeffs[r] = (
                    [cplxMod.d2_RM[r, m]
                     for b in cplxMod.B
                     for m in cplxMod.M])

            varIdxs[r].extend(
                    [cplxMod.decisionVarsIdxStarts["X_RB"]
                     + r*len(cplxMod.B) + b
                     for b in cplxMod.B])
            varCoeffs[r].extend(
                    [cplxMod.d1_RB[r, b]
                     + r*len(cplxMod.B) + b
                     for b in cplxMod.B])

        cplxMod.linear_constraints.add(
                lin_expr=[
                        cplxMod.SparsePair(
                                ind=varIdxs[r],
                                val=varCoeffs[r])
                        for r in cplxMod.R],
                senses=["L"]*len(cplxMod.R),
                rhs=[cplxMod.Gmax_R[r] - cplxMod.G_R[r]
                        for r in cplxMod.R])

        """ Save the relocation model to the instance """
        self.relocationModel = cplxMod

    def samplePaths(self):
        region = self.model.getRegion()
        regionSize = region.getX().size
        timeSteps = self.model.getTotalSteps()
        lookahead = self.model.getLookahead()
        noPatches = len(self.model.getX())

        simPaths = numpy.empty([self.model.getMCPaths(),
                                self.model.getTotalSteps()
                                + self.model.getLookahead(),
                                noPatches])

        # Generate a sample FFDI path for the region N times
        for path in self.model.getMCPaths():
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

            simPaths[ii] = FFDI

        root = ("../Experiments/Experiments/" +
                self.inputfile.split("../Experiments/Experiments/")[1]
                .split("/")[0])

        # Now write out the results to a csv file for later use
        with open(root + "Sample_Paths.csv") as sp:
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

    def simulateMPC(self):
        # Computes a Model Predictive Control approach for reallocating
        # aerial resources for fighting fires. The input conditional
        # probabilities are provided as inputs to the program. We just run the
        # simulations to determine the expected end area of the fire for the
        # given input conditions and what the variance around this is.
        region = self.model.getRegion()
        timeSteps = self.model.getTotalSteps()
        patches = len(region.getPatches())
        resources = region.getResources()
        fires = region.getFires()
        configs = self.model.getConfigurations()
        sampleFFDIs = self.model.getSamplePaths()

        """ Initial assignment of aircraft to bases (Col 1) and fires (Col 2)
        A value of zero indicates no assignment (only applicable for fires) """
        assignments = numpy.empty([len(resources) , 2])

        regionSize = region.getX().size
        samplePaths = self.model.getROVPaths()
        samplePaths2 = self.model.getMCPaths()
        lookahead = self.model.getLookahead()

        self.finalDamageMaps = [None]*samplePaths
        self.expectedDamages = [None]*samplePaths
        self.realisedAssignments = [None]*samplePaths
        self.realisedFires = []
        self.realisedFFDIs = [None]*samplePaths
        self.aircraftHours = [None]*samplePaths
        self.expectedConfigDamages = [None, None]*samplePaths

        wg = region.getWeatherGenerator()

        for ii in range(samplePaths):
            damage = 0
            assignmentsPath = [None]*(timeSteps + 1)
            assignmentsPath[0] = copy.copy(assignments)
            firesPath = copy.copy(fires)
            resourcesPath = copy.copy(resources)
            activeFires = [fire for fire in firesPath]
            self.finalDamageMaps[ii] = numpy.empty([timeSteps + 1, patches])

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
            accumulatedDamage = numpy.zeros([timeSteps+1+lookahead, patches])

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
                    windRegimesTemp = numpy.zeros([timeSteps + lookahead + 1])
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
                    expectedFFDI = sampleFFDIs[ii][:, tt:(tt + lookahead + 1)]

                expDamageExist = {
                        (jj, kk):
                        self.expectedDamageExisting(
                                activeFires[jj], expectedFFDI, configs[kk], tt)
                        for jj in range(len(activeFires))
                        for kk in range(len(configs))}

                expDamagePoten = {
                        (jj, kk):
                        self.expectedDamagePotential(
                                jj, expectedFFDI, configs[kk], tt)
                        for jj in range(len(patches))
                        for kk in range(len(configs))}

                # Assign aircraft using LP
                [assignmentsPath, patchConfigs, fireConfigs] = (
                        self.assignAircraft(
                                assignmentsPath, expDamageExist,
                                expDamagePoten, activeFires, resourcesPath,
                                expectedFFDI))

                # Simulate the fire growth, firefighting success and the new
                # positions of each resources
                damage = damage + self.simulateSinglePeriod(
                        assignmentsPath, resourcesPath, firesPath, activeFires,
                        accumulatedDamage, patchConfigs, fireConfigs, FFDI[tt],
                        tt)

                # Simulate the realised weather for the next time step
                wg.computeWeather(
                        rain, precipitation, temperatureMin, temperatureMax,
                        windRegimes, windNS, windEW, FFDI, tt)

            # Store the expected damage for output results
            self.expectedDamages[ii] = [
                    FFDI, assignmentsPath, accumulatedDamage, damage]

        # Print the results to an output file
        self.writeOutResults()

    def assignAircraft(self, assignmentsPath, expDamageExist, expDamagePoten,
                       activeFires, resourcesPath, ffdiPath):

        """ First copy the relocation model """
        tempModel = cplex.Cplex(self.relocationModel)
        self.copyNonCplexComponents(tempModel, self.relocationModel)
        bases = self.model.getRegion().getStations()
        patches = self.model.getRegion().getPatches()

        """////////////////////////////////////////////////////////////////////
        /////////////////////////// DECISION VARIABLES ////////////////////////
        ////////////////////////////////////////////////////////////////////"""

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
                ["Y_MRB_M" + str(m) + "_R" + str(r) + "_B" + str(b)
                 for r in tempModel.R
                 for b in tempModel.B]
                for m in tempModel.M]
        component2 = [
                ["Delta_MK_M" + str(m) + "_K" + str(k)
                 for k in tempModel.K]
                for m in tempModel.M]

        tempModel.decisionVars["Y_MRB_Delta_MK"] = []

        for m in tempModel.M:
            tempModel.decisionVars["Y_MRB_Delta_MK"].extend(component1[m-1])
            tempModel.decisionVars["Y_MRB_Delta_MK"].extend(component2[m-1])

        totalVars = tempModel.linear_constraints.get_num()
        tempModel.decisionVarsIdxStarts["Y_MRB_Delta_MK"] = totalVars
        totalVars = totalVars + len(tempModel.decisionVars["Y_MRB_Delta_MK"])

        tempModel.variables.add(
                types=([tempModel.variables.type.binary]
                       *len(tempModel.decisionVarsIdxStarts["Y_MRB_Delta_MK"])))

        """////////////////////////////////////////////////////////////////////
        ///////////////////////////// PARAMETERS  /////////////////////////////
        ////////////////////////////////////////////////////////////////////"""

        """ Now set the parameters and respective coefficients """
        """ Expected damage increase of each fire under each config """
        tempModel.D1_MK = expDamageExist

        """ Expected damage increase of each patch under each config """
        tempModel.D2_NK = expDamagePoten

        """ Cumulative flying hours """
        tempModel.G_R = {
                r: resourcesPath[r].getFlyingHours()
                for r in tempModel.R}

        """ Travel times between aircraft and bases """
        tempModel.d1_RB = {
                (r, b):
                (math.sqrt(
                        (bases[b].getLocation()[0]
                         - resourcesPath[r].getLocation()[0])**2
                        + (bases[b].getLocation()[1]
                           - resourcesPath[r].getLocation()[1])**2)/
                        resourcesPath[r].getSpeed())
                for r in tempModel.R
                for b in tempModel.B}

        """ Travel times between resource R and fire M """
        tempModel.d2_RM = {
                (r, m):
                math.sqrt(
                        (activeFires[m].getLocation()[0]
                         - resourcesPath[r].getLocation()[0])**2
                        + (activeFires[m].getLocation()[1]
                           - resourcesPath[r].getLocation()[1])**2)
                for r in tempModel.R
                for m in tempModel.M}

        """ Travel times between resource R and patch N"""
        tempModel.d3_RN = {
                (r, n):
                math.sqrt(
                        (patches[n].getLocation()[0]
                         - resourcesPath[r].getLocation()[0])**2
                        + (patches[n].getLocation()[1]
                           - resourcesPath[r].getLocation()[1])**2)
                for r in tempModel.R
                for n in tempModel.N}

        """ Travel times between base B and patch N"""
        tempModel.d4_BNC = {
                (b, n, c):
                (math.sqrt(
                        (patches[n].getLocation()[0]
                         - bases[b].getLocation()[0])**2
                        + (patches[n].getLocation()[1]
                           - bases[b].getLocation()[1])**2)/
                 (self.model.getResourceTypes()[0].getSpeed() if c in [1, 3]
                  else self.model.getResourceTypes()[1].getSpeed()))
                for b in tempModel.B
                for n in tempModel.N
                for c in [1, 2, 3, 4]}

        """ Expected number of fires for patch N over horizon """
        tempModel.no_N = {
                n:
                sum([numpy.interp(ffdiPath[t, n],
                                  self.model.getRegion().getPatches()[n].
                                  getVegetations()[0].getFFDIRange(),
                                  self.model.getRegion().getPatches()[n].
                                  getVegetations()[0].getOccurrence())
                     for t in self.tempModel.getLookahead()])
                for n in tempModel.N}

        """ Whether resource R satisfies component C for fire M """
        tempModel.d2_RCM = {
                (r, c, m):
                (1
                    if (((c == 1 or c == 3) and tempModel.d2_RM[r, m] <= 1/3)
                        or (c == 2 or c == 4 and tempModel.d2_RM[r, m] > 1/3))
                    else 0)
                for r in tempModel.R
                for c in tempModel.C
                for m in tempModel.M}

        """ Whether resource R satisfies component C for patch N """
        tempModel.d3_RCN = {
                (r, c, n):
                (1
                    if (((c == 1 or c == 3) and tempModel.d3_RNC[r, m] <= 1/3)
                        or (c == 2 or c == 4 and tempModel.d3_RNC[r, m] > 1/3))
                    else 0)
                for r in tempModel.R
                for c in tempModel.C
                for n in tempModel.N}

        """ Expected number of fires visible by base B for component C """
        tempModel.no_CB = {
                (c, b):
                sum([tempModel.d4_BNC[b, n, c]
                     if (((c == 1 or c == 3) and tempModel.d4_BN[b, n, c] <= 1/3)
                         or (c == 2 or c == 4 and tempModel.d4_BN[b, n, c] > 1/3))
                     else 0
                     for n in tempModel.N])
                for c in tempModel.C
                for b in tempModel.B}

        """////////////////////////////////////////////////////////////////////
        //////////////////////////// CONSTRAINTS //////////////////////////////
        ////////////////////////////////////////////////////////////////////"""

        """Makes sure that a particular aircraft configuration at fire m can
        only be met if the correct number of aircraft to satisfy each of the
        components in c in configuration k are available."""
        totalConstraints = tempModel.linear_constraints.get_num()
        tempModel.constraintNames["C_1"] = [
                "C_1_K" + str(k) + "_C" + str(c) + "_M" + str(m)
                for k in tempModel.K
                for c in tempModel.C
                for m in tempModel.M]
        tempModel.constraintIdxStarts["C_1"] = totalConstraints
        totalConstraints = totalConstraints + len(tempModel.constraintNames["C_1"])

        varIdxs = {}
        varCoeffs = {}
        for k in tempModel.K:
            for c in tempModel.C:
                for m in tempModel.M:
                    varIdxs[k, c, m] = [
                            tempModel.decisionVarsIdxsStarts["Y_MRB_Delta_MK"]
                            + m*(len(tempModel.R)*len(tempModel.B) +
                                    len(tempModel.K))
                            + len(tempModel.R)*len(tempModel.B) + k]
                    varCoeffs[k, c, m] = [tempModel.Q_KC[k, c]]

                    varIdxs[k, c, m].extend([
                            tempModel.decisionVarsIdxStarts["Y_MRB_Delta_MK"]
                            + m*(len(tempModel.R)*len(tempModel.B) +
                                    len(tempModel.K))
                            + r*len(tempModel.B) + b
                            for r in tempModel.R
                            for b in tempModel.B])
                    varCoeffs[k, c, m].extend([
                            -tempModel.d2_RCM[r, c, m]
                            for r in tempModel.R
                            for b in tempModel.B])

        tempModel.linear_constraints.add(
                lin_expr=[
                        tempModel.SparsePair(
                                ind=varIdxs[k, c, m],
                                val=varCoeffs[k, c, m])
                        for k in tempModel.K
                        for c in tempModel.C
                        for m in tempModel.M],
                senses=["L"]*len(varIdxs),
                rhs=[0]*len(varIdxs))

        """Ensures that aircraft r can only fight a fire out of base b if it is
        actually stationed at base b"""
        tempModel.constraintNames["C_5"] = [
                "C_5_M" + str(m) + "_R" + str(r) + "_B" + str(b)
                for m in tempModel.M
                for r in tempModel.R
                for b in tempModel.B]
        tempModel.constraintIdxStarts["C_5"] = totalConstraints
        totalConstraints = totalConstraints + len(tempModel.constraintNames["C_5"])

        varIdxs = {}
        varCoeffs = {}
        for m in tempModel.M:
            for r in tempModel.R:
                for b in tempModel.B:
                    varIdxs[m, r, b] = [
                            tempModel.decisionVarsIdxStarts["Y_MRB_Delta_MK"]
                            + m*(len(tempModel.R)*len(tempModel.B) +
                                    len(tempModel.K))
                            + r*(len(tempModel.B)) + b]
                    varCoeffs[m, r, b] = [1]

                    varIdxs[m, r, b].extend(
                            [tempModel.decisionVarsIdxStarts["X_RB"]
                            + r*len(tempModel.B) + b])
                    varCoeffs[m, r, b].extend([1])

        tempModel.linear_constraints.add(
                lin_expr=[
                        tempModel.SparsePair(
                                ind=varIdxs[m, r, b],
                                val=varCoeffs[m, r, b])
                        for m in tempModel.M
                        for r in tempModel.R
                        for b in tempModel.B],
                senses=["L"]*len(varIdxs),
                rhs=[0]*len(varIdxs))

        """Ensures that a fire can be assigned only one configuration"""
        tempModel.constraintNames["C_8"] = [
                "C_8_M" + str(m)
                for m in tempModel.M]
        tempModel.constraintIdxStarts["C_8"] = totalConstraints
        totalConstraints = totalConstraints + len(tempModel.constraintNames["C_8"])

        varIdxs = {}
        varCoeffs = {}
        for m in tempModel.M:
            varIdxs[m] = [
                    tempModel.decisionVarsIdxStarts["Y_MRB_Delta_MK"]
                    + m*(len(tempModel.R)*len(tempModel.B) + len(tempModel.K))
                    + len(tempModel.R)*len(tempModel.B)
                    + k]
            varCoeffs[m] = [1]*len(varIdxs[m])

        tempModel.linear_constraints.add(
                lin_expr=[
                        tempModel.SparsePair(
                                ind=varIdxs[m],
                                val=varCoeffs[m])
                        for m in tempModel.M],
                senses=["E"]*len(varIdxs),
                rhs=[1]*len(varIdxs))

        """ MODIFY THE COEFFICIENTS AND COMPONENTS OF MANY OTHER CONSTRAINTS """
        """ CONSTRAINT 2 """
        coefficients = [
                (tempModel.constraintIdxStarts["C_2"]
                 + k*len(tempModel.C)*len(tempModel.N) + c*len(tempModel.N) + n,
                 tempModel.decisionVarsIdxStarts["X_RB"]
                 + r*len(tempModel.B) + b,
                 tempModel.d3_RCN[r, c, n])
                for r in tempModel.R
                for b in tempModel.B
                for k in tempModel.K
                for c in tempModel.C
                for n in tempModel.N]

        coefficients.extend([
                (tempModel.constraintIdxStarts["C_2"]
                 + k*len(tempModel.C)*len(tempModel.N) + c*len(tempModel.N) + n,
                 tempModel.decisionVarsIdxStarts["Y_MRB_Delta_MK"]
                 + m*(len(tempModel.R)*len(tempModel.B) + len(tempModel.K))
                 + r*(len(tempModel.B)) + b,
                 -tempModel.d3_RCN[r, c, n])
                for r in tempModel.R
                for b in tempModel.B
                for m in tempModel.M
                for k in tempModel.K
                for c in tempModel.C
                for n in tempModel.N])

        tempModel.linear_constraints.set_coefficients(coefficients)

        """ CONSTRAINT 3 """
        coefficients = [
                (tempModel.constraintIdxStarts["C_3"]
                 + k*len(tempModel.C)*len(tempModel.N) + c*len(tempModel.N) + n,
                 tempModel.decisionVarsIdxStarts["X_RB"]
                 + r*len(tempModel.B) + b,
                 tempModel.d3_RCN[r, c, n]/tempModel.no_CB[c, b])
                for r in tempModel.R
                for b in tempModel.B
                for k in tempModel.K
                for c in tempModel.C
                for n in tempModel.N]

        coefficients.extend([
                (tempModel.constraintIdxStarts["C_3"]
                 + k*len(tempModel.C)*len(tempModel.N) + c*len(tempModel.N) + n,
                 tempModel.decisionVarsIdxStarts["Y_MRB_Delta_MK"]
                 + m*(len(tempModel.R)*len(tempModel.B) + len(tempModel.K))
                 + r*(len(tempModel.B)) + b,
                 -tempModel.d3_RCN[r, c, n]/tempModel.no_CB[c, b])
                for r in tempModel.R
                for b in tempModel.B
                for m in tempModel.M
                for k in tempModel.K
                for c in tempModel.C
                for n in tempModel.N])

        tempModel.linear_constraints.set_coefficients(coefficients)

        """ CONSTRAINT 7 """
        coefficients = [
                (tempModel.constraintIdxsStarts["C_7"]
                 + r*len(tempModel.B) + b,
                 tempModel.decisionVarsIdxStarts["Y_MRB_Delta_MK"]
                 + m*(len(tempModel.R)*len(tempModel.B) + len(tempModel.K))
                 + r*(len(tempModel.B)) + b,
                 1)
                for r in tempModel.R
                for b in tempModel.B
                for m in tempModel.M]

        tempModel.linear_constraints.set_coefficients(coefficients)

        """ CONSTRAINT 10 """
        coefficients = [
                (tempModel.constraintIdxStarts["C_10"] + r,
                 tempModel.decisionVarsIdxStarts["Y_MRB_Delta_MK"]
                 + m*(len(tempModel.R)*len(tempModel.B) + len(tempModel.K))
                 + r*(len(tempModel.B)) + b,
                 tempModel.d2_RM[r, m])
                for r in tempModel.R
                for b in tempModel.B
                for m in tempModel.M]

        coefficients.extend([
                (tempModel.constraintIdxStarts["C_10"] + r,
                 tempModel.decisionVarsIdxStarts["X_RB"]
                 + r*len(tempModel.B) + b,
                 tempModel.d1_RB[r, b])
                for r in tempModel.R
                for b in tempModel.B])

        tempModel.linear_constraints.set_coefficients(coefficients)

        tempModel.linear_constraints.set_rhs([
                (tempModel.constraintIdxStarts["C_10"] + r,
                 tempModel.Gmax_R[r] - tempModel.G_R[r])
                for r in tempModel.R])

        """ SOLVE THE MODEL """
        tempModel.solve()

        """ UPDATE THE RESOURCE ASSIGNMENTS IN THE SYSTEM """
        assignments = numpy.empty([len(tempModel.R), 2])

        x_RB = [[round(tempModel.solution.get_values(
                       tempModel.decisionVarsIdxStarts["X_RB"]
                       + r*len(tempModel.B) + b))
                 for b in tempModel.B]
                for r in tempModel.R]

        for r in tempModel.R:
            assignments[r-1, 0] = x_RB[r-1].index(1)

        y_RBM = [[[round(tempModel.solution.get_values(
                         tempModel.decisionVarsIdxStarts["Y_MRB_Delta_MK"]
                         + m*(len(tempModel.R)*len(tempModel.B)
                         + len(tempModel.K)) + r*(len(tempModel.B)) + b))
                   for m in tempModel.M]
                  for b in tempModel.B]
                 for r in tempModel.R]

        for r in tempModel.R:
            for b in tempModel.B:
                for m in tempModel.M:
                    if y_RBM[r, b, m] == 1:
                        assignments[r-1, 1] = m
                        break

        """ Update the attack configurations for each patch and active fire """
        configsN = [[round(tempModel.solution.get_values(
                           tempModel.decisionVarsIdxStarts["Delta_NK"]
                           + n*len(tempModel.K) + k))
                     for k in tempModel.K]
                    for n in tempModel.N]

        patchConfigs = [configsN[n-1].index(1) for n in tempModel.N]

        configsM = [[round(tempModel.solution.get_values(
                           tempModel.decisionVarsIdxStarts["Y_RBM_Delta_MK"]
                           + m*(len(tempModel.R)*len(tempModel.B)
                           + len(tempModel.K))
                           + len(tempModel.R)*len(tempModel.B) + k))
                     for k in tempModel.K]
                    for m in tempModel.M]

        fireConfigs = [configsM[m-1].index(1) for m in tempModel.M]

        return [assignments, patchConfigs, fireConfigs]

    def simulateStatic(self):
        """ This routine uses the expected weather for the ENTIRE horizon to
        perform a base assignment at the start of the day. Fire assignments are
        still computed at each time step but aircraft cannot relocate between
        bases. """
        region = self.model.getRegion()
        timeSteps = self.model.getTotalSteps()
        patches = len(region.getPatches())
        resources = region.getResources()
        fires = region.getFires()
        configs = self.model.getConfigurations()

        """ Initial assignment of aircraft to bases (Col 1) and fires (Col 2)
        A value of zero indicates no assignment (only applicable for fires) """
        assignments = numpy.empty([len(resources) , 2])

        regionSize = region.getX().size
        samplePaths = self.model.getROVPaths()
        samplePaths2 = self.model.getMCPaths()
        lookahead = self.model.getLookahead()

        self.finalDamageMaps = [None]*samplePaths
        self.expectedDamages = [None]*samplePaths

        wg = region.getWeatherGenerator()

        for ii in range(samplePaths):
            damage = 0
            assignmentsPath = [None]*(timeSteps + 1)
            assignmentsPath[0] = copy.copy(assignments)
            firesPath = copy.copy(fires)
            resourcesPath = copy.copy(resources)
            activeFires = [fire for fire in firesPath]
            self.finalDamageMaps[ii] = numpy.empty([timeSteps + 1, patches])

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
            accumulatedDamage = numpy.zeros([timeSteps+1+lookahead, patches])

        """ Initial time step """


        """ All remaining time steps """
        for tt in range(1, timeSteps):
            pass

    def simulateSinglePeriod(self, assignmentsPath, resourcesPath,
                             firesPath, activeFires, accumulatedDamage,
                             patchConfigs, fireConfigs, ffdi, tt):
        """ This routine updates the state of the system given decisions """
        damage = 0
        patches = self.model.getRegion().getPatches()

        """ Fight existing fires """
        inactiveFires = []

        for fire in range(len(activeFires)):
            sizeOld = activeFires[fire].getSize()

            activeFires[fire].growFire(
                    self.model,
                    ffdi[activeFires[fire].getPatchID()],
                    random=True)

            sizeCurr = activeFires[fire].getSize()

            if sizeCurr - sizeOld <= 1e-6:
                # Extinguish fire and remove from list of active fires
                inactiveFires.append(fire)

            else:
                damage += sizeCurr - sizeOld

            """ Update damage map for area burned for existing fires """
            accumulatedDamage[tt, activeFires[fire].getPatchID()] += (
                    sizeCurr - sizeOld)

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
            nfPatch = patches[patch].newFires(self, self.model, ffdi[patch])
            activeFires.extend(nfPatch)

            for fire in nfPatch:
                damage += fire.getSize()
                accumulatedDamage[tt, patch] += fire.getSize()

        """ Return incremental damage for region """
        return damage


    def copyNonCplexComponents(self, tempModel, copyModel):
        """ This routine only copies components that are immutable """
        tempModel.R = copy.copy(copyModel.R)
        tempModel.B = copy.copy(copyModel.B)
        tempModel.N = copy.copy(copyModel.N)
        tempModel.T = copy.copy(copyModel.T)
        tempModel.K = copy.copy(copyModel.K)
        tempModel.C = copy.copy(copyModel.C)
        tempModel.M_All = copy.copy(copyModel.M_All)
        tempModel.lambdas = copy.copy(copyModel.lambdas)
        tempModel.Gmax_R = copy.copy(copyModel.Gmax_R)
        tempModel.Q_KC = copy.copy(copyModel.Q_KC)

    def expectedDamageExisting(self, fire, ffdiPath, config, time):
        damage = 0
        fireTemp = copy.copy(fire)
        patch = fire.getPatchID()

        for tt in self.model.getLookahead():
            ffdi = ffdiPath[time, patch]
            originalSize = fireTemp.getSize()
            fireTemp.growFire(self.model, ffdi)
            damage += (fireTemp.getSize() - originalSize)

        return damage

    def expectedDamagePotential(self, patchID, ffdiPath, config, time):
        damage = 0
        vegetation = self.model.getRegion().getVegetations()[
                self.model.getRegion().getPatches()[patchID]
                .getVegetations()[0][0]]

        for tt in range(time, time + self.model.getLookahead()):
            # Only look at the expected damage of fires started at this time
            # period to the end of the horizon
            ffdi = ffdiPath[tt, patchID]

            occ = numpy.interp(ffdi,
                               vegetation.getFFDIRange(),
                               vegetation.getOccurrence())

            size = numpy.interp(ffdi,
                                vegetation.getFFDIRange(),
                                vegetation.getInitialSize())

            growth = 1

            for t2 in range(tt, self.model.getLookahead() + time):
                ffdi = ffdiPath[t2, patchID]

                grMean = numpy.interp(ffdi,
                          vegetation.getFFDIRange(),
                          vegetation.getROCA2PerHourMean())

                growth = growth*(1 + grMean)

            damage += occ*size*growth

    """////////////////////////////////////////////////////////////////////////
    ////////////////////////// ROV Routines for later /////////////////////////
    ////////////////////////////////////////////////////////////////////////"""

    def forwardPaths(self):
        os.system("taskset -p 0xff %d" % os.getpid())
        # We don't need to store the precipitation, wind, and temperature
        # matrices over time. We only need the resulting danger index

        paths = [None]*self.model.getROVPaths()
        for pathNo in range(self.model.getROVPaths()):
            paths[pathNo] = self.initialForwardPath()

        return paths

    def rov(self, exogenousPaths, randCont, endogenousPaths):
        pass

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

    def writeOutResults(self):
        root = ("../Experiments/Experiments/" +
                self.model.getInputfile().split(
                        "../Experiments/Experiments/")[1].split("/")[0])

        maxFFDI = max(self.realisedFFDIs)
        minFFDI = min(self.realisedFFDIs)

        """ Output folder """
        outfolder = root + "/Outputs"

        for run in self.model.getRuns():
            subOutfolder = outfolder + "/Run_" + str(run)

            """ FFDI Data """
            outputfile = subOutfolder + "FFDI_Paths.csv"
            with open(outputfile, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')

                # Compute columns
                columns = int(self.model.getTotalSteps()
                              + self.model.getLookahead()
                              + 1)

                writer.writerow(
                        ['FFDI_PATHS']
                        + ['']*(columns-1))


                writer.writerow(
                        ['PATHS', str(len(self.realisedFFDIs))]
                        + ['']*(columns-2))

                writer.writerow(
                        ['PATH_' + str(run + 1), 'HOUR']
                        + ['']*(columns-2))

                writer.writerow(
                        ['PATCH']
                        + [str(h + 1)
                           for h in range(len(
                                   self.model.getTotalSteps()
                                   + self.model.getLookahead()))])

                writer.writerows([
                        [str(patch + 1)]
                        + self.realisedFFDIs[run, patch, :].tolist()
                        for patch in range(len(
                                self.model.getRegion().getPatches()))])

                """ FFDI Map """
                outputGraphs = pdf.PdfPages(subOutfolder + "FFDI.pdf")
                for tt in range(self.model.getTotalSteps()
                                + self.model.getLookahead()):

                    rawPatches = len(self.model.getX())
                    fig, ax = plt.subplots()

                    patches = []
                    cmap = clrmp.get_cmap('Oranges')

                    for rp in range(rawPatches):
                        polygon = Polygon(
                                self.model.getVertices()[rp],
                                edgecolor="black",
                                facecolor=cmap((
                                        self.realisedFFDIs[run, rp, tt] -
                                        minFFDI)/maxFFDI),
                                closed=True)
                        patches.append(polygon)

                    p = PatchCollection(patches)
                    ax.add_collection(p)
                    fig.colorbar(p, ax=ax)

                    outputGraphs.savefig(fig)
                    plt.close(fig)

                if self.model.getAlgo() > 0:
                    subOutfolder = outfolder + "/Run_" + str(run)

                    outputfile = subOutfolder + "Resource_States.csv"

                    with open(outputfile, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')

                        columns = self.model.getTotalSteps()*5
                        # Aircraft Hours and Positions

                        row = ['RESOURCE_ID', 'TYPE', 'ACCUMULATED_HOURS']

                        row.extend(['']*(self.model.getTotalSteps() - 1))
                        row.extend([a + str(ii)
                                    for ii in range(self.model.getTotalSteps()
                                                    - 1)
                                    for a in ['X_POSITION_T', 'Y_POSITION_T']])
                        row.append('BASE_IDS')
                        row.extend(['']*(self.model.getTotalSteps() - 1))
                        row.append('FIRE_IDS')
                        row.extend(['']*(self.model.getTotalSteps() - 1))
                        writer.writerow(row)

                        resources = self.model.getRegion().getResources()
                        helis = self.model.getRegion().getHelicopters()
                        tankers = self.model.getRegion().getAirTankers()

                        for ii in range(resources):
                            row = [
                                str(ii),
                                'Fixed' if resource[ii] in tankers else False,
                                self.accumulatedHours[run, ii, tt]]
                            row.extend([
                                    self.aircraftPositions[tt, ii, pp]
                                    for ii in range(self.model.getTotalSteps()
                                                    - 1)
                                    for pp in [0, 1]])
                            row.extend([
                                    self.realisedAssignments[run][tt][ii, 0]
                                    for tt in range(len(
                                            self.model.getTotalSteps()))])
                            row.extend([
                                    self.realisedAssignments[run][tt][ii, 1]
                                    for tt in range(len(
                                            self.model.getTotalSteps()))])

                            writer.writerow(row)

        self.realisedFires = []
        self.expectedDamages = [None]*samplePaths

                    """ Accumulated Damage Per Patch """
                    outputfile = subOutfolder + "Accumulated_Patch_Damage.csv"

                    with open(outputfile, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')

                    self.finalDamageMaps = [None]*samplePaths

                    """ Active fires """
                    for tt in range(self.model.getTotalSteps()):
                        outputfile = subOutfolder + "Resource_States.csv"

                        with open(outputfile, 'w', newline='') as csvfile:
                            writer = csv.writer(csvfile, delimiter=',')

                    """ Accumulated Damage Maps """
                    for tt in range(self.model.getTotalSteps()):
                        # Damage heat map
                        self.finalDamageMaps = [None]*samplePaths

                        # Bases

                        # Fires

                        # Base assignment annotations

                        # Fire assignment annotations

                if self.model.getAlgo() == 2:
                    """ ROV Regressions """
                    pass

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

    @staticmethod
    def computeFFDI(temp, rh, wind, df):
        return 2*numpy.exp(-0.45 + 0.987*numpy.log(df) -
                           0.0345*rh + 0.0338*temp + 0.0234*wind)
