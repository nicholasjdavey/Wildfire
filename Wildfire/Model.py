# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:10:43 2017

@author: davey
"""

import numpy
import math
import csv
import copy
import sys
import geopandas as gp

from Control import Control
from Patch import Patch
from pathlib import Path
from Region import Region
from AirStrip import AirStrip
from Tanker import Tanker
from Heli import Heli
from Land import Land
from Base import Base
from Vegetation import Vegetation
from Fire import Fire
from Simulation import Simulation
from WeatherGenerator import WeatherGenerator
from VariableParameters import VariableParameters
from ExperimentalScenario import ExperimentalScenario


class Model():
    # Class for defining a model region

    def __init__(self):
        # Constructs an instance
        self.inputfile = ""
        self.shapefile = ""
        self.xDataFile = ""
        self.yDataFile = ""
        self.zDataFile = ""
        self.weatherDataFile = ""
        self.basesDataFile = ""
        self.occurrenceDataFile = ""
        self.exDamageDataFile = ""
        self.configurationsDataFile = ""
        self.aircraftDataFiles = []
        self.landcraftDataFiles = []
        self.vegetationDatafiles = []
        self.simulations = []
        self.resourceTypes = []
        self.variableParameters = None
        self.controls = []
        self.configurations = {}
        self.usefulConfigurationsE = []
        self.usefulConfigurationsP = []
        self.region = None
        self.stepSize = 0
        self.totalSteps = 0
        self.relocThreshold = 0
        self.hoursPerDay = 0
        self.rovPaths = 0
        self.nestedOptMethod = 0
        self.lookahead = 0
        self.coverTime = 0
        self.mcPaths = 0
        self.discountFactor = 0
        self.algo = 0
        self.samplePaths = []
        self.useSamplePaths = False
        self.runs = 0
        self.baseThreshold = 0
        self.fireThreshold = 0
        self.controlMethod = 0
        self.shape = None
        self.plot = False
        self.resolution = 1
        self.expDMethod = 1
        self.lpTimeout = 300
        self.mipGap = 0
        self.saveSchedule = 0

    def getShape(self):
        return self.shape

    def setShape(self, s):
        self.shape = s

    def getPlot(self):
        return self.plot

    def setPlot(self, p):
        self.plot = p

    def getROVResolution(self):
        return self.resolution

    def setROVResolution(self, r):
        self.resolution = r

    def getExpDMethod(self):
        return self.expDMethod

    def setExpDMethod(self, m):
        self.expDMethod = m

    def getLPTimeout(self):
        return self.lpTimeout

    def setLPTimeout(self, t):
        self.lpTimeout = t

    def getMIPGap(self):
        return self.mipGap

    def setMIPGap(self, g):
        self.mipGap = g

    def getSaveSchedule(self):
        return self.saveSchedule

    def setSaveSchedule(self, s):
        self.saveSchedule = s

    def getFFDIMR(self):
        return self.ffdiMR

    def setFFDIMR(self, f):
        self.ffdiMR = f

    def getFFDISD(self):
        return self.ffdiSD

    def setFFDISD(self, f):
        self.ffdiSD = f

    def getROVPaths(self):
        return self.rovPaths

    def setROVPaths(self, r):
        self.rovPaths = r

    def getStepSize(self):
        return self.stepSize

    def setStepSize(self, s):
        self.stepSize = s

    def getTotalSteps(self):
        return self.totalSteps

    def setTotalSteps(self, s):
        self.totalSteps = s

    def getRelocThreshold(self):
        return self.relocThreshold

    def setRelocThreshold(self, t):
        self.relocThreshold = t

    def getHoursPerDay(self):
        return self.hoursPerDay

    def setHoursPerDay(self, h):
        self.hoursPerDay = h

    def getInputFile(self):
        return self.inputfile

    def setInputFile(self, i):
        self.inputfile = i

    def getShapefile(self):
        return self.shapefile

    def setShapefile(self, s):
        self.shapefile = s

    def getXDataFile(self):
        return self.xDataFile

    def setXDataFile(self, x):
        self.xDataFile = x

    def getYDataFile(self):
        return self.yDataFile

    def setYDataFile(self, y):
        self.xDataFile = y

    def getZDataFile(self):
        return self.zDataFile

    def setZDataFile(self, z):
        self.zDataFile = z

    def getVegetationDataFiles(self):
        return self.vegetationDataFiles

    def setVegetationDataFiles(self, v):
        self.vegetationDataFiles

    def getWeatherDataFile(self):
        return self.weatherDataFile

    def setWeatherDataFile(self, w):
        self.weatherDataFile = w

    def getBasesDataFile(self):
        return self.basesDataFile

    def setBasesDataFile(self, b):
        self.basesDataFile = b

    def getOccurrenceDataFile(self):
        return self.occurrenceDataFile

    def setOccurrenceDataFile(self, o):
        self.occurrenceDataFile = o

    def getExDamageDataFile(self):
        return self.exDamageDataFile

    def setExDamageDataFile(self, e):
        self.exDamageDataFile = e

    def getConfigurationsDataFile(self):
        return self.configurationsDataFile

    def setConfigurationsDataFile(self, f):
        self.configurationsDataFile = f

    def getAircraftDataFiles(self):
        return self.aircraftDataFiles

    def setAircraftDataFiles(self, a):
        self.aircraftDataFiles = a

    def getLandcraftDataFiles(self):
        return self.landcraftDataFiles

    def setLandcraftDataFiles(self, l):
        self.landcraftDataFiles = l

    def getResourceTypes(self):
        return self.resourceTypes

    def setResourceTypes(self, r):
        self.resourceTypes = r

    def getSimulations(self):
        return self.simulations

    def setSimulations(self, s):
        self.simulations = s

    def getVariableParameters(self):
        return self.variableParameters

    def setVariableParameters(self, v):
        self.variableParameters = v

    def getControls(self):
        return self.controls

    def setControls(self, c):
        self.controls = c

    def getConfigurations(self):
        return self.configurations

    def setConfigurations(self, c):
        self.configurations = c

    def getUsefulConfigurationsExisting(self):
        return self.usefulConfigurationsE

    def setUsefulConfigurationsExisting(self, uc):
        self.usefulConfigurationsE = uc

    def getUsefulConfigurationsPotential(self):
        return self.usefulConfigurationsP

    def setUsefulConfigurationsPotential(self, uc):
        self.usefulConfigurationsP = uc

    def getRegion(self):
        return self.region

    def setRegion(self, r):
        self.region = r

    def getNestedOptMethod(self):
        return self.nestedOptMethod

    def setNestedOptMethod(self, m):
        self.nestedOptMethod = m

    def getLookahead(self):
        return self.lookahead

    def setLookahead(self, l):
        self.lookahead = l

    def getCoverTime(self):
        return self.coverTime

    def setCoverTime(self, t):
        self.coverTime = t

    def getMCPaths(self):
        return self.mcPaths

    def setMCPaths(self, p):
        self.mcPaths = p

    def getDiscountFactor(self):
        return self.discountFactor

    def setDiscountFactor(self, f):
        self.discountFactor = f

    def getAlgo(self):
        return self.algo

    def setAlgo(self, a):
        self.algo = a

    def getSamplePaths(self):
        return self.samplePaths

    def setSamplePaths(self, p):
        self.samplePaths = p

    def getUseSamplePaths(self):
        return self.useSamplePaths

    def setUseSamplePaths(self, u):
        self.useSamplePaths = u

    def getRuns(self):
        return self.runs

    def setRuns(self, r):
        self.runs = r

    def getFireThreshold(self):
        return self.fireThreshold

    def setFireThreshold(self, f):
        self.fireThreshold = f

    def getBaseThreshold(self):
        return self.baseThreshold

    def setBaseThreshold(self, b):
        self.baseThreshold = b

    def getControlMethod(self):
        return self.controlMethod

    def setControlMethod(self, c):
        self.controlMethod = c

    def generatePlots(self, setting):
        self.plots = setting

    def readInSourceData(self, filename):
        # First read the raw source files
        self.inputfile = filename
        inputfile = open(filename, "r")
        contents = inputfile.readlines()
        self.shapefile = contents[13].split(":")[1].strip()
        self.xDataFile = contents[14].split(":")[1].strip()
        self.yDataFile = contents[15].split(":")[1].strip()
        self.zDataFile = contents[16].split(":")[1].strip()
        self.vegetationDataFile = contents[17].split(":")[1].strip()
        self.valueDataFile = contents[17].split(":")[1].strip()
        self.weatherDataFile = contents[21].split(":")[1].strip()
        self.basesDataFile = contents[26].split(":")[1].strip()
        self.configurationsDataFile = contents[32].split(":")[1].strip()

        noAircraft = int(contents[33].split(":")[1].strip())
        aircraftData = []

        for ii in range(noAircraft):
            aircraftData.append(contents[34 + ii].split(":")[1].strip())

        self.aircraftDataFiles = aircraftData

        noLandcraft = int(contents[39+noAircraft].split(":")[1].strip())
        landcraftData = []

        for ii in range(noLandcraft):
            landcraftData.append(contents[40 + noAircraft + ii].split(":")[1]
                                 .strip())

        self.landcraftDataFiles = landcraftData

        noVegetations = int(contents[45 + noAircraft + noLandcraft]
                            .split(":")[1].strip())
        vegetationsData = []

        for ii in range(noVegetations):
            vegetationsData.append(contents[46 + noAircraft + noLandcraft]
                                   .split(":")[1].strip())

        self.vegetationDataFiles = vegetationsData

        # These controls are not used for the fourth type of linear program
        noControls = int(contents[62 + noAircraft + noLandcraft+
                                  noVegetations].split(":")[1]
                         .strip())

        for ii in range(noControls):
            control = Control()
            varsStr = (contents[64 + noAircraft + noLandcraft + noVegetations
                                + ii]
                       .split(":")[1])
            lambda1 = float(varsStr.split()[0].strip())
            lambda2 = float(varsStr.split()[1].strip())
            control.setLambda1(lambda1)
            control.setLambda2(lambda2)
            self.controls.append(control)

        varParams = VariableParameters()

        varsStr = (contents[67 + noAircraft + noLandcraft + noVegetations
                            + noControls]
                   .split(":")[1].strip())
        varsStrs = varsStr.split(",")
        varsFloat = [float(varsStrs[ii]) for ii in range(len(varsStrs))]
        varParams.setSpeedMultipliers(varsFloat)

        varsStr = (contents[68 + noAircraft + noLandcraft + noVegetations
                            + noControls]
                   .split(":")[1].strip())
        varsStrs = varsStr.split(",")
        varsFloat = [float(varsStrs[ii]) for ii in range(len(varsStrs))]
        varParams.setOccurrenceProbMultipliers(varsFloat)

        varsStr = (contents[69 + noAircraft + noLandcraft + noVegetations
                            + noControls]
                   .split(":")[1].strip())
        varsStrs = varsStr.split(",")
        varsFloat = [float(varsStrs[ii]) for ii in range(len(varsStrs))]
        varParams.setDamageIntensityMultipliers(varsFloat)

        varsStr = (contents[70 + noAircraft + noLandcraft + noVegetations
                            + noControls]
                   .split(":")[1].strip())
        varsStrs = varsStr.split(",")
        varsFloat = [float(varsStrs[ii]) for ii in range(len(varsStrs))]
        varParams.setWeatherUncertMultipliers(varsFloat)

        self.totalSteps = int(contents[71 + noAircraft + noLandcraft
                                       + noVegetations + noControls]
                              .split(":")[1]
                              .strip())

        self.stepSize = float(contents[72 + noAircraft + noLandcraft
                                       + noVegetations + noControls]
                              .split(":")[1].strip())
        self.hoursPerDay = float(contents[73 + noAircraft + noLandcraft
                                          + noVegetations + noControls]
                                 .split(":")[1]
                                 .strip())

        self.rovPaths = int(contents[74 + noAircraft + noLandcraft
                                     + noVegetations + noControls]
                            .split(":")[1]
                            .strip())

        self.nestedOptMethod = int(contents[75 + noAircraft + noLandcraft
                                            + noVegetations + noControls]
                                   .split(":")[1]
                                   .strip())

        self.lookahead = int(contents[76 + noAircraft + noLandcraft
                                      + noVegetations + noControls]
                             .split(":")[1]
                             .strip())

        self.coverTime = float(contents[77 + noAircraft + noLandcraft
                                        + noVegetations + noControls]
                               .split(":")[1]
                               .strip())

        self.mcPaths = int(contents[78 + noAircraft + noLandcraft
                                    + noVegetations + noControls]
                           .split(":")[1]
                           .strip())

        self.discountFactor = float(contents[79 + noAircraft + noLandcraft
                                             + noVegetations + noControls]
                                    .split(":")[1]
                                    .strip())

        self.algo = float(contents[80 + noAircraft + noLandcraft
                                   + noVegetations + noControls]
                          .split(":")[1]
                          .strip())

        usefulConfigs = (contents[81 + noAircraft + noLandcraft
                                  + noVegetations + noControls]
                         .split(":")[1].strip())
        usefulConfigsSplit = usefulConfigs.split(",")
        self.usefulConfigurationsE = numpy.array([
                int(usefulConfigsSplit[ii])
                for ii in range(len(usefulConfigsSplit))])

        usefulConfigs = (contents[82 + noAircraft + noLandcraft
                                  + noVegetations + noControls]
                         .split(":")[1].strip())
        usefulConfigsSplit = usefulConfigs.split(",")
        self.usefulConfigurationsP = numpy.array([
                int(usefulConfigsSplit[ii])
                for ii in range(len(usefulConfigsSplit))])

        self.useSamplePaths = [
                True if int(
                        contents[83 + noAircraft + noLandcraft
                                 + noVegetations + noControls]
                        .split(":")[1].strip()) == 1
                else False]

        self.runs = int(contents[84 + noAircraft + noLandcraft
                                 + noVegetations + noControls]
                        .split(":")[1].strip())

        self.fireThreshold = float(contents[85 + noAircraft + noLandcraft
                                            + noVegetations + noControls]
                                   .split(":")[1].strip())

        self.baseThreshold = float(contents[86 + noAircraft + noLandcraft
                                            + noVegetations + noControls]
                                   .split(":")[1].strip())

        self.controlMethod = int(contents[87 + noAircraft + noLandcraft
                                          + noVegetations + noControls]
                                 .split(":")[1].strip())

        plotVal = int(contents[88 + noAircraft + noLandcraft
                               + noVegetations + noControls]
                               .split(":")[1].strip())
        self.plot = True if plotVal == 1 else False
        self.resolution = int(contents[89 + noAircraft + noLandcraft
                                       + noVegetations + noControls]
                                       .split(":")[1].strip())

        self.ffdiMR = float(contents[90 + noAircraft + noLandcraft
                                     + noVegetations + noControls]
                                     .split(":")[1].strip())
        self.ffdiSD = float(contents[91 + noAircraft + noLandcraft
                                     + noVegetations + noControls]
                                     .split(":")[1].strip())
        self.expDMethod = int(contents[92 + noAircraft + noLandcraft
                                       + noVegetations + noControls]
                                       .split(":")[1].strip())
        self.lpTimeout = int(contents[93 + noAircraft + noLandcraft
                                       + noVegetations + noControls]
                                       .split(":")[1].strip())
        self.mipGap = int(contents[94 + noAircraft + noLandcraft
                                   + noVegetations + noControls]
                                   .split(":")[1].strip())
        self.saveSchedule = (
                True
                if int(contents[95 + noAircraft + noLandcraft + noVegetations
                                + noControls].split(":")[1].strip()) == 1
                else False)

        self.variableParameters = varParams

        # Initialise the simulations
        for ii in range(len(varParams.getSpeedMultipliers())):
            for jj in range(len(varParams.getOccurrenceProbMultipliers())):
                for kk in range(len(varParams
                                    .getDamageIntensityMultipliers())):
                    for ll in range(len(varParams
                                        .getWeatherUncertMultipliers())):
                        sim = Simulation()
                        es = ExperimentalScenario()
                        es.setSpeedMultIdx(varParams.getSpeedMultipliers()[ii])
                        es.setOccProbMultIdx(
                            varParams.getOccurrenceProbMultipliers()[jj])
                        es.setDamIntMultIdx(
                            varParams.getDamageIntensityMultipliers()[kk])
                        es.setWeatherUncertMultIdx(
                            varParams.getWeatherUncertMultipliers()[ll])
                        sim.setExperimentalScenario(es)
                        sim.setControls(self.controls)
                        sim.setModel(self)
                        self.simulations.append(sim)

        inputfile.close()

    def populateParameters(self):
        # First, initialise the Region object for the simulation
        self.region = Region()

        # Check if configuration files already exist
        root = ("../Experiments/Experiments/" +
                self.inputfile.split("../Experiments/Experiments/")[1]
                .split("/")[0])

        # Get resource details
        # Aircraft are first, then Helicopters
        # Tankers
        # Indices for vehicles at future calls refer to the order created here.
        # Order is A1...AI,H1...HJ,T1...TK
        # WE ONLY HAVE ONE OF EACH TYPE FOR NOW!
        resources = []
        aircraftDetails = []
        with open("../" + self.aircraftDataFiles[0]) as tf:
            reader = csv.reader(tf)
            aircraftDetails = [r[0] for r in reader]
        aircraft = Tanker()
        aircraft.setFlyingHours(float(aircraftDetails[4].split(":")[1]))
        aircraft.setMaxDailyHours(float(aircraftDetails[5].split(":")[1]))
        aircraft.setCapacity(float(aircraftDetails[6].split(":")[1]))
        aircraft.setSpeed(float(aircraftDetails[7].split(":")[1]))
        resources.append(aircraft)

        # Helicopters
        helicopterDetails = []
        with open("../" + self.aircraftDataFiles[1]) as hf:
            reader = csv.reader(hf)
            helicopterDetails = [r[0] for r in reader]
        heli = Heli()
        heli.setFlyingHours(float(helicopterDetails[4].split(":")[1]))
        heli.setMaxDailyHours(float(helicopterDetails[5].split(":")[1]))
        heli.setCapacity(float(helicopterDetails[6].split(":")[1]))
        heli.setSpeed(float(helicopterDetails[7].split(":")[1]))
        resources.append(heli)

        # Firetrucks
        truckDetails = []
        with open("../" + self.landcraftDataFiles[0]) as hf:
            reader = csv.reader(hf)
            truckDetails = [r[0] for r in reader]
        vehicle = Land()
        vehicle.setCrewSize(float(truckDetails[4].split(":")[1]))
        vehicle.setCapacity(float(truckDetails[5].split(":")[1]))
        vehicle.setSpeed(float(truckDetails[6].split(":")[1]))
        resources.append(vehicle)

        self.resourceTypes = resources

        # Configurations
        with open("../" + self.configurationsDataFile) as cf:
            reader = csv.reader(cf)
            rows = [r for r in reader]

            # For now we only have tankers and heli configurations. For each,
            # they can be early or late hence we only have four components of
            # the encoding:
            # TE HE TL HL
            iterator = 2
            configCount = 1
            while iterator < len(rows):
                config = [int(rows[iterator][ii]) for ii in range(1, 5)]
                self.configurations[configCount] = config
                configCount += 1

                iterator += 1

        # Vegetation details
        vegetations = []
        for ii in range(len(self.vegetationDataFiles)):
            with open("../" + self.vegetationDataFiles[ii]) as vp:
                reader = csv.reader(vp)
                rows = [r for r in reader]

                iterator = 0
                while iterator < len(rows):
                    vegetation = Vegetation()
                    vegetation.setName(rows[iterator][1])
                    iterator = iterator + 1
                    vegetation.setFlammability(rows[iterator][2])
                    iterator = iterator + 1

                    vegetation.setFFDIRange(
                        numpy.array([float(col)
                                     for col in
                                     rows[iterator][2:(len(rows[iterator]))]]))

                    iterator = iterator + 2
                    occurrence = {}
                    while True:
#                    for jj in range(self.totalSteps + self.lookahead + 1):
                        if rows[iterator][0] != "":
                            break

                        occurrence[int(rows[iterator][1])] = (
                                [float(col)
                                 for col in
                                 rows[iterator][2:(len(rows[iterator]))]])

                        iterator = iterator + 1

                    vegetation.setOccurrence(occurrence)

                    iterator = iterator + 1
                    rocMean = {}
                    for config in self.configurations:
                        rocMean[config] = (
                                [float(col)
                                 for col in
                                 rows[iterator][2:(len(rows[iterator]))]])

                        iterator = iterator + 1

                    vegetation.setROCA2PerHourMean(rocMean)

                    iterator = iterator + 1
                    rocSD = {}
                    for config in self.configurations:
                        rocSD[config] = (
                                [float(col)
                                 for col in
                                 rows[iterator][2:(len(rows[iterator]))]])

                        iterator = iterator + 1

                    vegetation.setROCA2PerHourSD(rocSD)

                    iterator = iterator + 1
                    initialSuccess = {}
                    for config in self.configurations:
                        initialSuccess[config] = (
                                [float(col)
                                 for col in
                                 rows[iterator][2:(len(rows[iterator]))]])

                        iterator = iterator + 1

                    vegetation.setInitialSuccess(initialSuccess)

                    iterator = iterator + 1
                    extendedSuccess = {}
                    for config in self.configurations:
                        extendedSuccess[config] = (
                                [float(col)
                                 for col in
                                 rows[iterator][2:(len(rows[iterator]))]])

                        iterator = iterator + 1

                    vegetation.setExtendedSuccess(extendedSuccess)

                    iterator = iterator + 1
                    initialSizeMean = {}
                    for config in self.configurations:
                        initialSizeMean[config] = (
                                [float(col)
                                 for col in
                                 rows[iterator][2:(len(rows[iterator]))]])

                        iterator = iterator + 1

                    vegetation.setInitialSizeMean(initialSizeMean)

                    iterator = iterator + 1
                    initialSizeSD = {}
                    for config in self.configurations:
                        initialSizeSD[config] = (
                                [float(col)
                                 for col in
                                 rows[iterator][2:(len(rows[iterator]))]])

                        iterator = iterator + 1

                    vegetation.setInitialSizeSD(initialSizeSD)

                vegetations.append(vegetation)

        # REGION CONFIGURATION ################################################
        regionConfig = Path(root + "/Region_Configuration.csv")
        regionShapefile = Path("../" + self.configurationsDataFile)

        if regionConfig.is_file():
            # If exists, just call the pre-created data
            regionConfigFile = open(root + "/Region_Configuration.csv")

            with open(root + "/Region_Configuration.csv") as fd:
                reader = csv.reader(fd)
                rows = [r for r in reader]

            # First few lines are airstrip details
            iterator = 1
            test = True

            airStrips = []
            bases = []

            resourceAssignments = []

            totalTankers = []
            totalHelis = []
            totalTrucks = []
            totalResources = []
            # Store all the air strips
            # We only have 1 tanker and 1 helicopter at the moment
            test = True
            baseIdx = 0

            while test:
                if (all('' == s or s.isspace()
                        for s in rows[iterator][1:4]) or
                        (len(rows[iterator]) < 8)):

                    test = False
                    iterator = iterator + 2
                else:
                    airStrip = AirStrip()
                    airStrip.setLocation(numpy.array(
                        [float(rows[iterator][1]), float(rows[iterator][2])]))
                    airStrip.setMaxTankers(float(rows[iterator][5]))
                    airStrip.setMaxHelicopters(float(rows[iterator][6]))
                    noAircraft = int(rows[iterator][3])
                    aircraftList = []
                    for ii in range(noAircraft):
                        aircraft = Tanker()
                        aircraft.setFlyingHours(float(
                            aircraftDetails[4].split(":")[1]))
                        aircraft.setMaxDailyHours(float(
                            aircraftDetails[5].split(":")[1]))
                        aircraft.setCapacity(float(
                            aircraftDetails[6].split(":")[1]))
                        aircraft.setSpeed(float(
                            aircraftDetails[7].split(":")[1]))
                        aircraft.setType("A")
                        aircraft.setLocation(airStrip.getLocation())
                        aircraft.setBase(baseIdx)
                        aircraftList.append(aircraft)
                        totalTankers.append(aircraft)
                        totalResources.append(aircraft)
                        resourceAssignments.append([len(airStrips) + 1, 0])
                    airStrip.setAirTankers(aircraftList)

                    noHelicopters = int(rows[iterator][4])
                    heliList = []
                    for ii in range(noHelicopters):
                        heli = Heli()
                        heli.setFlyingHours(float(
                            helicopterDetails[4].split(":")[1]))
                        heli.setMaxDailyHours(float(
                            helicopterDetails[5].split(":")[1]))
                        heli.setCapacity(float(
                            helicopterDetails[6].split(":")[1]))
                        heli.setSpeed(float(
                            helicopterDetails[7].split(":")[1]))
                        heli.setType("H")
                        heli.setLocation(airStrip.getLocation())
                        heli.setBase(baseIdx)
                        heliList.append(heli)
                        totalHelis.append(heli)
                        totalResources.append(heli)
                        resourceAssignments.append([len(airStrips) + 1, 0])
                    airStrip.setHelicopters(heliList)
                    airStrip.setCapacity(float(rows[iterator][5]))
                    airStrips.append(airStrip)
                    baseIdx += 1
                    iterator = iterator + 1

            self.region.setAssignments(numpy.array(resourceAssignments))

            # Store all the fire stations
            test = True
            while test:
                if (all('' == s or s.isspace()
                        for s in rows[iterator][1:4]) or
                        (len(rows[iterator]) < 8)):

                    test = False
                    iterator = iterator + 2
                else:
                    base = Base()
                    base.setLocation(numpy.array([float(
                        rows[iterator][1]), float(rows[iterator][2])]))
                    noVehicles = int(rows[iterator][3])
                    vehiclesList = []

                    for ii in range(noVehicles):
                        vehicle = Land()
                        vehicle.setCrewSize(float(
                            truckDetails[4].split(":")[1]))
                        vehicle.setCapacity(float(
                            truckDetails[5].split(":")[1]))
                        vehicle.setSpeed(float(
                            truckDetails[6].split(":")[1]))
                        vehiclesList.append(vehicle)
                        totalTrucks.append(vehicle)
                        totalResources.append(vehicle)
                    base.setLandResources(vehiclesList)
                    bases.append(base)
                    iterator = iterator + 1

            # Patch data. We store matrix versions here as well as patch
            # objects.
            test = True
            x = numpy.empty([len(rows)-iterator])
            y = numpy.empty([len(rows)-iterator])
            z = numpy.empty([len(rows)-iterator])
            veg = numpy.empty([len(rows)-iterator])
            rain = numpy.empty([len(rows)-iterator])
            precip = numpy.empty([len(rows)-iterator])
            tempMin = numpy.empty([len(rows)-iterator])
            tempMax = numpy.empty([len(rows)-iterator])
            windN = numpy.empty([len(rows)-iterator])
            windE = numpy.empty([len(rows)-iterator])
            vertices = [None]*(len(rows)-iterator)
            fireSeverity = numpy.empty([len(rows)-iterator])
            fireAges = numpy.empty([len(rows)-iterator])
            ii = 0

            patches = []
            if regionShapefile.is_file():
#                shape = shapefile.Reader("../" + self.shapefile)
                self.shape = gp.GeoDataFrame.from_file("../" + self.shapefile)
                shapeCart = self.shape.copy()
                shapeCart = shapeCart.to_crs({'init': 'epsg:32633'})

            while test:
                if (all('' == s or s.isspace()
                        for s in rows[iterator][1:4]) or
                        (len(rows[iterator]) < 8)):

                    test = False
                    iterator = iterator + 2
                else:
                    z[ii] = float(rows[iterator][3])
                    veg[ii] = int(rows[iterator][4]) - 1
                    precip[ii] = float(rows[iterator][5])
                    tempMin[ii] = float(rows[iterator][6])
                    tempMax[ii] = float(rows[iterator][7])
                    rain[ii] = float(rows[iterator][8])
                    fireSeverity[ii] = float(rows[iterator][9])
                    fireAges[ii] = float(rows[iterator][10])
                    windN[ii] = float(rows[iterator][11])
                    windE[ii] = float(rows[iterator][12])

                    # If we upload a shapefile, get the vertices from there
                    if regionShapefile.is_file():
                        regionID = int(rows[iterator][0])
                        vertices[ii] = self.shape.geometry[regionID - 1]

#                        feature = shape.shapeRecords()[regionID - 1]
#                        vertices[ii] = []
#
#                        for i in range(len(feature.shape.parts)):
#                            i_start = feature.shape.parts[i]
#                            if i == len(feature.shape.parts) - 1:
#                                i_end = len(feature.shape.points)
#                            else:
#                                i_end = feature.shape.parts[i + 1]
#                            vertices[ii].append(
#                                    numpy.array(feature.shape.points[
#                                            i_start:i_end]))
                    else:
                        vertices[ii] = numpy.array(
                                [[float(rows[iterator][13 + 2*jj]),
                                  float(rows[iterator][13 + 2*jj + 1])]
                                 for jj in range(int((len(rows[iterator]) -
                                                      12)/2))
                                 if rows[iterator][13 + 2*jj] != ''])

                    # Now create the patch objects
                    patch = Patch()
                    patch.setShapefileIndex(int(rows[iterator][0]))
                    patch.setVertices(vertices[ii])
                    patch.setAvDanger([float(rows[iterator][3])])
                    patch.setAvSeverity([float(rows[iterator][4])])
                    patch.setCentroid([
                            vertices[ii].centroid.xy[0][0],
                            vertices[ii].centroid.xy[1][0]])
                    x[ii] = vertices[ii].centroid.xy[0][0]
                    y[ii] = vertices[ii].centroid.xy[1][0]
                    # Area in hectares
                    patch.setArea(shapeCart.geometry[regionID - 1].area*1e-6)
                    patch.setVegetation(
                            vegetations[int(rows[iterator][4]) - 1])
                    patches.append(patch)

                    ii = ii + 1
                    iterator = iterator + 1

            patchIDs = [patch.getShapefileIndex() for patch in patches]
            fires = []
            test = True
            while test:
                if (iterator == len(rows)):
                    test = False
                else:
                    if (all('' == s or s.isspace()
                            for s in rows[iterator][1:4]) or
                            (len(rows[iterator]) < 8)):

                        test = False
                    else:
                        fire = Fire()
                        fire.setLocation(numpy.array(
                            [float(rows[iterator][1]),
                             float(rows[iterator][2])]))
                        fire.setSize(float(rows[iterator][3]))
                        fire.setStart(float(rows[iterator][4]))
                        fire.setInitialSize(float(rows[iterator][3]))
                        fire.setFinalSize(float(rows[iterator][3]))
                        fire.setPatchID(patchIDs.index(int(rows[iterator][6])))
                        fires.append(fire)
                        iterator = iterator + 1

            # Save values to the region object
            self.region.setAirTankers(totalTankers)
            self.region.setHelicopters(totalHelis)
            self.region.setFiretrucks(totalTrucks)
            self.region.setResources(totalResources)
            self.region.setX(x[0:ii])
            self.region.setY(y[0:ii])
            self.region.setZ(z[0:ii])
            self.region.setVegetation(veg[0:ii])
            self.region.setHumidity(precip[0:ii])
            self.region.setRain(rain[0:ii])
            self.region.setTemperatureMin(tempMin[0:ii])
            self.region.setTemperatureMax(tempMax[0:ii])
            self.region.setPatches(patches[0:ii])
            self.region.setStations([airStrips, bases])
            self.region.setFireSeverity(fireSeverity[0:ii])
            self.region.setFireAge(fireAges[0:ii])
            self.region.setFires(fires)
            self.region.setWindN(windN[0:ii])
            self.region.setWindE(windE[0:ii])
            self.region.setVegetations(vegetations)
        else:
            # If not, build the data
            self.computeRegionParameters()

            # Now save it
            regionConfigFile = open(root + "/Region_Configuration.csv", "w+")

            regionConfigFile.close()
        #######################################################################

        # PRE-COMPUTED SAMPLE PATHS ###########################################
        samplePaths = Path(root + "/Sample_Paths.csv")
        if samplePaths.is_file():
            # If exists, just call the pre-created data

            with open(root + "/Sample_Paths.csv") as sp:
                reader = csv.reader(sp)
                rows = [r for r in reader]

            self.samplePaths = [None]*(int(rows[1][1]))

            noPatches = len(self.region.getX())

            for path in range(len(self.samplePaths)):
                self.samplePaths[path] = numpy.empty([
                        self.totalSteps + self.lookahead + 1,
                        noPatches])

                self.samplePaths[path] = numpy.array([
                        [rows[4 + path*(noPatches + 2) + patch][tt + 1]
                         for tt in range(self.totalSteps + self.lookahead + 1)]
                        for patch in range(noPatches)])

                self.samplePaths[path] = self.samplePaths[path].astype(
                        numpy.float)
        #######################################################################

        # WEATHER CONFIGURATION ###############################################
        weatherConfig = Path(root + "/Weather_Configuration.csv")
        if len(self.samplePaths) == 0:
            if weatherConfig.is_file():
                # If exists, just call the data
                weatherConfigFile = open(root + "/Weather_Configuration.csv")

                with open(root + "/Weather_Configuration.csv") as fd:
                    reader = csv.reader(fd)
                    rows = [r for r in reader]

                wg = WeatherGenerator()

                # Precipitation occurrence transition probabilities
                iterator = 3
                test = True

                occW2W = numpy.empty([len(self.region.getX()), self.totalSteps
                                      + self.lookahead])
                occD2W = numpy.empty([len(self.region.getX()), self.totalSteps
                                      + self.lookahead])

                for ii in range(len(self.region.getX())):
                    occD2W[ii] = numpy.array(
                            rows[iterator + ii][1:((self.totalSteps +
                                                    self.lookahead)*2 + 1):2],
                            dtype=float)
                    occW2W[ii] = numpy.array(
                            rows[iterator + ii][2:((self.totalSteps +
                                                    self.lookahead)*2 + 2):2],
                            dtype=float)

                wg.setWetProbT0Wet(occW2W)
                wg.setWetProbT0Dry(occD2W)

                iterator = iterator + 3 + len(self.region.getX())

                # Precipitation occurrence covariances
                occCov = numpy.empty([len(self.region.getX()),
                                      len(self.region.getX())])

                for ii in range(len(self.region.getX())):
                    occCov[ii] = numpy.array(
                            rows[iterator + ii][1:(len(self.region.getX())+1)],
                            dtype=float)

                wg.setWetOccurrenceCovariance(occCov)

                iterator = iterator + 3 + len(self.region.getX())

                # Precipitation amount covariances
                precipAmountCov = numpy.empty([len(self.region.getX()),
                                               len(self.region.getX())])

                for ii in range(len(self.region.getX())):
                    precipAmountCov[ii, :] = numpy.array(
                            rows[iterator + ii][1:(len(self.region.getX())+1)],
                            dtype=float)

                wg.setPrecipAmountCovariance(precipAmountCov)

                iterator = iterator + 3 + len(self.region.getX())

                # Precipitation amount parameters
                alphas = numpy.array(
                        rows[iterator][1:((self.totalSteps +
                                          self.lookahead)*2 + 1):2],
                        dtype=float)

                wg.setPrecipAlpha(alphas)

                iterator = iterator + 2

                betas = []
                betas.append(numpy.empty([self.totalSteps + self.lookahead,
                                          len(self.region.getX())]))
                betas.append(numpy.empty([self.totalSteps + self.lookahead,
                                          len(self.region.getX())]))

                for ii in range(len(self.region.getX())):
                    betas[0][:, ii] = numpy.array(
                            rows[iterator + ii][1:((self.totalSteps +
                                                    self.lookahead)*2 + 1):2],
                            dtype=float)
                    betas[1][:, ii] = numpy.array(
                            rows[iterator + ii][2:((self.totalSteps +
                                                    self.lookahead)*2 + 2):2],
                            dtype=float)

                wg.setPrecipBetas(betas)

                iterator = iterator + 3 + len(self.region.getX())

                remainingMean = numpy.empty(self.totalSteps + self.lookahead)
                remainingSD = numpy.empty(self.totalSteps + self.lookahead)
                precipCont = numpy.empty(self.totalSteps + self.lookahead)

                remainingMean = numpy.array(
                        rows[iterator][1:(self.totalSteps + self.lookahead
                                          + 1)],
                        dtype=float)
                remainingSD = numpy.array(
                        rows[iterator+1][1:(self.totalSteps + self.lookahead
                                            + 1)],
                        dtype=float)
                precipCont = numpy.array(
                        rows[iterator+2][1:(self.totalSteps + self.lookahead
                                            + 1)],
                        dtype=float)

                wg.setHumidityReductionMean(remainingMean)
                wg.setHumidityReductionSD(remainingSD)
                wg.setPrecipitationContributionMultiplier(precipCont)

                iterator = iterator + 6

                humidityCorrelations = numpy.empty([len(self.region.getX()),
                                                    len(self.region.getX())])

                for ii in range(len(self.region.getX())):
                    humidityCorrelations[ii] = numpy.array(
                            rows[iterator + ii][1:(len(self.region.getX())+1)],
                            dtype=float)

                iterator = iterator + 2 + len(self.region.getX())

                # Temperature parameters
                # Means and Standard Deviations
                meanTempWet = numpy.empty([self.totalSteps + self.lookahead])
                meanTempDry = numpy.empty([self.totalSteps + self.lookahead])
                tempSDWet = numpy.empty([self.totalSteps + self.lookahead])
                tempSDDry = numpy.empty([self.totalSteps + self.lookahead])

                wg.setTempReversion(float(rows[iterator][1]))
                iterator = iterator + 2

                meanTempWet = numpy.array(
                        rows[iterator][1:((self.totalSteps +
                                       self.lookahead)*4 + 1):4],
                        dtype=float)
                meanTempDry = numpy.array(
                        rows[iterator][2:((self.totalSteps +
                                       self.lookahead)*4 + 2):4],
                        dtype=float)
                tempSDWet = numpy.array(
                        rows[iterator][3:((self.totalSteps +
                                       self.lookahead)*4 + 3):4],
                        dtype=float)
                tempSDDry = numpy.array(
                        rows[iterator][4:((self.totalSteps +
                                       self.lookahead)*4 + 4):4],
                        dtype=float)

                wg.setTempMeanWet(meanTempWet)
                wg.setTempMeanDry(meanTempDry)
                wg.setTempSDWet(tempSDWet)
                wg.setTempSDDry(tempSDDry)

                iterator = iterator + 2

                tempAlphas = [None]*(self.totalSteps + self.lookahead)

                for ii in range(self.totalSteps + self.lookahead):
                    ta = numpy.zeros([2*len(self.region.getX()),
                                      2*len(self.region.getX())])
                    tempAlphas[ii] = ta

                for ii in range(len(self.region.getX())):
                    for jj in range(self.totalSteps + self.lookahead):
                        tempAlphas[jj][2*ii][ii*2] = (
                            float(rows[iterator+ii][jj*4+1]))
                        tempAlphas[jj][2*ii][ii*2+1] = (
                            float(rows[iterator+ii][jj*4+2]))
                        tempAlphas[jj][2*ii+1][ii*2] = (
                            float(rows[iterator+ii][jj*4+3]))
                        tempAlphas[jj][2*ii+1][ii*2+1] = (
                            float(rows[iterator+ii][jj*4+4]))

                wg.setTempA(tempAlphas)

                iterator = iterator + 3 + len(self.region.getX())

                tempBetas = [None]*(self.totalSteps + self.lookahead)
                for ii in range(self.totalSteps + self.lookahead):
                    tb = numpy.zeros([2*len(self.region.getX()),
                                      2*len(self.region.getX())])

                    for jj in range(2*len(self.region.getX())):
                        tb[jj] = numpy.array(
                                rows[iterator + jj][
                                        2:(len(self.region.getX())*2 + 2)],
                                dtype=float)

                    tempBetas[ii] = tb
                    iterator = iterator + 2*len(self.region.getX())

                wg.setTempB(tempBetas)

                iterator = iterator + 1

                # Wind parameters
                wg.setWindRegimes(int(rows[iterator][1]))

                iterator = iterator + 2

                regimeTransitions = numpy.zeros([wg.getWindRegimes(),
                                                 wg.getWindRegimes()])

                for ii in range(wg.getWindRegimes()):
                    for jj in range(wg.getWindRegimes()):
                        regimeTransitions[ii][jj] = float(rows[iterator+ii][
                                2+jj])

                wg.setWindRegimeTransitions(regimeTransitions)

                iterator = iterator + wg.getWindRegimes() + 2

                wg.setWindReversion(float(rows[iterator][1]))
                iterator = iterator + 2

                meanWindNS = numpy.array(
                        rows[iterator][1:((self.totalSteps +
                                           self.lookahead)*4 + 1):4],
                        dtype=float)
                meanWindEW = numpy.array(
                        rows[iterator][2:((self.totalSteps +
                                           self.lookahead)*4 + 2):4],
                        dtype=float)
                windSDNS = numpy.array(
                        rows[iterator][3:((self.totalSteps +
                                           self.lookahead)*4 + 3):4],
                        dtype=float)
                windSDEW = numpy.array(
                        rows[iterator][4:((self.totalSteps +
                                           self.lookahead)*4 + 4):4],
                        dtype=float)

                wg.setWindMeanNS(meanWindNS)
                wg.setWindMeanEW(meanWindEW)
                wg.setWindSDNS(windSDNS)
                wg.setWindSDEW(windSDEW)

                iterator = iterator + 2

                windAlphas = [None]*wg.getWindRegimes()

                for ii in range(wg.getWindRegimes()):
                    wa = numpy.zeros([2*len(self.region.getX()),
                                      2*len(self.region.getX())])

                    for jj in range(2*len(self.region.getX())):
                        wa[jj] = numpy.array(
                                rows[iterator + jj][
                                        2:(2*len(self.region.getX()) + 2)],
                                dtype=float)

                    windAlphas[ii] = wa

                    iterator = iterator + 2*len(self.region.getX())

                wg.setWindA(windAlphas)

                iterator = iterator + 3

                windBetas = [None]*wg.getWindRegimes()

                for ii in range(wg.getWindRegimes()):
                    wb = numpy.zeros([2*len(self.region.getX()),
                                      2*len(self.region.getX())])

                    for jj in range(2*len(self.region.getX())):
                        wb[jj] = numpy.array(
                                rows[iterator + jj][
                                        2:(2*len(self.region.getX()) + 2)],
                                dtype=float)

                    windBetas[ii] = wb

                    iterator = iterator + 2*len(self.region.getX())

                wg.setWindB(windBetas)

                wg.setRegion(self.region)
                wg.setModel(self)

                self.region.setWeatherGenerator(wg)
                weatherConfigFile.close()
            else:
                # If not, build the data
                self.computeWeatherParameters()

                # Now save it
                weatherConfigFile = open(
                        root + "/Weather_Configuration.csv", "w+")

                weatherConfigFile.close()
        #######################################################################

    def computeWeatherParameters(self):
        pass

    def computeRegionParameters(self):
        pass

    def configureRegion(self):
        # Compute distances between nodes and patches, etc.
        origin = [self.shape.bounds['minx'].min(),
                  self.shape.bounds['miny'].min()]

        # Stations
        # Air Bases
        airBases = self.region.getStations()[0]
        X = numpy.array([
                (airBase.getLocation()[0] - origin[0])*40000*math.cos(
                        (origin[1] + airBase.getLocation()[1])*math.pi/360)/360
                for airBase in airBases])
        Y = numpy.array([
                (origin[1] - airBase.getLocation()[1])*40000/360
                for airBase in airBases])

        X = numpy.transpose(numpy.tile(X, (X.shape[0], 1)))
        Y = numpy.transpose(numpy.tile(Y, (Y.shape[0], 1)))
        airBaseDistances = numpy.sqrt((X - X.transpose())**2 +
                                      (Y - Y.transpose())**2)

        # Fire Stations
        fireStations = self.region.getStations()[1]
        X = numpy.array([
                (fireStation.getLocation()[0] - origin[0])*40000*math.cos(
                        (origin[1] + fireStation.getLocation()[1])*math.pi/360)/360
                for fireStation in fireStations])
        Y = numpy.array([
                (origin[1] - fireStation.getLocation()[1])*40000/360
                for fireStation in fireStations])

        X = numpy.transpose(numpy.tile(X, (X.shape[0], 1)))
        Y = numpy.transpose(numpy.tile(Y, (Y.shape[0], 1)))
        landBaseDistances = numpy.sqrt((X - X.transpose())**2 +
                                       (Y - Y.transpose())**2)

        self.region.setStationDistances([airBaseDistances, landBaseDistances])

        # Stations to Nodes
        nodes = self.region.getPatches()
        # Air Bases
        X1 = numpy.array([
                (airBase.getLocation()[0] - origin[0])*40000*math.cos(
                        (origin[1] + airBase.getLocation()[1])*math.pi/360)/360
                for airBase in airBases])
        Y1 = numpy.array([
                (origin[1] - airBase.getLocation()[1])*40000/360
                for airBase in airBases])
        X2 = numpy.array([
                (node.getCentroid()[0] - origin[0])*40000*math.cos(
                        (origin[1] + node.getCentroid()[1])*math.pi/360)/360
                for node in nodes])
        Y2 = numpy.array([
                (origin[1] - node.getCentroid()[1])*40000/360
                for node in nodes])

        noBases = X1.shape
        noNodes = X2.shape

        X1 = numpy.tile(X1.transpose(), (noNodes[0], 1))
        X2 = numpy.tile(X2, (noBases[0], 1)).transpose()
        Y1 = numpy.tile(Y1.transpose(), (noNodes[0], 1))
        Y2 = numpy.tile(Y2, (noBases[0], 1)).transpose()

        airBaseNodeDistances = numpy.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)

        # Fire Stations - NOT IMPLEMENTED
        X1 = numpy.array([fireStation.getLocation()[0]
                          for fireStation in fireStations])
        Y1 = numpy.array([fireStation.getLocation()[1]
                          for fireStation in fireStations])
        X2 = numpy.array([node.getCentroid()[0] for node in nodes])
        Y2 = numpy.array([node.getCentroid()[1] for node in nodes])

        noBases = X1.shape
        noNodes = X2.shape

        X1 = numpy.tile(X1.transpose(), (noNodes[0], 1))
        X2 = numpy.tile(X2, (noBases[0], 1)).transpose()
        Y1 = numpy.tile(Y1.transpose(), (noNodes[0], 1))
        Y2 = numpy.tile(Y2, (noBases[0], 1)).transpose()

        landBaseNodeDistances = numpy.sqrt((X1 - X2)**2 + (Y1 - Y2)**2)

        self.region.setStationPatchDistances([airBaseNodeDistances,
                                              landBaseNodeDistances])

        # Initialise the fires to resources and stations
        # We don't allocate fires to resources just yet because we do not need
        # this for our simulations at this stage

        # Compute danger index
        # We set the drought factor to 10
        wind = numpy.sqrt(self.region.getWindN()**2 +
                          self.region.getWindE()**2)
        self.region.setDangerIndex(
            Simulation.computeFFDI((
                self.region.getTemperatureMax() +
                self.region.getTemperatureMin()) / 2,
                self.region.getHumidity(), wind, 10))

#        if self.nestedOptMethod > 1:
#            self.computeExpectedDamage()

    def computeExpectedDamageGrid(self):
        if self.getNestedOptMethod() <= 1:
            # We exit as we do not need this
            return

        # First determine the probability distributions for <T minute and >T
        # minute aircraft attack times (base-to-patch)
        maxCoverDists = numpy.empty(2)
        speeds = numpy.empty(2)
        # N.B. distances are actually times here
        travelProbDistsLess = []
        travelProbDistsMore = []

        for aircraft in range(2):
            speeds[aircraft] = (
                self.model.getResourceTypes()[aircraft].getSpeed())
            # Convert into hours
            maxTime = self.model.getCoverTime()/60
            maxCoverDists[aircraft] = speeds[aircraft]*maxTime

        for aircraft in range(2):
            travelProbDistsLess.append(
                numpy.histogram(self.region.getStationPatchDistances()[0]
                                [numpy.nonzero(
                                 self.region.getStationPatchDistances()
                                 [0] <= maxCoverDists[aircraft])] /
                                speeds[aircraft]))

            travelProbDistsMore.append(
                numpy.histogram(self.region.getStationPatchDistances()[0]
                                [numpy.nonzero(
                                 self.region.getStationPatchDistances()
                                 [0] > maxCoverDists[aircraft])] /
                                speeds[aircraft]))

        configurations = numpy.empty([81, 4])
        submat = numpy.array[0, 1, 2]

        for ii in range(4):
            configurations[:, ii] = (
                numpy.tile(numpy.repeat(submat, 3**(3 - ii), axis=0),
                           (3**ii, 1)).flatten())

        # Compute the probability distributions for distances for aircraft to
        # demand nodes/patches for each aircraft given the assigned region
        # Order is TankerLess \ HeliLess\ TankerMore \ HeliMore
        binCentroids = []
        binWeights = []
        # Less than specified threshold
        binCentroids.append((travelProbDistsLess[0][1]
                             [0:(len(travelProbDistsLess[0][1]) - 1)] +
                             travelProbDistsLess[0][1]
                             [1:(len(travelProbDistsLess[0][1]))])/2)

        binCentroids.append((travelProbDistsLess[1][1]
                             [0:(len(travelProbDistsLess[1][1]) - 1)] +
                             travelProbDistsLess[1][1]
                             [1:(len(travelProbDistsLess[1][1]))])/2)

        binWeights.append(travelProbDistsLess[0][0] /
                          (travelProbDistsLess[0][0].sum()))

        binWeights.append(travelProbDistsLess[1][0] /
                          (travelProbDistsLess[1][0].sum()))

        # Greater than specified threshold
        binCentroids.append((travelProbDistsMore[0][1]
                             [0:(len(travelProbDistsMore[0][1]) - 1)] +
                             travelProbDistsMore[0][1]
                             [1:(len(travelProbDistsMore[0][1]))])/2)

        binCentroids.append((travelProbDistsMore[1][1]
                             [0:(len(travelProbDistsMore[1][1]) - 1)] +
                             travelProbDistsMore[1][1]
                             [1:(len(travelProbDistsMore[1][1]))])/2)

        binWeights.append(travelProbDistsMore[0][0] /
                          (travelProbDistsMore[0][0].sum()))

        binWeights.append(travelProbDistsMore[1][0] /
                          (travelProbDistsMore[1][0].sum()))

        # Initialise expected damage matrices (for now we don't care about SD)
        # We might consider only using a fixed number of points spaced evenly
        # over the lookahead period instead of using the entire lookahead. As
        # We will only test up to a five period lookahead, which should be
        # manageable for memory and pre-computation purposes.

        # Each lookahead (as well as the configuration and vegetations) is a
        # dimension/predictor
        expectedPotentialDamage = []
        expectedExistingDamage = []

        for configuration in len(configurations):
            expPD = [None]*len(self.region.getVegetations())
            expED = [None]*len(self.region.getVegetations())

            for vegetation in range(len(self.region.getVegetations())):
                # Consider all possible lookaheads
                ffdis = len(self.region.getVegetations()[vegetation]
                            .getFFDIRange())
                lookaheadCombos = ffdis**self.lookahead

                expPDFFDIPaths = [None]*lookaheadCombos
                expEDFFDIPaths = [None]*lookaheadCombos

                for combo in range(lookaheadCombos):
                    # Get the progression of FFDIs (unravelling flattened
                    # index)
                    ffdiPath = list(numpy.unravel_index(combo, tuple([ffdis] *
                                                        self.lookahead)))

                    [expPDFFDIPaths[combo], expEDFFDIPaths[combo]] = (
                        self.computeSingleExpectedDamage(
                            configuration, vegetation, ffdiPath,
                            configurations, binCentroids, binWeights))

                expPD[vegetation] = expPDFFDIPaths
                expED[vegetation] = expEDFFDIPaths

            expectedPotentialDamage.append(expPD)
            expectedExistingDamage.append(expED)

        self.region.setExpectedPotentialDamage(expectedPotentialDamage)
        self.region.setExpectedExistingDamage(expectedExistingDamage)

    def computeExpectedDamageStatistical(self, exogenousPaths):
        if self.getNestedOptMethod() <= 1:
            # We exit as we do not need this
            return

        # First, determine which grid points to compute based on the computed
        # paths (we only need to consider the exogenous paths that will
        # actually be encountered). We fit the data points of the exogenous
        # paths to the grid, incrementing their count with each number of
        # elements encountered.

        # Upper limits for bins for FFDI.
        bins = self.region.getVegetations()[0].getFFDIRange()
        # Assume bin spacings are even for now
        binSpacing = bins[1] - bin[0]
        # We won't use the values outside the range
        computedPaths = []

        for path in self.rovPaths:
            # Find bin path to which this rov path belongs and populate
            pathTrace = numpy.divide(exogenousPaths[path, :] -
                                     bin[0],
                                     binSpacing,
                                     dtype=int)
            computedPaths.append(pathTrace)

        computedPaths = numpy.array(computedPaths)

        # Remove duplicate bin paths
        new_array = copy.copy(computedPaths)
        computedPaths = numpy.unique(new_array)

        # Compute expected damages in the same manner as the full grid approach
        maxCoverDists = numpy.empty(2)
        speeds = numpy.empty(2)
        # N.B. distances are actually times here
        travelProbDistsLess = []
        travelProbDistsMore = []

        for aircraft in range(2):
            speeds[aircraft] = (self.model.getResourceTypes()[aircraft]
                                .getSpeed())
            # Convert into hours
            maxTime = self.model.getCoverTime()/60
            maxCoverDists[aircraft] = speeds[aircraft]*maxTime

        for aircraft in range(2):
            travelProbDistsLess.append(
                numpy.histogram(self.region.getStationPatchDistances()[0]
                                [numpy.nonzero(
                                 self.region.getStationPatchDistances()
                                 [0] <= maxCoverDists[aircraft])] /
                                speeds[aircraft]))

            travelProbDistsMore.append(
                numpy.histogram(self.region.getStationPatchDistances()[0]
                                [numpy.nonzero(
                                 self.region.getStationPatchDistances()
                                 [0] > maxCoverDists[aircraft])] /
                                speeds[aircraft]))

        configurations = numpy.empty([81, 4])
        submat = numpy.array[0, 1, 2]

        for ii in range(4):
            configurations[:, ii] = (
                    numpy.tile(numpy.repeat(submat, 3**(3 - ii), axis=0),
                               (3**ii, 1)).flatten())

        # Compute the probability distributions for distances for aircraft to
        # demand nodes/patches for each aircraft given the assigned region
        # Order is TankerLess \ HeliLess\ TankerMore \ HeliMore
        binCentroids = []
        binWeights = []
        # Less than specified threshold
        binCentroids.append((travelProbDistsLess[0][1]
                            [0:(len(travelProbDistsLess[0][1])-1)] +
                            travelProbDistsLess[0][1]
                            [1:(len(travelProbDistsLess[0][1]))])/2)

        binCentroids.append((travelProbDistsLess[1][1]
                            [0:(len(travelProbDistsLess[1][1])-1)] +
                            travelProbDistsLess[1][1]
                            [1:(len(travelProbDistsLess[1][1]))])/2)

        binWeights.append(travelProbDistsLess[0][0] /
                          (travelProbDistsLess[0][0].sum()))

        binWeights.append(travelProbDistsLess[1][0] /
                          (travelProbDistsLess[1][0].sum()))

        # Greater than specified threshold
        binCentroids.append((travelProbDistsMore[0][1]
                            [0:(len(travelProbDistsMore[0][1])-1)] +
                            travelProbDistsMore[0][1]
                            [1:(len(travelProbDistsMore[0][1]))])/2)

        binCentroids.append((travelProbDistsMore[1][1]
                            [0:(len(travelProbDistsMore[1][1])-1)] +
                            travelProbDistsMore[1][1]
                            [1:(len(travelProbDistsMore[1][1]))])/2)

        binWeights.append(travelProbDistsMore[0][0] /
                          (travelProbDistsMore[0][0].sum()))

        binWeights.append(travelProbDistsMore[1][0] /
                          (travelProbDistsMore[1][0].sum()))

        # Initialise expected damage matrices (for now we don't care about SD)
        # We might consider only using a fixed number of points spaced evenly
        # over the lookahead period instead of using the entire lookahead. As
        # We will only test up to a five period lookahead, which should be
        # manageable for memory and pre-computation purposes.

        # Each lookahead (as well as the configuration and vegetations) is a
        # dimension/predictor
        expectedPotentialDamage = []
        expectedExistingDamage = []

        for configuration in len(configurations):
            expPD = [None]*len(self.region.getVegetations())
            expED = [None]*len(self.region.getVegetations())

            for vegetation in range(len(self.region.getVegetations())):
                # Consider only lookaheads that we found earlier and populate
                # those
                ffdis = len(self.region.getVegetations()[vegetation]
                            .getFFDIRange())
                lookaheadCombos = ffdis**self.lookahead
                expPDFFDIPaths = [None]*lookaheadCombos
                expEDFFDIPaths = [None]*lookaheadCombos
                shape = tuple([len(self.region.getVegetations()[vegetation]
                              .getFFDIRange())]*self.lookahead)

                for combo in range(len(computedPaths)):
                    # Compute the index of this path
                    pathIdx = numpy.ravel_multi_index(tuple(
                        computedPaths[combo]), shape)
                    [expPDFFDIPaths[pathIdx], expEDFFDIPaths[pathIdx]] = (
                        self.computeSingleExpectedDamage(
                            configuration, vegetation, computedPaths[combo],
                            configurations, binCentroids, binWeights))

            expectedPotentialDamage.append(expPD)
            expectedExistingDamage.append(expED)

        # We do not need the intervening points as they will not be encountered
        # in the simulation

        # If we have a reasonable spread, populate the remaining points through
        # weighted averaging.
#        self.region.setExpectedPotentialDamage(expectedPotentialDamage)
#        self.region.setExpectedExistingDamage(expectedExistingDamage)

    def computeSingleExpectedDamage(self, configuration, vegetation, ffdiPath,
                                    configurations, binCentroids, binWeights):
        # Computes the expected damage for a single configuration, ffdiPath,
        # vegetation combination
        # So far we only have tankers, helicopters, and trucks in the file but
        # we only use the tankers and helicopters. All three have early and
        # late success values.
        veg = self.model.getRegion().getVegetation()[vegetation]

        # Changes in weather parameters (at the start of each FFDI interval)
        occurrenceRange = (self.model.getRegion().getVegetations[veg]
                           .getOccurrence())
        occProbs = numpy.array([occurrenceRange[ffdi] for ffdi in ffdiPath])

        ffdiRange = self.model.getRegion().getVegetations[veg].getFFDIRange()
        rocMeanRange = (self.model.getRegion().getVegetations[veg]
                        .getROCA2PerHour()[0])
        rocSDRange = (self.model.getRegion().getVegetations[veg]
                      .getROCA2PerHourSD()[1])
        succTankerRangeEarly = (self.model.getRegion().getVegetations[veg]
                                .getExtinguishingSuccess()[0])
        succHeliRangeEarly = (self.model.getRegion().getVegetations[veg]
                              .getExtinguishingSuccess()[1])
        succTankerRangeLate = (self.model.getRegion().getVegetations[veg]
                               .getExtinguishingSuccess()[3])
        succHeliRangeLate = (self.model.getRegion().getVegetations[veg]
                             .getExtinguishingSuccess()[4])
        resources = (self.model.getRegion().getResourceTypes())

        # FFDI path indices
        ffdiTimings = numpy.linspace(0, self.stepSize*(len(ffdiPath) - 1),
                                     len(ffdiPath))
        ffdis = numpy.array([ffdiRange[ffdi] for ffdi in ffdiPath])

        # Rate of change per hour for each ffdi path step
        rocMean = numpy.array([rocMeanRange[ffdi] for ffdi in ffdis])
        rocSD = numpy.array([rocSDRange[ffdi] for ffdi in ffdis])

        # Extinguishing success for each aircraft type
        svse = numpy.empty(2)
        svsl = numpy.empty(2)
        svse[0] = numpy.empty([succTankerRangeEarly[ffdi] for ffdi in ffdis])
        svse[1] = numpy.empty([succHeliRangeEarly[ffdi] for ffdi in ffdis])
        svsl[0] = numpy.empty([succTankerRangeLate[ffdi] for ffdi in ffdis])
        svsl[1] = numpy.empty([succHeliRangeLate[ffdi] for ffdi in ffdis])

        # We perform Monte Carlo simulation now to determine expected damage
        # for potential and existing fires at this location given this
        # configuration, vegetation, and ffdi path
        config = configurations[configuration]

        # Before computing the expected damage, we need to randomly generate
        # the travel times for the different aircraft for each path
        randDistsTankerEarly = (numpy.random.choice(binCentroids[0],
                                                    p=binWeights[0],
                                                    size=self.mcPaths *
                                                    config[0])
                                .reshape(self.mcPaths, config[0]))

        randDistsTankerLate = (numpy.random.choice(binCentroids[1],
                                                   p=binWeights[1],
                                                   size=self.mcPaths *
                                                   config[1])
                               .reshape(self.mcPaths, config[1]))

        randDistsHeliEarly = (numpy.random.choice(binCentroids[2],
                                                  p=binWeights[2],
                                                  size=self.mcPaths *
                                                  config[2])
                              .reshape(self.mcPaths, config[2]))

        randDistsHeliLate = (numpy.random.choice(binCentroids[3],
                                                 p=binWeights[3],
                                                 size=self.mcPaths *
                                                 config[3])
                             .reshape(self.mcPaths, config[3]))

        # Random firefighting success
        # We do not know at this stage how many visits will occur so we allow
        # for the maximum number of visits for the shortest distance. We also
        # do not know a priori what the ffdi will be, so we cannot just compute
        # the true/false attack success but the random number itself.
        maxVisits = int(self.lookahead*self.timeStep/binCentroids[0][0])
        tankerEarlySuccess = (numpy.random.uniform(0,
                                                   1,
                                                   self.mcPaths *
                                                   config[0]*maxVisits)
                              .reshape((self.mcPaths,
                                        config[0],
                                        maxVisits)))

        maxVisits = int(self.lookahead*self.timeStep/config[1][0])

        tankerLateSuccess = (numpy.random.uniform(0,
                                                  1,
                                                  self.mcPaths *
                                                  config[2]*maxVisits)
                             .reshape((self.mcPaths,
                                       config[2],
                                       maxVisits)))

        maxVisits = int(self.lookahead*self.timeStep/binCentroids[2][0])

        heliEarlySuccess = (numpy.random.uniform(0,
                                                 1,
                                                 self.mcPaths *
                                                 config[1]*maxVisits)
                            .reshape((self.mcPaths,
                                      config[1],
                                      maxVisits)))

        maxVisits = int(self.lookahead*self.timeStep/binCentroids[3][0])

        heliLateSuccess = (numpy.random.uniform(0,
                                                1,
                                                self.mcPaths *
                                                config[3]*maxVisits)
                           .reshape((self.mcPaths,
                                     config[3],
                                     maxVisits)))

        # Now perform the Monte Carlo simulations using the random numbers we
        # have just generated to determine the expected damage
        damagesE = numpy.empty(self.mcPaths)
        damagesP = numpy.empty(self.mcPaths)

        for path in range(self.mcPaths):
            travelDists = []
            aircraftTypes = []

            # Early Tankers
            for tanker in range(config[0]):
                travelDists.append(randDistsTankerEarly[path][tanker] /
                                   resources[0].getSpeed())
                aircraftTypes.append(0)

            # Late Tankers
            for tanker in range(config[1]):
                travelDists.append(randDistsTankerLate[path][tanker] /
                                   resources[0].getSpeed())
                aircraftTypes.append(0)

            # Early Helicopters
            for heli in range(config[2]):
                travelDists.append(randDistsHeliEarly[path][heli] /
                                   resources[1].getSpeed())
                aircraftTypes.append(1)

            # Late Helicopters
            for heli in range(config[3]):
                travelDists.append(randDistsHeliLate[path][heli] /
                                   resources[1].getSpeed())
                aircraftTypes.append(1)

            # EXISTING FIRES ##################################################
            damagesE[path] = self.computeSinglePathE(
                path, config, ffdiTimings, rocMean, rocSD, svse, svsl,
                travelDists, aircraftTypes, tankerEarlySuccess,
                tankerLateSuccess, heliEarlySuccess, heliLateSuccess)

            # POTENTIAL FIRES #################################################
            damagesP[path] = self.computeSinglePathP(
                path, config, ffdiRange, ffdiTimings, occProbs, rocMean, rocSD,
                svse, svsl, travelDists, aircraftTypes)

        # Build a probability distribution for the expected damage (not used
        # yet)
        expectedDamageExisting = numpy.mean(damagesE)
        expectedDamagePotential = numpy.mean(damagesP)

        return [expectedDamagePotential, expectedDamageExisting]

    def computeSinglePathE(self, path, configuration, ffdiTimings, rocMean,
                           rocSD, svse, svsl, travelDists, aircraftTypes,
                           tankerEarlySuccess, tankerLateSuccess,
                           heliEarlySuccess, heliLateSuccess):
        # Figure out how many times each aircraft assigned to the fire will
        # visit over this lookahead
        visits = numpy.empty(0)
        visitingAircraft = numpy.empty(0)

        # Compute visits by all aircraft
        for aircraft in range(len(aircraftTypes)):
            noVisits = math.floor(self.lookahead*self.timeStep /
                                  (2*travelDists[aircraft]))
            visits.extend(numpy.linspace(0, noVisits-1) *
                          travelDists[aircraft]*2 +
                          travelDists[aircraft])

            visitingAircraft.extend(numpy.ones(noVisits) *
                                    aircraftTypes[aircraft])

        visitList = numpy.array([visits, visitingAircraft])

        # Sort by arrival times
        visitList = visitList[:, numpy.argsort(visitList[0])]

        # Now that we have our visits, determine the end damage caused
        extinguished = False
        elapsedTime = 0.0
        visit = 0
        totalVisits = len(visits)
        severity = 1.0
        prevIdx = 0

        while (not(extinguished) and
               elapsedTime < self.stepSize*len(ffdiTimings) and
               visit < totalVisits):

            timeInterval = visitList[0][visit] - elapsedTime
            aircraft = int(visitList[1][visit])

            # First compute the growth of the fire from the previous period up
            # to now
            # What is the current FFDI index?
            ffdiIdx = int(math.floor(visitList[0][visit]/self.timeStep))
            if ffdiIdx == prevIdx:
                severity = (severity*numpy.random.normal(rocMean[ffdiIdx],
                                                         rocSD[ffdiIdx]) *
                            timeInterval)
            else:
                # We need to grow the fire incrementally for each different
                # FFDI
                ffdiIdxes = numpy.linspace(prevIdx, ffdiIdx, ffdiIdx -
                                           prevIdx+1)
                # First index contribution
                severity = (severity*numpy.random.normal(rocMean[ffdiIdxes[0]],
                                                         rocSD[ffdiIdxes[0]]) *
                            (ffdiTimings[ffdiIdxes[1]] - elapsedTime))
                # Intervening index contributions (full interval used)
                for idx in ffdiIdxes[1:(len(ffdiIdxes)-2)]:
                    severity = (severity*numpy.random.normal(
                        rocMean[ffdiIdxes[idx]], rocSD[ffdiIdxes[idx]]) *
                        (ffdiTimings[ffdiIdxes[idx+1]] -
                         ffdiTimings[ffdiIdxes[idx]]))
                # Final index contribution
                severity = (severity*numpy.random.normal(
                    rocMean[ffdiIdxes[len(ffdiIdxes)-1]]) *
                    (visitList[0][visit] - ffdiIdxes[len(ffdiIdxes)-1]))
                prevIdx = ffdiIdx

            elapsedTime = visitList[0][visit]
            # This is extended attack, so we only use late successes, not
            # initial.
            extinguished = (True
                            if numpy.random.uniform() <
                            svsl[visitList[1][visit]][ffdiIdx]
                            else False)
            visit = visit + 1

        if not(extinguished):
            # We need to now compute the increase in damage if the fire is
            # still active
            ffdiIdx = math.floor(elapsedTime/self.timeStep)
            if ffdiIdx == (len(ffdiTimings)-1):
                severity = (severity*numpy.random.normal(
                            rocMean[ffdiIdxes[ffdiIdx]],
                            rocSD[ffdiIdxes[ffdiIdx]]) *
                            (self.lookahead*self.timeStep - elapsedTime))
            else:
                ffdiIdxes = numpy.linspace(ffdiIdx,
                                           len(ffdiTimings) - 1,
                                           len(ffdiTimings) - 1 - ffdiIdx)
                # First index contribution
                severity = (severity*numpy.random.normal(
                            rocMean[ffdiIdxes[0]],
                            rocSD[ffdiIdxes[0]]) *
                            (ffdiTimings[ffdiIdxes[1]]-elapsedTime))
                # Intervening index contributions
                for idx in ffdiIdxes[1:(len(ffdiIdxes) - 2)]:
                    severity = (severity*numpy.random.normal(
                                rocMean[ffdiIdxes[idx]],
                                rocSD[ffdiIdxes[idx]]) *
                                (ffdiTimings[ffdiIdxes[idx+1]] -
                                ffdiTimings[ffdiIdxes[idx]]))
                # Final index contribution
                severity = (severity*numpy.random.normal(
                            rocMean[ffdiIdxes[len(ffdiIdxes)-1]]) *
                            (self.lookahead*self.timeStep -
                            ffdiIdxes[len(ffdiIdxes)-1]))

        return severity

    def computeSinglePathP(self, path, configuration, ffdiRange, ffdiTimings,
                           occurrenceProbs, rocMean, rocSD, svse, svsl,
                           travelDists, aircraftTypes):
        # We assume that the patch exists in isolation

        # First sample a number of fires occurring in this patch randomly for
        # each time window
        newFiresPeriod = None*[len(ffdiTimings)]
        newFires = []
        damage = 0

        for ii in len(range(ffdiTimings)):
            newFiresPeriod[ii] = ((numpy.multiply(
                numpy.divide(1, occurrenceProbs[ii]),
                numpy.log(1 - numpy.random.uniform(0, 1, 1)))).astype(int))
            newFiresPeriod[ii] = (numpy.array(newFires[ii],
                                  numpy.random.uniform(0, 1, newFires[ii]) *
                                  self.stepSize) + ii*self.stepSize)

            # Sort the fires in this period by time of occurrence
            newFiresPeriod[ii] = (newFiresPeriod[ii]
                                  [:, numpy.argsort(newFires[ii][1, :])])

        # Put all of the fires into one contiguous array
        for ii in len(range(ffdiTimings)):
            newFires.append(newFiresPeriod[ii])

        newFires = numpy.array(newFires)

        # Now fight each of these fires using the probabilities of early and
        # late success for each type of aircraft
        resources = self.region.getResourceTypes()
        speed = numpy.empty(2)
        speed[0] = resources[0].getSpeed()
        speed[1] = resources[1].getSpeed()
        elapsedTime = 0
        start = 0
        newFireObjs = []
        assignments = []
        resourceTypes = []

        # Convert configurations to aircraft to assign (early first, then late)
        # Early tankers
        for ii in range(configuration[ii]):
            assignments.append([0])
            resourceTypes.append(0)

        # Early helicopters
        for ii in range(configuration[ii]):
            assignments.append([0])
            resourceTypes.append(1)

        # Late tankers
        for ii in range(configuration[ii]):
            assignments.append([0])
            resourceTypes.append(0)

        # Late helicopters
        for ii in range(configuration[ii]):
            assignments.append([0])
            resourceTypes.append(1)

        currLocs = travelDists
        cumHours = numpy.zeros(len(assignments[:, 1]))

        while elapsedTime < self.stepSize*len(ffdiTimings):
            timeStep = 0
            if start < (len(newFires[:, 1])-1):
                timeStep = newFires[start+1, 1] - elapsedTime
            else:
                timeStep = self.stepSize*len(ffdiTimings) - elapsedTime

            # Find aircraft to assign to this new fire ########################
            [nearestTanker, nearestHeli] = (
                Simulation().assignNearestAvailable(self, assignments,
                                                    currLocs, cumHours,
                                                    resourceTypes, [0, 0],
                                                    timeStep))
            if nearestTanker > 0:
                assignments[nearestTanker] = len(newFireObjs)
            if nearestHeli > 0:
                assignments[nearestHeli] = len(newFireObjs)

            # Append the fire to the list of active fires
            fire = Fire()
            fire.setLocation([0, 0])
            noOldFires = len(newFireObjs)
            newFireObjs.append(fire)

            # Fight this new fire (plus other new fires still active) up to the
            # start of the next fire
            for fireIdx in range(len(newFireObjs)):
                fire = newFireObjs[fireIdx]
                if fire.getSize() > 0:
                    assignedAircraft = numpy.nonzero(assignments[:, 1] ==
                                                     (fireIdx + 1))
                    currPeriod = int(elapsedTime/self.stepSize)

                    svsT = svsl[0]
                    svsH = svsl[1]

                    if fireIdx == (noOldFires - 1):
                        svsT = svse[0]
                        svsH = svse[1]

                    damage = damage + Simulation.fightFire(
                            self.model, fire, assignedAircraft, currLocs,
                            cumHours, resourceTypes, ffdiTimings[currPeriod],
                            ffdiRange, rocMean, rocSD, svsT, svsH,
                            self.stepSize)
                    # If this pass extinguished the fire, record and make
                    # aircraft available again
                    if fire.getSize() == 0:
                        assignments[numpy.nonzero(assignments[:, 1] ==
                                                  (fireIdx + 1))] = 0

            if start < (len(newFires[1, :]) - 1):
                elapsedTime = newFires[start+1, 1]
            else:
                elapsedTime = self.stepSize*len(ffdiTimings)

            start = start + 1

        return damage
