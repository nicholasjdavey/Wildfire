# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:10:43 2017

@author: davey
"""

import numpy
import csv
from Control import Control
from Patch import Patch
from pathlib import Path
from Region import Region
from AirStrip import AirStrip
from Tanker import Tanker
from Heli import Heli
from Base import Base
from Simulation import Simulation
from VariableParameters import VariableParameters
from ExperimentalScenario import ExperimentalScenario

class Model():
    # Class for defining a model region

    def __init__(self):
        # Constructs an instance
        self.inputfile = ""
        self.xDataFile = ""
        self.yDataFile = ""
        self.zDataFile = ""
        self.vegetationDatafile = ""
        self.valeDataFile = ""
        self.weatherDataFile = ""
        self.basesDataFile = ""
        self.occurrenceDataFile = ""
        self.exDamageDataFile = ""
        self.aircraftDataFiles = []
        self.simulations = []
        self.variableParameters = None
        self.controls = []
        self.region = None

    def readInSourceData(self,filename):
        # First read the raw source files
        self.inputfile = filename
        inputfile = open(filename,"r")
        contents = inputfile.readlines()
        self.xDataFile = contents[13].split(":")[1].strip()
        self.yDataFile = contents[14].split(":")[1].strip()
        self.zDataFile = contents[15].split(":")[1].strip()
        self.vegetationDataFile = contents[16].split(":")[1].strip()
        self.valueDataFile = contents[17].split(":")[1].strip()
        self.weatherDataFile = contents[20].split(":")[1].strip()
        self.basesDataFile = contents[25].split(":")[1].strip()

        noAircraft = int(contents[31].split(":")[1].strip())
        aircraftData = []

        for ii in range(noAircraft):
            aircraftData.append(contents[32+ii].split(":")[1].strip())

        self.aircraftDataFile = aircraftData

        occurrenceData = contents[39].split(":")[1].strip()
        exDamageData = contents[45].split(":")[1].strip()

        noControls = int(contents[62].split(":")[1].strip())

        for ii in range(noControls):
            control = Control()
            varsStr = contents[64+ii].split(":")[1]
            lambda1 = float(varsStr.split()[0].strip())
            lambda2 = float(varsStr.split()[1].strip())
            control.setLambda1(lambda1)
            control.setLambda2(lambda2)
            self.controls.append(control)

        varParams = VariableParameters()

        varsStr = contents[67+noControls].split(":")[1].strip()
        varsStrs = varsStr.split(",")
        varsFloat = [float(varsStrs[ii]) for ii in range(len(varsStrs))]
        varParams.setSpeedMultipliers(varsFloat)

        varsStr = contents[68+noControls].split(":")[1].strip()
        varsStrs = varsStr.split(",")
        varsFloat = [float(varsStrs[ii]) for ii in range(len(varsStrs))]
        varParams.setOccurrenceProbMultipliers(varsFloat)

        varsStr = contents[69+noControls].split(":")[1].strip()
        varsStrs = varsStr.split(",")
        varsFloat = [float(varsStrs[ii]) for ii in range(len(varsStrs))]
        varParams.setDamageIntensityMultipliers(varsFloat)

        varsStr = contents[70+noControls].split(":")[1].strip()
        varsStrs = varsStr.split(",")
        varsFloat = [float(varsStrs[ii]) for ii in range(len(varsStrs))]
        varParams.setWeatherUncertMultipliers(varsFloat)

        self.variableParameters = varParams

        # Initialise the simulations
        for ii in range(len(varParams.getSpeedMultipliers())):
            for jj in range(len(varParams.getOccurrenceProbMultipliers())):
                for kk in range(len(varParams.getDamageIntensityMultipliers())):
                    for ll in range(len(varParams.getWeatherUncertMultipliers())):
                        sim = Simulation()
                        es = ExperimentalScenario()
                        es.setSpeedMultIdx(varParams.getSpeedMultipliers()[ii])
                        es.setOccProbMultIdx(varParams.getOccurrenceProbMultipliers()[jj])
                        es.setDamIntMultIdx(varParams.getDamageIntensityMultipliers()[kk])
                        es.setWeatherUncertMultIdx(varParams.getWeatherUncertMultipliers()[ll])
                        sim.setExperimentalScenario(es)
                        sim.setControls(self.controls)
                        self.simulations.append(sim)

        inputfile.close()

    def populateParameters(self):
        # First, initialise the Region object for the simulation
        self.region = Region()

        # Check if configuration files already exist
        root = "../Experiments/Experiments/" + self.inputfile.split("../Experiments/Experiments/")[1].split("/")[0]

        regionConfig = Path(root + "/Region_Configuration.csv")
        if regionConfig.is_file():
            # If exists, just call the pre-created data
            regionConfigFile = open(root + "/Region_Configuration.csv")
            print(regionConfigFile.readline())

            with open(root + "/Region_Configuration.csv") as fd:
                reader = csv.reader(fd)
                rows = [r for r in reader]

            # First few lines are demand locations
            iterator = 1
            test = True

            patches = []
            while test:
                if all('' == s or s.isspace() for s in rows[iterator][1:6]):
                    test = False
                    iterator = iterator + 2
                else:
                    patch = Patch()
                    patch.setCentroid(numpy.array([float(rows[iterator][1]),float(rows[iterator][2])]))
                    patch.setAvDanger([float(rows[iterator][3])])
                    patch.setAvSeverity([float(rows[iterator][4])])
                    patch.setArea(float(rows[iterator][6]))
                    patch.setIndices([int(rows[iterator][7].split(" ")[ii]) for ii in range(len(rows[iterator][7].split(" ")))])
                    patches.append(patch)
                    iterator = iterator + 1

            airStrips = []
            bases = []
            # Store all the air strips
            test = True
            while test:
                if (all('' == s or s.isspace() for s in rows[iterator][1:4]) or (len(rows[iterator]) < 8)):
                    test = False
                    iterator = iterator + 2
                else:
                    airStrip = AirStrip()
                    airStrip.setLocation(numpy.array([float(rows[iterator][1]),float(rows[iterator][2])]))
                    noAircraft = int(rows[iterator][3])
                    aircraftList = []
                    for ii in range(noAircraft):
                        aircraft = Tanker()
                        aircraftList.append(aircraft)
                    noHelicopters = int(rows[iterator][4])
                    heliList = []
                    for ii in range(noHelicopters):
                        heli = Heli()
                        heliList.append(heli)
                    airStrip.setHelicopters(heliList)
                    airStrip.setCapacity(float(rows[iterator][5]))
                    airStrips.append(airStrip)
                    iterator = iterator + 1

            # Store all the bases
            test = True
            while test:
                if (all('' == s or s.isspace() for s in rows[iterator][1:4]) or (len(rows[iterator]) < 8)):
                    test = False
                    iterator = iterator + 2
                else:
                    bases.append([float(ii) for ii in rows[iterator][1:4]])
                    iterator = iterator + 1

            # Raw patch data

            self.region.setPatches(patches)
            self.region.setBases(bases)
            self.region.setAirStrips(airStrips)

            regionConfigFile.close()
        else:
            # If not, build the data
            regionConfigFile = open(root + "/Region_Configuration.csv","w+")

            regionConfigFile.close()

        weatherConfig = Path(root + "/Weather_Configuration.csv")
        if weatherConfig.is_file():
            # If exists, just call the data
            weatherConfigFile = open(root + "/Weather_Configuration.csv")

            weatherConfigFile.close()
        else:
            # If not, build the data
            weatherConfigFile = open(root + "/Weather_Configuration.csv","w+")

            weatherConfigFile.close()

        # Attack success probabilities

        # Resource Data

    def computeWeatherParameters(self):
        pass

    def computeRegionParameters(self):
        pass

    def saveConfigurationData(self):
        pass
