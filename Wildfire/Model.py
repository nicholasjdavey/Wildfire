# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:10:43 2017

@author: davey
"""

import numpy
import math
import csv
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
from VariableParameters import VariableParameters
from ExperimentalScenario import ExperimentalScenario
from datetime import datetime, date, time, timedelta

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
        self.landcraftDataFiles = []
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

        self.aircraftDataFiles = aircraftData
        
        noLandcraft = int(contents[37+noAircraft].split(":")[1].strip())
        landcraftData = []
        
        for ii in range(noLandcraft):
            landcraftData.append(contents[38+noAircraft+ii].split(":")[1].strip())

        self.landcraftDataFiles = landcraftData

        self.occurrenceDataFile = contents[43+noAircraft+noLandcraft].split(":")[1].strip()
        self.exDamageDataFile = contents[49+noAircraft+noLandcraft].split(":")[1].strip()

        noControls = int(contents[66+noAircraft+noLandcraft].split(":")[1].strip())

        for ii in range(noControls):
            control = Control()
            varsStr = contents[68+noAircraft+noLandcraft+ii].split(":")[1]
            lambda1 = float(varsStr.split()[0].strip())
            lambda2 = float(varsStr.split()[1].strip())
            control.setLambda1(lambda1)
            control.setLambda2(lambda2)
            self.controls.append(control)

        varParams = VariableParameters()

        varsStr = contents[71+noAircraft+noLandcraft+noControls].split(":")[1].strip()
        varsStrs = varsStr.split(",")
        varsFloat = [float(varsStrs[ii]) for ii in range(len(varsStrs))]
        varParams.setSpeedMultipliers(varsFloat)

        varsStr = contents[72+noAircraft+noLandcraft+noControls].split(":")[1].strip()
        varsStrs = varsStr.split(",")
        varsFloat = [float(varsStrs[ii]) for ii in range(len(varsStrs))]
        varParams.setOccurrenceProbMultipliers(varsFloat)

        varsStr = contents[73+noAircraft+noLandcraft+noControls].split(":")[1].strip()
        varsStrs = varsStr.split(",")
        varsFloat = [float(varsStrs[ii]) for ii in range(len(varsStrs))]
        varParams.setDamageIntensityMultipliers(varsFloat)

        varsStr = contents[74+noAircraft+noLandcraft+noControls].split(":")[1].strip()
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
                
        # Get resource details
        # Aircraft are first, then Helicopters
        # At the moment we only have one of each
        # Tankers
        aircraftDetails = []
        with open("../"+self.aircraftDataFiles[0]) as tf:
            reader = csv.reader(tf)
            aircraftDetails = [r[0] for r in reader]
        # Helicopters
        helicopterDetails = []
        with open("../"+self.aircraftDataFiles[1]) as hf:
            reader = csv.reader(hf)
            helicopterDetails = [r[0] for r in reader]
        
        # Firetrucks
        truckDetails = []
        with open("../"+self.landcraftDataFiles[0]) as hf:
            reader = csv.reader(hf)
            truckDetails = [r for r in reader]
            
        # Vegetation details
        vegetations = []
        print("../"+self.occurrenceDataFile)
        with open("../"+self.occurrenceDataFile) as vp:
            reader = csv.reader(vp)
            rows = [r for r in reader]
            
            iterator = 2
            while iterator < len(rows):
                vegetation = Vegetation()
                vegetationNo = rows[iterator][0].strip("VEGETATION_")
                iterator = iterator + 1
                vegetation.setName(rows[iterator][2])
                iterator = iterator + 1
                vegetation.setFlammability(rows[iterator][2])
                iterator = iterator + 1
                vegetation.setFFDIRange(numpy.array([float(col) for col in rows[iterator][3:(len(rows[iterator])-3)]]))
                iterator = iterator + 1
                vegetation.setOccurrence(numpy.array([float(col) for col in rows[iterator][3:(len(rows[iterator])-3)]]))
                iterator = iterator + 1
                rocMean = numpy.array([float(col) for col in rows[iterator][3:(len(rows[iterator])-3)]])
                iterator = iterator + 1
                rocSD = numpy.array([float(col) for col in rows[iterator][3:(len(rows[iterator])-3)]])
                vegetation.setROCA2PerHour(numpy.array([rocMean,rocSD]))
                iterator = iterator + 2
                test = True
                successes = []
                while test:
                    if iterator >= len(rows):
                        break
                    else:
                        if (all('' == s or s.isspace() for s in rows[iterator][3:len(rows[iterator])-3])):
                            test = False
                        else:
                            successes.append([float(col) for col in rows[iterator][3:(len(rows[iterator])-3)]])
                        iterator = iterator + 1
        
        regionConfig = Path(root + "/Region_Configuration.csv")
        if regionConfig.is_file():
            # If exists, just call the pre-created data
            regionConfigFile = open(root + "/Region_Configuration.csv")

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
                        aircraft.setFlyingHours(float(aircraftDetails[4].split(":")[1]))
                        aircraft.setMaxDailyHours(float(aircraftDetails[5].split(":")[1]))
                        aircraft.setCapacity(float(aircraftDetails[6].split(":")[1]))
                        aircraft.setSpeed(float(aircraftDetails[7].split(":")[1]))
                        aircraftList.append(aircraft)
                    noHelicopters = int(rows[iterator][4])
                    heliList = []
                    for ii in range(noHelicopters):
                        heli = Heli()
                        heli.setFlyingHours(float(helicopterDetails[4].split(":")[1]))
                        heli.setMaxDailyHours(float(helicopterDetails[5].split(":")[1]))
                        heli.setCapacity(float(helicopterDetails[6].split(":")[1]))
                        heli.setSpeed(float(helicopterDetails[7].split(":")[1]))
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
                    base = Base()
                    base.setLocation(numpy.array([float(rows[iterator][1]),float(rows[iterator][2])]))
                    noVehicles = int(rows[iterator][3])
                    vehiclesList = []
                    for ii in range(noVehicles):
                        vehicle = Land()
                        vehiclesList.append(vehicle)
                    bases.append([float(ii) for ii in rows[iterator][1:4]])
                    iterator = iterator + 1

            # Raw patch data
            test = True
            x = numpy.empty([len(rows)-iterator])
            y = numpy.empty([len(rows)-iterator])
            z = numpy.empty([len(rows)-iterator])
            veg = numpy.empty([len(rows)-iterator])
            north = numpy.empty([len(rows)-iterator])
            south = numpy.empty([len(rows)-iterator])
            east = numpy.empty([len(rows)-iterator])
            west = numpy.empty([len(rows)-iterator])
            precip = numpy.empty([len(rows)-iterator])
            temp = numpy.empty([len(rows)-iterator])
            windN = numpy.empty([len(rows)-iterator])
            windE = numpy.empty([len(rows)-iterator])
            fires = numpy.empty([len(rows)-iterator])
            fireAges = numpy.empty([len(rows)-iterator])
            ii = 0

            fireObjects = []
            while test:
                if (iterator == len(rows)):
                    test = False
                else:
                    if (all('' == s or s.isspace() for s in rows[iterator][1:4]) or (len(rows[iterator]) < 8)):
                        test = False
                        iterator = iterator + 2
                    else:
                        x[ii] = float(rows[iterator][1])
                        y[ii] = float(rows[iterator][2])
                        z[ii] = float(rows[iterator][3])
                        veg[ii] = int(rows[iterator][4])
                        north[ii] = int(rows[iterator][5])
                        south[ii] = int(rows[iterator][6])
                        east[ii] = int(rows[iterator][7])
                        west[ii] = int(rows[iterator][8])
                        precip[ii] = float(rows[iterator][9])
                        temp[ii] = float(rows[iterator][10])
                        fires[ii] = float(rows[iterator][11])
                        fireAges[ii] = float(rows[iterator][12])
                        windN[ii] = float(rows[iterator][13])
                        windE[ii] = float(rows[iterator][14])
                        ii = ii + 1
                        iterator = iterator + 1

            # Save values to the region object
            self.region.setX(x)
            self.region.setY(y)
            self.region.setZ(z)
            self.region.setVegetation(veg)
            self.region.setNorth(north)
            self.region.setSouth(south)
            self.region.setEast(east)
            self.region.setWest(west)
            self.region.setHumidity(precip)
            self.region.setTemperature(temp)
            self.region.setPatches(patches)
            self.region.setStations([airStrips,bases])
            self.region.setFireSeverity(fires)
            self.region.setFireAge(fireAges)
            self.region.setWindN(windN)
            self.region.setWindE(windE)

            regionConfigFile.close()
        else:
            # If not, build the data
            self.computeRegionParameters()

            # Now save it
            regionConfigFile = open(root + "/Region_Configuration.csv","w+")

            regionConfigFile.close()

        weatherConfig = Path(root + "/Weather_Configuration.csv")
        if weatherConfig.is_file():
            # If exists, just call the data
            weatherConfigFile = open(root + "/Weather_Configuration.csv")

            weatherConfigFile.close()
        else:
            # If not, build the data
            self.computeWeatherParameters()

            # Now save it
            weatherConfigFile = open(root + "/Weather_Configuration.csv","w+")

            weatherConfigFile.close()

    def computeWeatherParameters(self):
        pass

    def computeRegionParameters(self):
        pass

    def configureRegion(self):
        # Compute distances between nodes and patches, etc.

        # Stations
        # Air Bases
        airBases = self.region.getStations()[0]
        X = numpy.array([airBase.getLocation()[0] for airBase in airBases])
        Y = numpy.array([airBase.getLocation()[1] for airBase in airBases])

        X = numpy.transpose(numpy.tile(X,(X.shape[0],1)))
        Y = numpy.transpose(numpy.tile(Y,(Y.shape[0],1)))
        airBaseDistances = numpy.sqrt((X-X.transpose())**2+(Y-Y.transpose())**2)

        # Fire Stations
        fireStations = self.region.getStations()[1]
        X = [fireStation.getLocation()[0] for fireStation in fireStations]
        Y = [fireStation.getLocation()[1] for fireStation in fireStations]
        X = numpy.array(X)
        Y = numpy.array(Y)

        X = numpy.transpose(numpy.tile(X,(X.shape[0],1)))
        Y = numpy.transpose(numpy.tile(Y,(Y.shape[0],1)))
        landBaseDistances = numpy.sqrt((X-X.transpose())**2+(Y-Y.transpose())**2)

        self.region.setStationDistances([airBaseDistances,landBaseDistances])

        # Stations to Nodes
        nodes = self.region.getPatches()
        # Air Bases
        X1 = numpy.array([airBase.getLocation()[0] for airBase in airBases])
        Y1 = numpy.array([airBase.getLocation()[1] for airBase in airBases])
        X2 = numpy.array([node.getCentroid()[0] for node in nodes])
        Y2 = numpy.array([node.getCentroid()[1] for node in nodes])

        noBases = X1.shape
        noNodes = X2.shape

        X1 = numpy.tile(X1.transpose(),(noNodes[0],1))
        X2 = numpy.tile(X2,(noBases[0],1)).transpose()
        Y1 = numpy.tile(Y1.transpose(),(noNodes[0],1))
        Y2 = numpy.tile(Y2,(noBases[0],1)).transpose()

        airBaseNodeDistances = numpy.sqrt((X1-X2)**2+(Y1-Y2)**2)

        # Fire Stations
        X1 = numpy.array([fireStation.getLocation()[0] for fireStation in fireStations])
        Y1 = numpy.array([fireStation.getLocation()[1] for fireStation in fireStations])
        X2 = numpy.array([node.getCentroid()[0] for node in nodes])
        Y2 = numpy.array([node.getCentroid()[1] for node in nodes])

        noBases = X1.shape
        noNodes = X2.shape

        X1 = numpy.tile(X1.transpose(),(noNodes[0],1))
        X2 = numpy.tile(X2,(noBases[0],1)).transpose()
        Y1 = numpy.tile(Y1.transpose(),(noNodes[0],1))
        Y2 = numpy.tile(Y2,(noBases[0],1)).transpose()

        landBaseNodeDistances = numpy.sqrt((X1-X2)**2+(Y1-Y2)**2)

        self.region.setStationPatchDistances([airBaseNodeDistances,landBaseNodeDistances])

        # Initialise the fires to resources and stations
        # We don't allocate fires to resources just yet because we do not need
        # this for our simulations at this stage

        # Compute danger index
        # We set the drought factor to 10
        wind = numpy.sqrt(self.region.getWindN()**2+self.region.getWindE()**2)
        self.region.setDangerIndex(Simulation.computeFFDI(self.region.getTemperature(),self.region.getHumidity(),wind,10))
        
