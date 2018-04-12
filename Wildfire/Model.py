# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 23:10:43 2017

@author: davey
"""

import numpy
#import numba
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
from WeatherGenerator import WeatherGenerator
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
        self.weatherDataFile = ""
        self.basesDataFile = ""
        self.occurrenceDataFile = ""
        self.exDamageDataFile = ""
        self.aircraftDataFiles = []
        self.landcraftDataFiles = []
        self.simulations = []
        self.resourceTypes = []
        self.variableParameters = None
        self.controls = []
        self.region = None
        self.stepSize = 0
        self.totalSteps = 0
        self.hoursPerDay = 0
        self.rovPaths = 0
        self.nestedOptMethod = 0
        self.lookahead = 0
        self.coverTime = 0
        self.mcPaths = 0

    def getROVPaths(self):
        return self.rovPaths

    def setROVPaths(self,r):
        self.rovPaths = r

    def getStepSize(self):
        return self.stepSize

    def setStepSize(self,s):
        self.stepSize = s

    def getTotalSteps(self):
        return self.totalSteps

    def setTotalSteps(self,s):
        self.totalSteps = s

    def getHoursPerDay(self):
        return self.hoursPerDay

    def setHoursPerDay(self,h):
        self.hoursPerDay = h

    def getInputFile(self):
        return self.inputfile

    def setInputFile(self,i):
        self.inputfile = i

    def getXDataFile(self):
        return self.xDataFile

    def setXDataFile(self,x):
        self.xDataFile = x

    def getYDataFile(self):
        return self.yDataFile

    def setYDataFile(self,y):
        self.xDataFile = y

    def getZDataFile(self):
        return self.zDataFile

    def setZDataFile(self,z):
        self.zDataFile = z

    def getVegetationDataFile(self):
        return self.vegetationDataFile

    def setVegetationDataFile(self,v):
        self.vegetationDataFile

    def getWeatherDataFile(self):
        return self.weatherDataFile

    def setWeatherDataFile(self,w):
        self.weatherDataFile = w

    def getBasesDataFile(self):
        return self.basesDataFile

    def setBasesDataFile(self,b):
        self.basesDataFile = b

    def getOccurrenceDataFile(self):
        return self.occurrenceDataFile

    def setOccurrenceDataFile(self,o):
        self.occurrenceDataFile = o

    def getExDamageDataFile(self):
        return self.exDamageDataFile

    def setExDamageDataFile(self,e):
        self.exDamageDataFile = e

    def getAircraftDataFiles(self):
        return self.aircraftDataFiles

    def setAircraftDataFiles(self,a):
        self.aircraftDataFiles = a

    def getLandcraftDataFiles(self):
        return self.landcraftDataFiles

    def setLandcraftDataFiles(self,l):
        self.landcraftDataFiles = l
        
    def getResourceTypes(self):
        return self.resourceTypes
        
    def setResourceTypes(self,r):
        self.resourceTypes = r

    def getSimulations(self):
        return self.simulations

    def setSimulations(self,s):
        self.simulations = s

    def getVariableParameters(self):
        return self.variableParameters

    def setVariableParameters(self,v):
        self.variableParameters = v

    def getControls(self):
        return self.controls

    def setControls(self,c):
        self.controls = c

    def getRegion(self):
        return self.region

    def setRegion(self,r):
        self.region = r
        
    def getNestedOptMethod(self):
        return self.nestedOptMethod
        
    def setNestedOptMethod(self,m):
        self.nestedOptMethod = m
        
    def getLookahead(self):
        return self.lookahead
        
    def setLookahead(self,l):
        self.lookahead = l
        
    def getCoverTime(self):
        return self.coverTime
        
    def setCoverTime(self,t):
        self.coverTime = t
        
    def getMCPaths(self):
        return self.mcPaths
        
    def setMCPaths(self,p):
        self.mcPaths = p

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

        # These controls are not used for the fourth type of linear program
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

        self.totalSteps = int(contents[75+noAircraft+noLandcraft+noControls].split(":")[1].strip())
        self.stepSize = float(contents[76+noAircraft+noLandcraft+noControls].split(":")[1].strip())
        self.hoursPerDay = float(contents[77+noAircraft+noLandcraft+noControls].split(":")[1].strip())
        self.rovPaths = int(contents[78+noAircraft+noLandcraft+noControls].split(":")[1].strip())
        self.nestedOptMethod = int(contents[79+noAircraft+noLandcraft+noControls].split(":")[1].strip())
        self.lookahead = int(contents[80+noAircraft+noLandcraft+noControls].split(":")[1].strip())
        self.coverTime = float(contents[81+noAircraft+noLandcraft+noControls].split(":")[1].strip())

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
                        sim.setModel(self)
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
        # Indices for vehicles at future calls refer to the order created here.
        # Order is A1...AI,H1...HJ,T1...TK
        # WE ONLY HAVE ONE OF EACH TYPE FOR NOW!
        resources = []
        aircraftDetails = []
        with open("../"+self.aircraftDataFiles[0]) as tf:
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
        with open("../"+self.aircraftDataFiles[1]) as hf:
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
        with open("../"+self.landcraftDataFiles[0]) as hf:
            reader = csv.reader(hf)
            truckDetails = [r[0] for r in reader]
        vehicle = Land()
        vehicle.setCrewSize(float(truckDetails[4].split(":")[1]))
        vehicle.setCapacity(float(truckDetails[5].split(":")[1]))
        vehicle.setSpeed(float(truckDetails[6].split(":")[1]))
        resources.append(vehicle)

        self.resourceTypes = resources        
        
        # Vegetation details
        vegetations = []
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
                # The first half of the rows for the success are initial successes
                # The second half are the successes for extended attack
                while test:
                    if iterator >= len(rows):
                        break
                    else:
                        if (all('' == s or s.isspace() for s in rows[iterator][3:len(rows[iterator])-3])):
                            test = False
                        else:
                            successes.append([float(col) for col in rows[iterator][3:(len(rows[iterator])-3)]])
                        iterator = iterator + 1
                vegetation.setExtinguishingSuccess(successes)
                vegetations.append(vegetation)
        
        # REGION CONFIGURATION ################################################
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
            # We only have 1 tanker and 1 helicopter at the moment
            test = True
            while test:
                if (all('' == s or s.isspace() for s in rows[iterator][1:4]) or (len(rows[iterator]) < 8)):
                    test = False
                    iterator = iterator + 2
                else:
                    airStrip = AirStrip()
                    airStrip.setLocation(numpy.array([float(rows[iterator][1]),float(rows[iterator][2])]))
                    airStrip.setMaxTankers(float(rows[iterator][5]))
                    airStrip.setMaxHelicopters(float(rows[iterator][6]))
                    noAircraft = int(rows[iterator][3])
                    aircraftList = []
                    for ii in range(noAircraft):
                        aircraft = Tanker()
                        aircraft.setFlyingHours(float(aircraftDetails[4].split(":")[1]))
                        aircraft.setMaxDailyHours(float(aircraftDetails[5].split(":")[1]))
                        aircraft.setCapacity(float(aircraftDetails[6].split(":")[1]))
                        aircraft.setSpeed(float(aircraftDetails[7].split(":")[1]))
                        aircraftList.append(aircraft)
                    airStrip.setAirTankers(aircraftList)
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
                        vehicle.setCrewSize(float(truckDetails[4].split(":")[1]))
                        vehicle.setCapacity(float(truckDetails[5].split(":")[1]))
                        vehicle.setSpeed(float(truckDetails[6].split(":")[1]))
                        vehiclesList.append(vehicle)
                    base.setLandResources(vehiclesList)
                    bases.append(base)
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
            rain = numpy.empty([len(rows)-iterator])
            precip = numpy.empty([len(rows)-iterator])
            tempMin = numpy.empty([len(rows)-iterator])
            tempMax = numpy.empty([len(rows)-iterator])
            windN = numpy.empty([len(rows)-iterator])
            windE = numpy.empty([len(rows)-iterator])
            fireSeverity = numpy.empty([len(rows)-iterator])
            fireAges = numpy.empty([len(rows)-iterator])
            ii = 0

            while test:
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
                    tempMin[ii] = float(rows[iterator][10])
                    tempMax[ii] = float(rows[iterator][11])
                    rain[ii] = float(rows[iterator][12])
                    fireSeverity[ii] = float(rows[iterator][13])
                    fireAges[ii] = float(rows[iterator][14])
                    windN[ii] = float(rows[iterator][15])
                    windE[ii] = float(rows[iterator][16])
                    ii = ii + 1
                    iterator = iterator + 1
            
            fires = []
            test = True
            while test:
                if (iterator == len(rows)):
                    test = False
                else:
                    if (all('' == s or s.isspace() for s in rows[iterator][1:4]) or (len(rows[iterator]) < 8)):
                        test = False
                    else:
                        fire = Fire()
                        fire.setLocation(numpy.array([float(rows[iterator][1]),float(rows[iterator][2])]))
                        fire.setSize(float(rows[iterator][3]))
                        fire.setStart(float(rows[iterator][4]))
                        fire.setInitialSize(float(rows[iterator][5]))
                        fires.append(fire)
                        iterator = iterator + 1

            # Save values to the region object
            self.region.setX(x[0:ii])
            self.region.setY(y[0:ii])
            self.region.setZ(z[0:ii])
            self.region.setVegetation(veg[0:ii])
            self.region.setNorth(north[0:ii])
            self.region.setSouth(south[0:ii])
            self.region.setEast(east[0:ii])
            self.region.setWest(west[0:ii])
            self.region.setHumidity(precip[0:ii])
            self.region.setRain(rain[0:ii])
            self.region.setTemperatureMin(tempMin[0:ii])
            self.region.setTemperatureMax(tempMax[0:ii])
            self.region.setPatches(patches[0:ii])
            self.region.setStations([airStrips,bases])
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
            regionConfigFile = open(root + "/Region_Configuration.csv","w+")

            regionConfigFile.close()            
        #######################################################################
            
        # WEATHER CONFIGURATION ###############################################
        weatherConfig = Path(root + "/Weather_Configuration.csv")
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
            
            occW2W = numpy.empty([len(self.region.getX()),self.totalSteps])
            occD2W = numpy.empty([len(self.region.getX()),self.totalSteps])
            
            for ii in range(len(self.region.getX())):
                for jj in range(self.totalSteps):
                    occD2W[ii,jj] = float(rows[ii+iterator][2*(jj+1)-1])
                    occW2W[ii,jj] = float(rows[ii+iterator][2*(jj+1)])
            
            wg.setWetProbT0Wet(occW2W)
            wg.setWetProbT0Dry(occD2W)         
            
            iterator = iterator + 3 + len(self.region.getX())

            # Precipitation occurrence covariances
            occCov = numpy.empty([len(self.region.getX()),len(self.region.getX())])
            
            for ii in range(len(self.region.getX())):
                for jj in range(len(self.region.getX())):
                    occCov[ii,jj] = float(rows[iterator+ii][jj+1])
            
            wg.setWetOccurrenceCovariance(occCov)            
            
            iterator = iterator + 3 + len(self.region.getX())
            
            # Precipitation amount covariances
            precipAmountCov = numpy.empty([len(self.region.getX()),len(self.region.getX())])
            
            for ii in range(len(self.region.getX())):
                for jj in range(len(self.region.getX())):
                    precipAmountCov[ii,jj] = float(rows[iterator+ii][jj+1])
                    
            wg.setPrecipAmountCovariance(precipAmountCov)
            
            iterator = iterator + 3 + len(self.region.getX())
            
            # Precipitation amount parameters
            alphas = numpy.empty([self.totalSteps])
            
            for ii in range(self.totalSteps):
                alphas[ii] = float(rows[iterator][2*ii+1])
                
            wg.setPrecipAlpha(alphas)
            
            iterator = iterator + 2

            betas = []
            betas.append(numpy.empty([self.totalSteps,len(self.region.getX())]))
            betas.append(numpy.empty([self.totalSteps,len(self.region.getX())]))
            
            for ii in range(len(self.region.getX())):
                for jj in range(self.totalSteps):
                    betas[0][jj][ii] = float(rows[iterator+ii][jj*2+1])
                    betas[1][jj][ii] = float(rows[iterator+ii][jj*2+2])

            wg.setPrecipBetas(betas)
            
            iterator = iterator + 3 + len(self.region.getX())

            remainingMean = numpy.empty(self.totalSteps)
            remainingSD = numpy.empty(self.totalSteps)
            precipCont = numpy.empty(self.totalSteps)
            
            for ii in range(self.totalSteps):
                remainingMean[ii] = float(rows[iterator][ii+1])
                remainingSD[ii] = float(rows[iterator+1][ii+1])
                precipCont[ii] = float(rows[iterator+2][ii+1])
            
            wg.setHumidityReductionMean(remainingMean)
            wg.setHumidityReductionSD(remainingSD)
            wg.setPrecipitationContributionMultiplier(precipCont)
            
            iterator = iterator + 6

            humidityCorrelations = numpy.empty([len(self.region.getX()),len(self.region.getX())])
            
            for ii in range(len(self.region.getX())):
                for jj in range(len(self.region.getX())):
                    humidityCorrelations[ii][jj] = float(rows[iterator+ii][jj+1])

            iterator = iterator + 2 + len(self.region.getX())
            
            # Temperature parameters
            # Means and Standard Deviations
            meanTempWet = numpy.empty([self.totalSteps])
            meanTempDry = numpy.empty([self.totalSteps])
            tempSDWet = numpy.empty([self.totalSteps])
            tempSDDry = numpy.empty([self.totalSteps])
            
            wg.setTempReversion(float(rows[iterator][1]))
            iterator = iterator + 2
            
            for ii in range(self.totalSteps):
                meanTempWet[ii] = float(rows[iterator][ii*4+1])
                meanTempDry[ii] = float(rows[iterator][ii*4+2])
                tempSDWet[ii] = float(rows[iterator][ii*4+3])
                tempSDDry[ii] = float(rows[iterator][ii*4+4])
                
            wg.setTempMeanWet(meanTempWet)
            wg.setTempMeanDry(meanTempDry)
            wg.setTempSDWet(tempSDWet)
            wg.setTempSDDry(tempSDDry)
            
            iterator = iterator + 2
            
            tempAlphas = []
            for ii in range(self.totalSteps):
                ta = numpy.zeros([2*len(self.region.getX()),2*len(self.region.getX())])
                tempAlphas.append(ta)

            for ii in range(len(self.region.getX())):
                for jj in range(self.totalSteps):
                    tempAlphas[jj][2*ii][ii*2] = float(rows[iterator+ii][jj*4+1])
                    tempAlphas[jj][2*ii][ii*2+1] = float(rows[iterator+ii][jj*4+2])
                    tempAlphas[jj][2*ii+1][ii*2] = float(rows[iterator+ii][jj*4+3])
                    tempAlphas[jj][2*ii+1][ii*2+1] = float(rows[iterator+ii][jj*4+4])
            
            wg.setTempA(tempAlphas)
            
            iterator = iterator + 3 + len(self.region.getX())
            
            tempBetas = []
            for ii in range(self.totalSteps):
                tb = numpy.zeros([2*len(self.region.getX()),2*len(self.region.getX())])
                
                for jj in range(2*len(self.region.getX())):
                    for kk in range(2*len(self.region.getX())):
                        tb[jj][kk] = float(rows[iterator+jj][kk+2])
                        if abs(tb[jj][kk]) > 1:
                            tb[jj][kk] = 0.0
                
                tempBetas.append(tb)
                iterator = iterator + 2*len(self.region.getX())
                
            wg.setTempB(tempBetas)
            
            iterator = iterator + 1
            
            # Wind parameters
            wg.setWindRegimes(int(rows[iterator][1]))
            
            iterator = iterator + 2
            
            regimeTransitions = numpy.zeros([wg.getWindRegimes(),wg.getWindRegimes()])
            
            for ii in range(wg.getWindRegimes()):
                for jj in range(wg.getWindRegimes()):
                    regimeTransitions[ii][jj] = float(rows[iterator+ii][2+jj])
                    
            wg.setWindRegimeTransitions(regimeTransitions)
                    
            iterator = iterator + wg.getWindRegimes() + 2
        
            # Means and Standard Deviations
            meanWindNS = numpy.empty([self.totalSteps])
            meanWindEW = numpy.empty([self.totalSteps])
            windSDNS = numpy.empty([self.totalSteps])
            windSDEW = numpy.empty([self.totalSteps])
            
            wg.setWindReversion(float(rows[iterator][1]))
            iterator = iterator + 2
            
            for ii in range(self.totalSteps):
                meanWindNS[ii] = float(rows[iterator][ii*4+1])
                meanWindEW[ii] = float(rows[iterator][ii*4+2])
                windSDNS[ii] = float(rows[iterator][ii*4+3])
                windSDEW[ii] = float(rows[iterator][ii*4+4])
                
            wg.setWindMeanNS(meanWindNS)
            wg.setWindMeanEW(meanWindEW)
            wg.setWindSDNS(windSDNS)
            wg.setWindSDEW(windSDEW)
            
            iterator = iterator + 2
            
            windAlphas = []

            for ii in range(wg.getWindRegimes()):
                wa = numpy.zeros([2*len(self.region.getX()),2*len(self.region.getX())])
                
                for jj in range(2*len(self.region.getX())):
                    for kk in range(2*len(self.region.getX())):                        
                        wa[jj][kk] = float(rows[iterator+jj][kk+2])
                        if abs(wa[jj][kk]) > 1:
                            wa[jj][kk] = 0.0
                        
                windAlphas.append(wa)
                
                iterator = iterator + 2*len(self.region.getX())
            
            wg.setWindA(windAlphas)
            
            iterator = iterator + 3
            
            windBetas = []
            
            for ii in range(wg.getWindRegimes()):
                wb = numpy.zeros([2*len(self.region.getX()),2*len(self.region.getX())])
                
                for jj in range(2*len(self.region.getX())):                
                    for kk in range(2*len(self.region.getX())):
                        wb[jj][kk] = float(rows[iterator+jj][kk+2])
                        if abs(wb[jj][kk]) > 1:
                            wb[jj][kk] = 0.0
                        
                windBetas.append(wb)
                
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
            weatherConfigFile = open(root + "/Weather_Configuration.csv","w+")

            weatherConfigFile.close()        
        #######################################################################

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
        self.region.setDangerIndex(Simulation.computeFFDI((self.region.getTemperatureMax()+self.region.getTemperatureMin())/2,self.region.getHumidity(),wind,10))
        
        if self.nestedOptMethod > 1:
            self.computeExpectedDamage()
        
    def computeExpectedDamage(self):
        # First determine the probability distributions for <T minute and >T
        # minute aircraft attack times (base-to-patch)
        maxCoverDists = numpy.empty(2)
        speeds = numpy.empty(2)
        # N.B. distances are actually times here
        travelProbDistsLess = []
        travelProbDistsMore = []
        
        for aircraft in range(2):
            speeds[aircraft] = self.model.getResourceTypes()[aircraft].getSpeed()
            maxTime = self.model.getCoverTime()/60
            maxCoverDists[aircraft] = speed*maxTime
            
        for aircraft in range(2):
            travelProbDistsLess.append(numpy.histogram(self.region.getStationPatchDistances()[aircraft][numpy.nonzero(self.region.getStationPatchDistances()[aircraft] <= maxCoverDists[aircraft])]/speeds[aircraft]))
            travelProbDistsMore.append(numpy.histogram(self.region.getStationPatchDistances()[aircraft][numpy.nonzero(self.region.getStationPatchDistances()[aircraft] > maxCoverDists[aircraft])]/speeds[aircraft]))

        configurations = numpy.empty([81,4])
        for ii in range(4):
            configurations[:,ii] = numpy.tile(numpy.repeat(submat,3**(3-ii),axis=0),(3**ii,1)).flatten()
        
        # Compute the probability distributions for distances for aircraft to
        # demand nodes/patches for each aircraft given the assigned region
        # Order is TankerLess \ HeliLess\ TankerMore \ HeliMore
        binCentroids = []
        binWeights = []
        # Less than specified threshold
        binCentroids.append((travelProbDistsLess[0][1][0:(len(travelProbDistsLess[0][1])-1)] + travelProbDistsLess[0][1][1:(len(travelProbDistsLess[0][1]))])/2)
        binCentroids.append((travelProbDistsLess[1][1][0:(len(travelProbDistsLess[1][1])-1)] + travelProbDistsLess[1][1][1:(len(travelProbDistsLess[1][1]))])/2)
        binWeights.append(travelProbDistsLess[0][0]/(travelProbDistsLess[0][0].sum()))
        binWeights.append(travelProbDistsLess[1][0]/(travelProbDistsLess[1][0].sum()))
        
        # Greater than specified threshold
        binCentroids.append((travelProbDistsMore[0][1][0:(len(travelProbDistsMore[0][1])-1)] + travelProbDistsMore[0][1][1:(len(travelProbDistsMore[0][1]))])/2)
        binCentroids.append((travelProbDistsMore[1][1][0:(len(travelProbDistsMore[1][1])-1)] + travelProbDistsMore[1][1][1:(len(travelProbDistsMore[1][1]))])/2)
        binWeights.append(travelProbDistsMore[0][0]/(travelProbDistsMore[0][0].sum()))
        binWeights.append(travelProbDistsMore[1][0]/(travelProbDistsMore[1][0].sum()))
        
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
                ffdis = len(self.region.getVegetations()[vegetation].getFFDIRange())
                lookaheadCombos = ffdis**self.lookahead
                
                expPDFFDIPaths = [None]*lookaheadCombos
                expEDFFDIPaths = [None]*lookaheadCombos
                
                for combo in range(lookaheadCombos):
                    # Get the progression of FFDIs (unravelling flattened index)
                    ffdiPath = list(numpy.unravel_index(combo,tuple([ffdis]*self.lookahead)))
                
                    [expPDFFDIPaths[combo],expEDFFDIPaths[combo]]= self.computeSingleExpectedDamage(configuration,vegetation,ffdiPath,configurations,binCentroids,binWeights)
                    
                expPD[vegetation] = expPDFFDIPaths
                expED[vegetation] = expEDFFDIPaths
            
            expectedPotentialDamage.append(expPD)
            expectedExistingDamage.append(expED)

        self.region.setExpectedPotentialDamage(expectedPotentialDamage)
        self.region.setExpectedExistingDamage(expectedExistingDamage)
        
    def computeSingleExpectedDamage(self,configuration,vegetation,ffdiPath,configurations,binCentroids,binWeights):
        # Computes the expected damage for a single configuration, ffdiPath, vegetation combination
        veg = self.model.getRegion().getVegetation()[vegetation]
        
        ffdiRange = self.model.getRegion().getVegetations[veg].getFFDIRange()
        rocMeanRange = self.model.getRegion().getVegetations[veg].getROCA2PerHour()[0]
        rocSDRange = self.model.getRegion().getVegetations[veg].getROCA2PerHour()[1]
        succTankerRange = self.model.getRegion().getVegetations[veg].getExtinguishingSuccess()[0]
        succHeliRange = self.model.getRegion().getVegetations[veg].getExtinguishingSuccess()[1]
        resources = self.model.getRegion().getResourceTypes()

        # Timing of ffdi changes (i.e. start of each FFDI interval)
        occurrenceProbs = self.model.getRegion().getVegetations[veg].getOccurrence()[ffdiPath]
        ffdiTimings = numpy.linspace(0,self.stepSize*(len(ffdiPath)-1),len(ffdiPath))
        ffdis = ffdiRange[ffdiPath]
        # Rate of change per hour for each ffdi path step
        rocMean = rocMeanRange[ffdiPath]
        rocSD = rocSDRange[ffdiPath]
        
        # Extinguishing success for each aircraft type
        svs = numpy.empty(2)
        svs[0] = succTankerRange[ffdiPath]
        svs[1] = succHeliRange[ffdiPath]
        
        # We perform Monte Carlo simulation now to determine expected damage
        # for potential and existing fires at this location given this
        # configuration, vegetation, and ffdi path
        
        # Before computing the expected damage, we need to randomly generate
        # the travel times for the different aircraft for each path
        randDistsTankerEarly = numpy.random.choice(binCentroids[0],p=binWeights[0],size=self.mcPaths*configuration[0]).reshape(mcPaths,configuration[0])
        randDistsTankerLate = numpy.random.choice(binCentroids[2],p=binWeights[2],size=self.mcPaths*configuration[2]).reshape(mcPaths,configuration[2])
        randDistsHeliEarly = numpy.random.choice(binCentroids[1],p=binWeights[1],size=self.mcPaths*configuration[1]).reshape(mcPaths,configuration[1])
        randDistsHeliLate = numpy.random.choice(binCentroids[3],p=binWeights[3],size=self.mcPaths*configuration[3]).reshape(mcPaths,configuration[3])
        
        # Random firefighting success
        # We do not know at this stage how many visits will occur so we allow
        # for the maximum number of visits for the shortest distance. We also
        # do not know a priori what the ffdi will be, so we cannot just compute
        # the true/false attack success but the random number itself.
        maxVisits = int(floor(self.lookahead*self.timeStep/binCentroids[0][0]))
        tankerEarlySuccess = numpy.random.uniform(0,1,mcPaths*configuration[0]*maxVisits).reshape((mcPaths,configuration[0],maxVisits))
        maxVisits = int(floor(self.lookahead*self.timeStep/binCentroids[2][0]))
        tankerLateSuccess = numpy.random.uniform(0,1,mcPaths*configuration[2]*maxVisits).reshape((mcPaths,configuration[2],maxVisits))
        maxVisits = int(floor(self.lookahead*self.timeStep/binCentroids[1][0]))
        heliEarlySuccess = numpy.random.uniform(0,1,mcPaths*configuration[1]*maxVisits).reshape((mcPaths,configuration[1],maxVisits))
        maxVisits = int(floor(self.lookahead*self.timeStep/binCentroids[3][0]))
        heliLateSuccess = numpy.random.uniform(0,1,mcPaths*configuration[3]*maxVisits).reshape((mcPaths,configuration[3],maxVisits))
        
        # Now perform the Monte Carlo simulations using the random numbers we
        # have just generated to determine the expected damage
        damagesE = numpy.empty(self.mcPaths)
        damagesP = numpy.empty(self.mcPaths)
        
        for path in range(self.mcPaths):
            travelDists = []
            aircraftTypes = []
            
            # Early Tankers
            for tanker in range(configurations[configuration][0]):
                travelDists.append(randDistsTankerEarly[path][tanker])
                aircraftTypes.append(0)
                
            # Late Tankers
            for tanker in range(configurations[configuration][2]):
                travelDists.append(randDistsTankerLate[path][tanker])
                aircraftTypes.append(0)
            
            # Early Helicopters
            for heli in range(configurations[configuration][1]):
                travelDists.append(randDistsHeliEarly[path][heli])
                aircraftTypes.append(1)

            # Late Helicopters
            for heli in range(configurations[configuration][3]):
                travelDists.append(randDistsHeliLate[path][heli])
                aircraftTypes.append(1)
            
            
            # EXISTING FIRES ##################################################
            damagesE[path] = self.computeSinglePathE(path,configurations[configuration],ffdiTimings,rocMean,rocSD,svs,travelDists,aircraftTypes,tankerEarlySuccess,tankerLateSuccess,heliEarlySuccess,heliLateSuccess)
            
            # POTENTIAL FIRES #################################################
            damagesP[path] = self.computeSinglePathE(path,configurations[configuration],ffdiTimings,occurrenceProbs,rocMean,rocSD,svs,travelDists,aircraftTypes,tankerEarlySuccess,tankerLateSuccess,heliEarlySuccess,heliLateSuccess)
            
        # Build a probability distribution for the expected damage (not used
        # yet)
                
        return [expectedDamagePotential,expectedDamageExisting]
    
    def computeSinglePathE(self,path,configuration,ffdiTimings,rocMean,rocSD,svs,travelDists,aircraftTypes,tankerEarlySuccess,tankerLateSuccess,heliEarlySuccess,heliLateSuccess):
        # Figure out how many times each aircraft assigned to the fire will
        # visit over this lookahead
        visits = numpy.empty(0)
        visitingAircraft = numpy.empty(0)
        # Compute visits by all aircraft
        for aircraft in range(len(aircraftTypes)):
            noVisits = math.floor(self.lookahead*self.timeStep/(2*travelDists[aircraft]))
            visits.extend(numpy.linspace(0,noVisits-1)*travelDists[aircraft]*2+travelDists[aircraft])
            visitingAircraft.extend(numpy.ones(noVisits)*aircraftTypes[aircraft])

        visitList = numpy.array([visits,visitingAircraft])
        
        # Sort by arrival times
        visitList = visitList[:,numpy.argsort(visitList[0])]
        
        # Now that we have our visits, determine the end damage caused
        extinguished = False
        elapsedTime = 0.0
        visit = 0
        totalVisits = len(visits)
        severity = 1.0
        prevIdx = 0
        damage = 0.0
        
        while not(extinguished) and elapsedTime < timeStep and visit < totalVisits:
            timeInterval = visitList[0][visit] - elapsedTime
            aircraft = int(visitList[1][visit])
            
            # First compute the growth of the fire from the previous period up
            # to now
            # What is the current FFDI index?
            ffdiIdx = int(math.floor(visitList[0][visit]/self.timeStep))
            if ffdiIdx == prevIdx:
                severity = severity*numpy.random.normal(rocMean[ffdiIdx],rocSD[ffdiIdx])*timeInterval
            else:
                # We need to grow the fire incrementally for each different
                # FFDI
                ffdiIdxes = numpy.linspace(prevIdx,ffdiIdx,ffdiIdx-prevIdx+1)
                # First index contribution
                severity = severity*numpy.random.normal(rocMean[ffdiIdxes[0]],rocSD[ffdiIdxes[0]])*(ffdiTimings[ffdiIdxes[1]]-elapsedTime)
                # Intervening index contributions (full interval used)
                for idx in ffdiIdxes[1:(len(ffdiIdxes)-2)]:
                    severity = severity*numpy.random.normal(rocMean[ffdiIdxes[idx]],rocSD[ffdiIdxes[idx]])*(ffdiTimings[ffdiIdxes[idx+1]]-ffdiTimings[ffdiIdxes[idx]])
                # Final index contribution
                severity = severity*numpy.random.normal(rocMean[ffdiIdxes[len(ffdiIdxes)-1]])*(visitList[0][visit] - ffdiIdxes[len(ffdiIdxes)-1])
                prevIdx = ffdiIdx                
                
            elapsedTime = visitList[0][visit]
            extinguished = True if numpy.random.uniform() < svs[visitList[1][visit]][ffdiIdx] else False
            visit = visit + 1
        
        if not(extinguished):
            # We need to now compute the increase in damage if the fire is still
            # active
            ffdiIdx = math.floor(elapsedTime/self.timeStep)
            if ffdiIdx == (len(ffdiTimings)-1):
                severity = severity*numpy.random.normal(rocMean[ffdiIdxes[ffdiIdx]],rocSD[ffdiIdxes[ffdiIdx]])*(self.lookahead*self.timeStep - elapsedTime)
            else:
                ffdiIdxes = numpy.linspace(ffdiIdx,len(ffdiTimings)-1,len(ffdiTimings)-1-ffdiIdx)
                # First index contribution
                severity = severity*numpy.random.normal(rocMean[ffdiIdxes[0]],rocSD[ffdiIdxes[0]])*(ffdiTimings[ffdiIdxes[1]]-elapsedTime)
                # Intervening index contributions
                for idx in ffdiIdxes[1:(len(ffdiIdxes)-2)]:
                    severity = severity*numpy.random.normal(rocMean[ffdiIdxes[idx]],rocSD[ffdiIdxes[idx]])*(ffdiTimings[ffdiIdxes[idx+1]]-ffdiTimings[ffdiIdxes[idx]])
                # Final index contribution
                severity = severity*numpy.random.normal(rocMean[ffdiIdxes[len(ffdiIdxes)-1]])*(self.lookahead*self.timeStep - ffdiIdxes[len(ffdiIdxes)-1])
        
        return severity
        
    def computeSinglePathP(self,path,configuration,ffdiTimings,occurrenceProbs,rocMean,rocSD,svs,travelDists,aircraftTypes,tankerEarlySuccess,tankerLateSuccess,heliEarlySuccess,heliLateSuccess):
        # First need to determine if a fire occurs and when
        
        clear = True
        counter = 0
        
        while clear and counter < len(ffdiTimings):
            clear = False if numpy.random.uniform() < occurrenceProbs[counter] else True