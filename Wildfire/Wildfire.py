# -*- coding: utf-8 -*-

from Model import Model
from Region import Region
from Control import Control
from Simulation import Simulation
from JointProbArray import JointProbArray
from Patch import Patch
from Vegetation import Vegetation
from Process import Process
from Patch import Patch
from Station import Station
from Fire import Fire
from AirStrip import AirStrip
from Base import Base
from Tanker import Tanker
from Heli import Heli
from Land import Land
from Resource import Resource
import Utility
import numpy
import sys

def main():
    # Create the model
    model = Model()

    # Read in source data
    # Input file is the first argument after the program name
    model.readInSourceData(sys.argv[1])
    model.populateParameters()
    model.configureRegion()

    # Perform experiments
    for simulation in model.getSimulations():
        simulation.simulate()

    # Save results

    print("Simulations complete")

if True:
    main()
