# -*- coding: utf-8 -*-
import sys

from Model import Model


def main():
    # Create the model
    model = Model()

    # Read in source data
    # Input file is the first argument after the program name
    if len(sys.argv) > 2:
        if sys.argv[2] == 1:
            model.generatePlots(True)

    model.readInSourceData(sys.argv[1])
    model.populateParameters()
    model.configureRegion()
#    model.computeExpectedDamageGrid()

    # Perform experiments
    for simulation in model.getSimulations():
        simulation.simulate()

    print("Simulations complete")

if True:
    main()
