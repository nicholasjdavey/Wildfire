# -*- coding: utf-8 -*-
import sys

from Model import Model


def main():
    # Create the model
    model = Model()

    # Read in source data
    # Input file is the first argument after the program name
    model.readInSourceData(sys.argv[1])
    model.populateParameters()
    model.configureRegion()
#    model.computeExpectedDamageGrid()

    # Perform experiments
    for simulation in model.getSimulations():
        simulation.simulate()

        # Save results

    print("Simulations complete")

if True:
    main()
