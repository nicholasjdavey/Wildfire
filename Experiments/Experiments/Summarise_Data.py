import os
import numpy

root = sys.argv[1] + "/Outputs"

dirtree = {os.listdir()[scenario]: {} for scenario in os.listdir()}

for scenario in dirtree:
    dirtree[scenario] = os.listdir(scenario)

