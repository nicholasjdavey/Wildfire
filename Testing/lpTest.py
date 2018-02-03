# -*- coding: utf-8 -*-

import numpy
import csv
from pulp import *
import time

# Read in LP Data
filename = sys.argv[1]

aircraft = 20
coverRadius = 60

bases = []
nodes = []
baseNodeSpacing = []

with open(filename) as fd:
    reader = csv.reader(fd)
    rows = [r for r in reader]

# First few lines are demand locations
iterator = 1
test = True

while test:
    if all('' == s or s.isspace() for s in rows[iterator][1:6]):
        test = False
        iterator = iterator + 2
    else:
        nodes.append([float(ii) for ii in rows[iterator][1:6]])
        iterator = iterator + 1

nodes = numpy.array(nodes)

# Next batch is for bases
test = True

while test:
    if all('' == s or s.isspace() for s in rows[iterator][1:4]):
        test = False
        iterator = iterator + 2
    else:
        bases.append([float(ii) for ii in rows[iterator][1:4]])
        iterator = iterator + 1

start = time.time()
bases = numpy.array(bases)

# Compute distances between bases
X = numpy.transpose(numpy.tile(bases[:,0],(bases.shape[0],1)))
Y = numpy.transpose(numpy.tile(bases[:,1],(bases.shape[0],1)))

baseSpacing = numpy.sqrt((X-X.transpose())**2+(Y-Y.transpose())**2)
#baseSpacing = numpy.reshape(numpy.sqrt((X-X.transpose())**2+(Y-Y.transpose())**2).transpose(),(bases.shape[0]**2,1))

# Compute distances between bases and nodes
X1 = numpy.tile(bases[:,0].transpose(),(nodes.shape[0],1))
X2 = numpy.tile(nodes[:,0],(bases.shape[0],1)).transpose()
Y1 = numpy.tile(bases[:,1].transpose(),(nodes.shape[0],1))
Y2 = numpy.tile(nodes[:,1],(bases.shape[0],1)).transpose()

# Create numpy array and reshape it
baseNodeSpacing = numpy.sqrt((X1-X2)**2+(Y1-Y2)**2)
#baseNodeSpacing = numpy.reshape(numpy.sqrt((X1-X2)**2+(Y1-Y2)**2),(nodes.shape[0]*bases.shape[0],1))

# State whether nodes are within cover radius from bases
baseNodeSufficient = baseNodeSpacing < coverRadius

# Set up Linear Program
relocate = LpProblem("Fire Resource Relocation", LpMinimize)

# Decision Variables
Bases = ["Base_" + str(ii+1) for ii in range(bases.shape[0])]
Binary = [["Bin_" + str(jj+1) + "_" + str(ii+1) for ii in range(bases.shape[0])] for jj in range(nodes.shape[0])]
Reloc = [["Reloc_" + str(jj+1) + "_" + str(ii+1) for ii in range(bases.shape[0])] for jj in range(bases.shape[0])]
DummySupply = ["Supply_" + str(ii+1) for ii in range(bases.shape[0])]
DummyDemand = ["Demand_" + str(ii+1) for ii in range(bases.shape[0])]

baseVars = [LpVariable(Bases[ii],lowBound=0,upBound=aircraft,cat="Integer") for ii in range(bases.shape[0])]
binaryVars = [[LpVariable(Binary[ii][jj],cat="Binary") for jj in range(bases.shape[0])] for ii in range(nodes.shape[0])]
relocVars = [[LpVariable(Reloc[ii][jj],lowBound=0,upBound=aircraft,cat="Integer") for ii in range(bases.shape[0])] for jj in range(bases.shape[0])]
supplyVars = [LpVariable(DummySupply[ii],lowBound=0,upBound=aircraft,cat="Integer") for ii in range(bases.shape[0])]
demandVars = [LpVariable(DummyDemand[ii],lowBound=0,upBound=aircraft,cat="Integer") for ii in range(bases.shape[0])]

#baseVars = LpVariable.dicts("Bases",Bases,lowBound = 0,upBound = aircraft,cat='Integer')
#binaryVars = LpVariable.dicts("BinaryVars",Binary,cat='Binary')
#relocVars = LpVariable.dicts("RelocVars",Reloc,lowBound = 0,upBound = aircraft,cat='Integer')
#supplyVars = LpVariable.dicts("DummySupply",DummySupply,lowBound = 0,upBound = aircraft,cat='Integer')
#demandVars = LpVariable.dicts("DummyDemand",DummyDemand,lowBound = 0,upBound = aircraft,cat='Integer')

# Objective Function
relocate += lpSum([(nodes[ii,3] + nodes[ii,4])*baseNodeSpacing[ii][jj]*binaryVars[ii][jj] for ii in range(nodes.shape[0]) for jj in range(bases.shape[0])] + [(baseNodeSpacing[ii][jj])*relocVars[ii][jj] for ii in range(bases.shape[0]) for jj in range(bases.shape[0])]), "Total location and relocation cost"

# Constraints
#for ii in range(nodes.shape[0]):
#    relocate += lpSum([int(baseNodeSufficient[ii][jj])*binaryVars[ii][jj] for jj in range(bases.shape[0])]) >= nodes[:,4], "Node " + str(ii) + " minimum existing demand"

for ii in range(nodes.shape[0]):
    for jj in range(bases.shape[0]):
        relocate += lpSum(binaryVars[ii][jj] - baseVars[jj]) <= 0, "Cover at node " + str(ii) + " by base " + str(jj)

relocate += lpSum([baseVars[jj] for jj in range(bases.shape[0])]) == aircraft, "Sum of aircraft is total"

for jj in range(bases.shape[0]):
    relocate += lpSum(-supplyVars[jj]+baseVars[jj]-bases[jj,2]) <= 0, "Supply of aircraft from base " + str(jj)

for jj in range(bases.shape[0]):
    relocate += lpSum(-demandVars[jj]-baseVars[jj]+bases[jj,2]) <= 0, "Demand of aircraft by base " + str(jj)

for jj in range(bases.shape[0]):
    relocate += lpSum([relocVars[ii][jj] for ii in range(bases.shape[0])]) == supplyVars[jj], "Supply flow conservation for base " + str(jj)

relocate.writeLP("Relocation.lp")
relocate.solve()
end = time.time()

print("Status: ",LpStatus[relocate.status])
print("Elapsed Time: ",end-start)
