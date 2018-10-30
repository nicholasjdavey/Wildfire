# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:35:12 2018

@author: nicholas
"""

import numpy
import math
from numba import jit

@jit
def simulateSinglePath(sampleFFDIs, patchVegetations, patchLocations,
                       ffdiRanges, rocA2PHMeans, rocA2PHSDs, occurrence,
                       resourceTypes, resourceSpeeds, configurations, configsE,
                       configsP, totalSteps, lookahead, accumulatedDamages,
                       accumulatedHours, fires, fireSizes, fireLocations,
                       firePatches, aircraftLocations, aircraftAssignments,
                       static=False):

    for tt in range(totalSteps):
        expectedFFDI = sampleFFDIs[:, tt:(totalSteps + lookahead + 1)]

        expDamageExist = expectedDamageExisting(
                expectedFFDI[tt], firePatches[tt], fireSizes[tt],
                patchVegetations, ffdiRanges, rocA2PHMeans, rocA2PHSDs,
                occurrence, configsE, lookahead)

#        expDamagePotential = expectedDamagePotential()
    #     assignAircraft
    #     simulateNextStep:

@jit
def expectedDamageExisting(ffdi_path, fire_patches, fire_sizes,
                           patch_vegetations, ffdi_ranges, roc_a2_ph_means,
                           roc_a2_ph_sds, occurrence, configs, lookahead):

    expected = numpy.array([len(fire_sizes), len(configs)])

    for fire, _ in enumerate(fire_sizes):
        patch = fire_patches[fire]
        vegetation = patch_vegetations[int(patch)]
        ffdi_range = ffdi_ranges[int(vegetation)]
#        roc_a2_ph_mean = roc_a2_ph_means[vegetation]
#        roc_a2_ph_sd = roc_a2_ph_sds[vegetation]

#        for config, _ in enumerate(configs):
#            size = fire_sizes[fire]
#
#            for tt in range(lookahead):
#                ffdi = ffdi_path[patch, tt]
#                size = growFire(ffdi, config, ffdi_range, roc_a2_ph_mean,
#                                roc_a2_ph_sd, size)
#
#            expected[fire, config] = max(0, size - fire_sizes[fire])

    return expected


@jit(nopython=True)
def expectedDamagePotential():
    pass

@jit
def growFire(ffdi, config, ffdi_range, roc_a2_ph_mean, roc_a2_ph_sd, size,
             random=False):

    gr_mean = numpy.interp(ffdi, ffdi_range, roc_a2_ph_mean)
    rad_curr = math.sqrt(size*10000/math.pi)

    if random:
        gr_sd = max(0, numpy.interp(ffdi, ffdi_range, roc_a2_ph_sd))
        rad_new = rad_curr + max(0, numpy.random.normal(gr_mean, gr_sd))
    else:
        rad_new = rad_curr + max(0, gr_mean)

    return (math.pi * rad_new**2)/10000


@jit(nopython=True)
def assignAircraft():
    pass

@jit(nopython=True)
def simulateNextStep():
    pass

@jit
def simulateMC(paths, sampleFFDIs, patchVegetations, patchLocations,
               resourceTypes, resourceSpeeds, configurations, configsE,
               configsP, ffdiRanges, rocA2PHMeans, rocA2PHSDs, occurrence,
               totalSteps, lookahead, accumulatedDamages, accumulatedHours,
               fires, fireSizes, fireLocations, firePatches, aircraftLocations,
               aircraftAssignments, static=False):

    for path in range(paths):
        simulateSinglePath(sampleFFDIs, patchVegetations, patchLocations,
                           ffdiRanges, rocA2PHMeans, rocA2PHSDs, occurrence,
                           resourceTypes, resourceSpeeds, configurations,
                           configsE, configsP, totalSteps, lookahead,
                           accumulatedDamages[path], accumulatedHours[path],
                           fires[path], fireSizes[path], fireLocations[path],
                           firePatches[path], aircraftLocations[path],
                           aircraftAssignments[path])