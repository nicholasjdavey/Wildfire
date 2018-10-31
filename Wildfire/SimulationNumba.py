# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:35:12 2018

@author: nicholas
"""

import numpy
import math
from numba import jit, njit, prange

@jit(nopython=True)
def simulateSinglePath(sampleFFDIs, patchVegetations, patchLocations,
                       ffdiRanges, rocA2PHMeans, rocA2PHSDs, occurrence,
                       resourceTypes, resourceSpeeds, configurations, configsE,
                       configsP, totalSteps, lookahead, accumulatedDamages,
                       accumulatedHours, fires, fireSizes, fireLocations,
                       firePatches, aircraftLocations, aircraftAssignments,
                       expectedE, expectedP, static=False):

    for tt in range(totalSteps):
        expectedFFDI = sampleFFDIs[:, tt:(totalSteps + lookahead + 1)]

        expectedDamageExisting(
                expectedFFDI, firePatches[tt], fireSizes[tt],
                patchVegetations, ffdiRanges, rocA2PHMeans, rocA2PHSDs,
                occurrence, configsE, lookahead, expectedE)

#        expDamagePotential = expectedDamagePotential()
    #     assignAircraft
    #     simulateNextStep:

@jit(nopython=True)
def expectedDamageExisting(ffdi_path, fire_patches, fire_sizes,
                           patch_vegetations, ffdi_ranges, roc_a2_ph_means,
                           roc_a2_ph_sds, occurrence, configs, lookahead,
                           expected):

    for fire, _ in enumerate(fire_sizes):
        patch = int(fire_patches[fire])
        vegetation = int(patch_vegetations[patch])
        ffdi_range = ffdi_ranges[vegetation]
        roc_a2_ph_mean = roc_a2_ph_means[vegetation]
        roc_a2_ph_sd = roc_a2_ph_sds[vegetation]

        for config in configs:
            size = fire_sizes[fire]

            for tt in range(lookahead):
                ffdi = ffdi_path[patch, tt]
                size = 1
                size = growFire(ffdi, config - 1, ffdi_range, roc_a2_ph_mean,
                                roc_a2_ph_sd, size)

#            expected[fire, config] = max(0, size - fire_sizes[fire])

@jit(nopython=True)
def expectedDamagePotential():
    pass

@jit(nopython=True)
def growFire(ffdi, config, ffdi_range, roc_a2_ph_mean, roc_a2_ph_sd, size,
             random=False):

    gr_mean = interpolate1D(ffdi, ffdi_range, roc_a2_ph_mean[config])
    rad_curr = math.sqrt(size*10000/math.pi)

    if random:
        gr_sd = max(0, interpolate1D(ffdi, ffdi_range, roc_a2_ph_sd[config]))
        rad_new = rad_curr + max(0, numpy.random.normal(gr_mean, gr_sd))
    else:
        rad_new = rad_curr + max(0, gr_mean)

    return (math.pi * rad_new**2)/10000

@jit(nopython=True)
def interpolate1D(xval, xrange, yrange):
    """ This assumes that the xrange is already sorted and that FFDI values
    are evenly spaced in the FFDI range. Linear extrapolation used at
    boundaries """
    xspan = xrange[1] - xrange[0]
    idxMin = int(max(0, (xval - xrange[0])/xspan))

    if idxMin < (len(xrange) - 1):
        idxMax = idxMin + 1
    else:
        idxMin = len(xrange) - 2
        idxMax = len(xrange) - 1

    remainder = xval - xrange[idxMin]
    value = (yrange[idxMin] + (remainder / xspan) *
             (yrange[idxMax] - yrange[idxMin]))

    return value

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


    expectedE = numpy.zeros([100, len(configsE)])
    expectedP = numpy.zeros([100, len(configsP)])

    for path in range(paths):
        simulateSinglePath(sampleFFDIs, patchVegetations, patchLocations,
                           ffdiRanges, rocA2PHMeans, rocA2PHSDs, occurrence,
                           resourceTypes, resourceSpeeds, configurations,
                           configsE, configsP, totalSteps, lookahead,
                           accumulatedDamages[path], accumulatedHours[path],
                           fires[path], fireSizes[path], fireLocations[path],
                           firePatches[path], aircraftLocations[path],
                           aircraftAssignments[path], expectedE, expectedP)