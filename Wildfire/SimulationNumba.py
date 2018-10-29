# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:35:12 2018

@author: nicholas
"""

import numpy
import numba
from numba import jit

@jit(nopython=True)
def simulateSinglePath(static=False):
    # REQUIRED:
    # sampleFFDI
    # initAssignments
    # initFires
    # timeSteps
    # lookahead
    # configurations
    # patchLocations
    # patchVegetations
    # vegetations:

    # OUTPUTS:
    # finalDamageMaps
    # expectedDamages
    # realisedAssignments
    # realisedFires
    # realisedFFDIs
    # aircraftHours

    # noFires
    # accumulatedDamage
    # accumulatedHours
    # patchConfigs
    # fireConfigs

    # STEPS:
    for tt in range(timeSteps):
        expectedFFDI = sampleFFDIs[:, tt:(totalSteps + lookahead + 1)]

        expDamageExist = expectedDamageExisting(expectedFFDI, firePatches,
                                                fireSizes, patchVegetations,
                                                ffdiRanges, rocA2PHMeans,
                                                rocA2PHSDs)

        expDamagePotential = expectedDamagePotential()
    #     assignAircraft
    #     simulateNextStep:

@jit(nopython=True)
def expectedDamageExisting(ffdi_path, fire_patches, fire_sizes,
                           patch_vegetations, ffdi_ranges, roc_a2_ph_means,
                           roc_a2_ph_sds):

    expected = numpy.array([len(fire_sizes), len(configs)])

    for fire, _ in enumerate(fire_sizes):
        patch = fire_patches[fire]
        vegetation = patch_vegetations[patch]
        ffdi_range = ffdi_ranges[vegetation]
        roc_a2_ph_mean = roc_a2_ph_means[vegetation]
        roc_a2_ph_sd = roc_a2_ph_sds[vegetation]
        fire_sizes[fire]

        for config, _ in enumerate(configs):
            for tt in range(look):
                ffdi = ffdi_path[patch, tt]
                size_new = growFire(ffdi, config, ffdi_range, roc_a2_ph_mean,
                                    roc_a2_ph_sd, size)

            expected[fire, config] = max(0, size_new - size)

    return expected


@jit(nopython=True)
def expectedDamagePotential():


@jit(nopython=True)
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

@jit(nopython=True)
def simulateNextStep():


@jit(nopython=True)
def simulateMC(static=False):
