# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:35:12 2018

@author: nicholas
"""

import numpy
import math
from numba import jit, cuda, float32
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_normal_float32


@cuda.jit
def simulateSinglePath(paths, totalSteps, lookahead, sampleFFDIs,
                       patchVegetations, patchLocations, ffdiRanges,
                       rocA2PHMeans, rocA2PHSDs, occurrence, resourceTypes,
                       resourceSpeeds, configurations, configsE, configsP,
                       accumulatedDamages, accumulatedHours, fires, fireSizes,
                       fireLocations, firePatches, aircraftLocations,
                       aircraftAssignments, rng_states):
    # ThreadID
    tx = cuda.threadIdx.x
    # BlockID
    ty = cuda.blockIdx.x
    # BlockWidth
    bw = cuda.blockDim.x
    # GlobalID (pathNo)
    path = tx + ty * bw

    if path < paths:
        expectedE = cuda.local.array(shape=16000, dtype=float32)
        expectedP = cuda.local.array(shape=16000, dtype=float32)

        for tt in range(totalSteps):
            expectedFFDI = sampleFFDIs[:, tt:(totalSteps + lookahead + 1)]

            expectedDamageExisting(
                    expectedFFDI, firePatches[path][tt], fireSizes[path][tt],
                    patchVegetations, ffdiRanges, rocA2PHMeans, rocA2PHSDs,
                    occurrence, configsE, lookahead, expectedE, rng_states,
                    path)

            expDamagePotential = expectedDamagePotential()
#    #     assignAircraft
#    #     simulateNextStep:

#@jit(nopython=True, fastmath=True)
@cuda.jit(device=True)
def expectedDamageExisting(ffdi_path, fire_patches, fire_sizes,
                           patch_vegetations, ffdi_ranges, roc_a2_ph_means,
                           roc_a2_ph_sds, occurrence, configs, lookahead,
                           expected, rng_states, thread_id):

    for fire in range(len(fire_sizes)):
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
                                roc_a2_ph_sd, size, rng_states, thread_id,
                                True)

            expected[fire*len(configs) + config] = max(
                    0, size - fire_sizes[fire])

@cuda.jit(device=True)
def expectedDamagePotential():

    for patch in range(len(patches)):
        damage = 0
        vegetation = int(patch_vegetatios[patch])
        ffdi_range = ffdi_ranges[vegetation]
        roc_a2_ph_mean = roc_a2_ph_means[vegetation]
        roc_a2_ph_sd = roc_a2_ph_sds[vegetation]
        occur_veg = occurrence[vegetation]

        for tt in range(lookahead):
            # Only look at the expected damage of fires started at this time
            # period to the end of the horizon
            occ = max(0, interpolate1D(ffdi, ffdi_range, occur_veg[time + tt]))
            size = interpolate1D(ffdi, ffdi_range, initial_size[config])
            success = interpolate1D(ffdi, ffdi_range, initial_success[config])

            sizeInitial = size

            for t2 in range(tt, lookahead):
                ffdi = ffdi_path[patch, t2]

                gr_mean = interpolate1D(ffdi, ffdi_range,
                                        roc_a2_ph_mean[config])

                radCurr = (math.sqrt(size*10000.0/math.pi))
                radNew = radCurr + max(0, gr_mean)
                size = (math.pi * radNew**2)/10000.0

            damage += occ * size * (1 - success) + occ*sizeInitial*success

            expected[patch * len(configs) + config] = damage*patch_areas[patch]

#@jit(nopython=True, fastmath=True)
@cuda.jit(device=True)
def growFire(ffdi, config, ffdi_range, roc_a2_ph_mean, roc_a2_ph_sd, size,
             rng_states, thread_id, random):

    gr_mean = interpolate1D(ffdi, ffdi_range, roc_a2_ph_mean[config])
    rad_curr = math.sqrt(size*10000/math.pi)

    if random:
        gr_sd = max(0, interpolate1D(ffdi, ffdi_range, roc_a2_ph_sd[config]))
        rand_no = xoroshiro128p_normal_float32(rng_states, thread_id)
        rad_new = rad_curr + max(0, gr_mean + rand_no * gr_sd)
        rad_new = 0.5
    else:
        rad_new = rad_curr + max(0, gr_mean)

    return (math.pi * rad_new**2)/10000

#@jit(nopython=True)
@cuda.jit(device=True)
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

#@jit(parallel=True, fastmath=True)
def simulateMC(paths, sampleFFDIs, patchVegetations, patchLocations,
               resourceTypes, resourceSpeeds, configurations, configsE,
               configsP, ffdiRanges, rocA2PHMeans, rocA2PHSDs, occurrence,
               totalSteps, lookahead, accumulatedDamages, accumulatedHours,
               fires, fireSizes, fireLocations, firePatches, aircraftLocations,
               aircraftAssignments, static=False):

    # CUDA requirements
    threadsperblock = 32
    blockspergrid = (paths + (threadsperblock - 1)) // threadsperblock

    # Run this in chunks (i.e. multiple paths at a time)
    # Initialise all random numbers state to use for each thread
    rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid,
                                             seed=1)

    # Compute the paths in batches to preserve memory (see if we can exploit
    # both GPUs to share the computational load)
    simulateSinglePath[blockspergrid, threadsperblock](
            paths, totalSteps, lookahead, sampleFFDIs, patchVegetations,
            patchLocations, ffdiRanges, rocA2PHMeans, rocA2PHSDs, occurrence,
            resourceTypes, resourceSpeeds, configurations, configsE, configsP,
            accumulatedDamages, accumulatedHours, fires, fireSizes,
            fireLocations, firePatches, aircraftLocations, aircraftAssignments,
            rng_states)

#    for path in prange(paths):

#    simulateSinglePath[blockspergrid, threadsperblock](
#            sampleFFDIs, patchVegetations, patchLocations, ffdiRanges,
#            rocA2PHMeans, rocA2PHSDs, occurrence, resourceTypes,
#            resourceSpeeds, configurations, configsE, configsP, totalSteps,
#            lookahead, accumulatedDamages, accumulatedHours, fires, fireSizes,
#            fireLocations, firePatches, aircraftLocations, aircraftAssignments,
#            expectedE, expectedP)

#        simulateSinglePath(sampleFFDIs, patchVegetations, patchLocations,
#                           ffdiRanges, rocA2PHMeans, rocA2PHSDs, occurrence,
#                           resourceTypes, resourceSpeeds, configurations,
#                           configsE, configsP, totalSteps, lookahead,
#                           accumulatedDamages[path], accumulatedHours[path],
#                           fires[path], fireSizes[path], fireLocations[path],
#                           firePatches[path], aircraftLocations[path],
#                           aircraftAssignments[path], expectedE, expectedP)