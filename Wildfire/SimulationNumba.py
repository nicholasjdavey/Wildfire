# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:35:12 2018

@author: nicholas
"""

#import numpy
import math
from numba import jit, cuda, float32
#from pyculib import rand
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_normal_float32
from numba.cuda.random import xoroshiro128p_uniform_float32


@cuda.jit
def simulateSinglePath(paths, totalSteps, lookahead, sampleFFDIs,
                       patchVegetations, patchAreas, patchLocations,
                       baseLocations, ffdiRanges, rocA2PHMeans, rocA2PHSDs,
                       occurrence, initSizeM, initSizeSD, initSuccess,
                       resourceTypes, resourceSpeeds, configurations, configsE,
                       configsP, accumulatedDamages, accumulatedHours, fires,
                       fireSizes, fireLocations, firePatches, aircraftLocations,
                       aircraftAssignments, rng_states, controls, regressionX,
                       regressionY, start, stepSize, optimal):
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
        weightsP = cuda.local.array(shape=15000, dtype=float32)

        for ii in range(15000):
            weightsP[ii] = 0.1

        for tt in range(start, totalSteps):
            expectedFFDI = sampleFFDIs[:, tt:(totalSteps + lookahead + 1)]

            expectedDamageExisting(
                    expectedFFDI, firePatches[path][tt], fireSizes[path][tt],
                    patchVegetations, ffdiRanges, rocA2PHMeans, rocA2PHSDs,
                    occurrence, configsE, lookahead, expectedE, rng_states,
                    path, False)

            expectedDamagePotential(
                    expectedFFDI, patchVegetations, patchAreas, ffdiRanges,
                    rocA2PHMeans, occurrence, initSizeM, initSuccess, configsP,
                    tt, lookahead, expectedP)

            if optimal:
                # Ccompute the optimal control
                pass
            else:
                control = xoroshiro128p_uniform_float32(rng_states, path)

            # AssignAircraft
            # This could potentially be very slow. Just copy for now

            # SimulateNextStep
            simulateNextStep(aircraftAssignments, resourceSpeeds,
                             aircraftLocations, accumulatedHours,
                             baseLocations, fires, patchVegetations,
                             patchLocations, ffdiRanges, fireLocations,
                             firePatches, sampleFFDIs, rocA2PHMeans,
                             rocA2PHSDs, fireSizes, configsE, configsP,
                             weightsP, initSizeM, initSizeSD, initSuccess,
                             occurrence, accumulatedDamages, tt, stepSize,
                             rng_states, path)

#@jit(nopython=True, fastmath=True)
@cuda.jit(device=True)
def expectedDamageExisting(ffdi_path, fire_patches, fire_sizes,
                           patch_vegetations, ffdi_ranges, roc_a2_ph_means,
                           roc_a2_ph_sds, occurrence, configs, lookahead,
                           expected, rng_states, thread_id, random):

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
                size = growFire(ffdi, config - 1, ffdi_range, roc_a2_ph_mean,
                                roc_a2_ph_sd, size, rng_states, thread_id,
                                random)

            expected[fire*len(configs) + config] = max(
                    0, size - fire_sizes[fire])

@cuda.jit(device=True)
def expectedDamagePotential(ffdi_path, patch_vegetations, patch_areas,
                            ffdi_ranges, roc_a2_ph_means, occurrence,
                            init_size, init_success, configs, time, lookahead,
                            expected):

    for patch in range(len(patch_vegetations)):
        vegetation = int(patch_vegetations[patch])
        ffdi_range = ffdi_ranges[vegetation]
        roc_a2_ph_mean = roc_a2_ph_means[vegetation]
        occur_veg = occurrence[vegetation]
        initial_size = init_size[vegetation]
        initial_success = init_success[vegetation]

        for config in configs:
            damage = 0

            for tt in range(lookahead):
                # Only look at the expected damage of fires started at this
                # time period to the end of the horizon
                ffdi = ffdi_path[patch, tt]
                occ = max(0, interpolate1D(ffdi, ffdi_range,
                                           occur_veg[time + tt]))
                size = interpolate1D(ffdi, ffdi_range,
                                     initial_size[config])
                success = interpolate1D(ffdi, ffdi_range,
                                        initial_success[config])

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
def simulateNextStep(aircraftAssignments, aircraftSpeeds, aircraftLocations,
                     accumulatedHours, baseLocations, noFires,
                     patchVegetations, patchLocations, ffdiRanges,
                     fireLocations, firePatches, ffdis, roc_a2_ph_means,
                     roc_a2_ph_sds, fireSizes, fireConfigs, patchConfigs,
                     configWeights, initM, initSD, init_succ, occurrence,
                     accumulatedDamage, time, stepSize, rng_states, thread_id):

    """ Update aircraft locations """
    for resource in range(len(aircraftLocations)):
        baseAssignment = int(aircraftAssignments[thread_id][time][resource][0])
        fireAssignment = int(aircraftAssignments[thread_id][time][resource][1])
        speed = aircraftSpeeds[resource]
        acX = aircraftLocations[thread_id][time][resource][0]
        acY = aircraftLocations[thread_id][time][resource][1]

        if fireAssignment > 0:
            [fireX, fireY] = fireLocations[thread_id][time][fireAssignment]

            distance = math.sqrt((acX - fireX) ** 2 + (acY - fireY) ** 2)

            if distance / speed > stepSize:
                frac = stepSize * distance / speed
                aircraftLocations[thread_id][time+1][resource][0] = (
                        acX + (fireX - acX) * frac)
                aircraftLocations[thread_id][time+1][resource][1] = (
                        acY + (fireY - acY) * frac)
                accumulatedHours[thread_id][time+1][resource] += stepSize
            else:
                aircraftLocations[thread_id][time+1][resource][0] = (
                        fireLocations[thread_id][time][fireAssignment][0])
                aircraftLocations[thread_id][time+1][resource][1] = (
                        fireLocations[thread_id][time][fireAssignment][1])
                accumulatedHours[thread_id][time+1][resource] = (
                        accumulatedHours[thread_id][time][resource] + stepSize)
        else:
            [baseX, baseY] = baseLocations[baseAssignment]

            distance = math.sqrt((acX - baseX) ** 2 + (acY - baseY) ** 2)

            if distance / speed > stepSize:
                frac = stepSize * distance / speed
                aircraftLocations[thread_id][time+1][resource][0] = (
                        acX + (baseX - acX) * frac)
                aircraftLocations[thread_id][time+1][resource][1] = (
                        acY + (baseY - acY) * frac)
                accumulatedHours[thread_id][time+1][resource] += stepSize
            else:
                aircraftLocations[thread_id][time+1][resource][0] = (
                        baseLocations[baseAssignment][0])
                aircraftLocations[thread_id][time+1][resource][1] = (
                        baseLocations[baseAssignment][1])
                accumulatedHours[thread_id][time+1][resource] = (
                        accumulatedHours[thread_id][time][resource] + stepSize)

    """ Carry over the accumulated damage from the previous period """
    for patch in range(len(patchLocations)):
        accumulatedDamage[thread_id][time + 1][patch] = (
                accumulatedDamage[thread_id][time][patch])

    """ Fight existing fires, maintaining ones that are not extinguished """
    noFires[thread_id][time + 1] = 0

    for fire in range(int(noFires[thread_id][time])):
        patch = int(firePatches[thread_id][time][fire])
        vegetation = int(patchVegetations[patch])
        ffdi_range = ffdiRanges[vegetation]
        roc_a2_ph_mean = roc_a2_ph_means[vegetation]
        roc_a2_ph_sd = roc_a2_ph_sds[vegetation]

        ffdi = ffdis[patch, time]
        config = int(fireConfigs[fire])
        size = max(growFire(ffdi, config, ffdi_range, roc_a2_ph_mean,
                            roc_a2_ph_sd, fireSizes[thread_id][time][fire],
                            rng_states, thread_id, True),
                   fireSizes[thread_id][time][fire])

        count = int(noFires[thread_id][time+1])

        if size > fireSizes[thread_id][time, fire]:
            fireSizes[thread_id][time+1][count] = size
            fireLocations[thread_id][time + 1][count][0] = (
                    fireLocations[thread_id][time][fire][0])
            fireLocations[thread_id][time + 1][count][0] = (
                    fireLocations[thread_id][time][fire][1])
            firePatches[thread_id][time + 1][count] = firePatches[
                    thread_id][time][fire]
            accumulatedDamage[thread_id][time + 1][patch] += size - fireSizes[
                    thread_id][time][fire]
            count += 1
            noFires[thread_id][time + 1] += 1

    """ New fires in each patch """
    for patch in range(len(patchLocations)):
        vegetation = int(patchVegetations[patch])
        ffdi_range = ffdiRanges[vegetation]
        ffdi = ffdis[patch, time]
        initial_size_M = initM[vegetation]
        initial_size_SD = initSD[vegetation]
        initial_success = init_succ[vegetation]

        rand = xoroshiro128p_uniform_float32(rng_states, thread_id)
        scale = interpolate1D(ffdi, ffdi_range, occurrence[vegetation][time])
        # Bottom up summation
        cumPr = math.exp(-scale) * scale
        factor = 1
        newFires = 0

        while cumPr < rand:
            newFires += 1
            cumPr += math.exp(-scale) * scale ** (-(newFires + 1)) / (
                    factor * (factor + 1))

        if newFires > 0:
            sizeMean = 0.0
            sizeSD = 0.0
            initS = 0.0

            for config in range(len(patchConfigs)):
                weight = configWeights[patch*100 + config]
                sizeMean += weight * interpolate1D(
                        ffdi, ffdi_range, initial_size_M[patchConfigs[config]])
                sizeSD += weight * interpolate1D(
                        ffdi, ffdi_range, initial_size_SD[
                                patchConfigs[config]])
                initS += weight * interpolate1D(
                        ffdi, ffdi_range, initial_success[
                                patchConfigs[config]])

            for fire in range(newFires):
                success = True if initS > xoroshiro128p_uniform_float32(
                        rng_states, thread_id) else False
                size = max(0,
                           xoroshiro128p_normal_float32(rng_states, thread_id)
                           * sizeSD + sizeMean)

                accumulatedDamage[thread_id][time + 1][patch] += size

                if not success:
                    fireSizes[thread_id][time+1][fire] = size
                    fireLocations[thread_id][time + 1][newFires][0] = (
                            fireLocations[thread_id][time][fire][0])
                    fireLocations[thread_id][time + 1][newFires][1] = (
                            fireLocations[thread_id][time][fire][1])
                    firePatches[thread_id][time + 1][newFires] = firePatches[
                            thread_id][time][fire]
                    noFires[thread_id][time + 1] += 1

#@jit(parallel=True, fastmath=True)
def simulateMC(paths, sampleFFDIs, patchVegetations, patchAreas,
               patchLocations, baseLocations, resourceTypes, resourceSpeeds,
               configurations, configsE, configsP, ffdiRanges, rocA2PHMeans,
               rocA2PHSDs, occurrence, initSizeM, initSizeSD, initSuccess,
               totalSteps, lookahead, stepSize, accumulatedDamages,
               accumulatedHours, fires, fireSizes, fireLocations, firePatches,
               aircraftLocations, aircraftAssignments, controls, regressionX,
               regressionY, costs2Go, static=False):

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
            patchAreas, patchLocations, baseLocations, ffdiRanges,
            rocA2PHMeans, rocA2PHSDs, occurrence, initSizeM, initSizeSD,
            initSuccess, resourceTypes, resourceSpeeds, configurations,
            configsE, configsP, accumulatedDamages, accumulatedHours, fires,
            fireSizes, fireLocations, firePatches, aircraftLocations,
            aircraftAssignments, rng_states, controls, regressionX,
            regressionY, 0, stepSize, False)

#    for path in prange(paths):

#        simulateSinglePath(sampleFFDIs, patchVegetations, patchLocations,
#                           ffdiRanges, rocA2PHMeans, rocA2PHSDs, occurrence,
#                           resourceTypes, resourceSpeeds, configurations,
#                           configsE, configsP, totalSteps, lookahead,
#                           accumulatedDamages[path], accumulatedHours[path],
#                           fires[path], fireSizes[path], fireLocations[path],
#                           firePatches[path], aircraftLocations[path],
#                           aircraftAssignments[path], expectedE, expectedP)

def analyseMCPaths():
    pass

@cuda.jit(device=True)
def computeStatesAndControl():
    pass