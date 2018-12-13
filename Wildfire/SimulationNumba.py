# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:35:12 2018

@author: nicholas
"""

import sys
import numpy
import math
from numba import cuda, float32, float64, int32
from numba.types import b1
#from pyculib import rand
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_normal_float32
from numba.cuda.random import xoroshiro128p_uniform_float32
import pyqt_fit.nonparam_regression as smooth
from pyqt_fit import npr_methods

""" GLOBAL VARIABLES USED FOR ARRAY DIMENSIONS IN CUDA """
""" Remember to modify the values in the wrapper functions before using them in
the kernels """
global noBases
global noPatches
global noAircraft
global noFiresMax
global noConfigs
global noConfE
global noConfP
global method
global noControls
global epsilon


@cuda.jit
def simulateSinglePath(paths, totalSteps, lookahead, sampleFFDIs, expFiresComp,
                       lambdas, patchVegetations, patchAreas, patchLocations,
                       baseLocations, tankerDists, heliDists, ffdiRanges,
                       rocA2PHMeans, rocA2PHSDs, occurrence, initSizeM,
                       initSizeSD, initSuccess, extSuccess, resourceTypes,
                       resourceSpeeds, maxHours, configurations, configsE,
                       configsP, baseConfigsMax, fireConfigsMax, thresholds,
                       accumulatedDamages, accumulatedHours, fires, fireSizes,
                       fireLocations, firePatches, aircraftLocations,
                       aircraftAssignments, rng_states, states, controls,
                       regressionX, regressionY, costs2Go, start, stepSize,
                       method, optimal, static, expectedTemp):

    # Batched values start at zero here, but at different locations in the host
    # arrays. This is because only a portion of the host array is transferred.

    global noBases
    global noPatches
    global noAircraft
    global noFiresMax
    global noConfigs
    global noConfE
    global noConfP
    global noControls

    # ThreadID
    tx = cuda.threadIdx.x
    # BlockID
    ty = cuda.blockIdx.x
    # BlockWidth
    bw = cuda.blockDim.x
    # GlobalID (pathNo)
    path = tx + ty * bw

    if path < paths:
        # Modify to take variables as assignments. Use tuples to define shape
        # e.g. cuda.local.array((16, 1000), dtype=float32)
        expectedE = cuda.local.array((noFiresMax, noConfE), dtype=float32)
        expectedP = cuda.local.array((noPatches, noConfP), dtype=float32)
        weightsP = cuda.local.array((noPatches, noConfP), dtype=float32)
        selectedE = cuda.local.array((noFiresMax), dtype=int32)

        for tt in range(start, totalSteps):
            expectedFFDI = sampleFFDIs[:, tt:(totalSteps + lookahead + 1)]

            expectedDamageExisting(
                    expectedFFDI, firePatches[path][tt - start], fires[path][tt
                    - start], fireSizes[path][tt - start], patchVegetations,
                    ffdiRanges, rocA2PHMeans, rocA2PHSDs, extSuccess, configsE,
                    lookahead, expectedE, rng_states, path, False)

            expectedDamagePotential(
                    expectedFFDI, patchVegetations, patchAreas, ffdiRanges,
                    rocA2PHMeans, rocA2PHSDs, occurrence[:][start:(totalSteps
                    + lookahead + 1)], initSizeM, initSizeSD, initSuccess,
                    extSuccess, configsP, tt - start, lookahead, expectedP,
                    rng_states, path, False)

#            if path == 0:
#                if tt == start:
#                    for config in range(len(configsE)):
#                        for fire in range(fires[path][tt]):
#                            expectedTemp[1, fire, config] = expectedE[fire][config]

            if optimal:
                # Compute the optimal control using regressions. Need to
                # compute the expected damage for fires and patches under
                # each control. This could be time-consuming.
                bestControl = 0
                bestC2G = math.inf

                for control in range(noControls):
                    """ Need to determine the expected cost to go for each
                    control in order to determine the best one to pick. This
                    requires computing the state for each control using the
                    assignment heuristic and then the regressions. """

                    assignAircraft(aircraftAssignments, resourceSpeeds,
                                   resourceTypes, maxHours, aircraftLocations,
                                   accumulatedHours, baseLocations,
                                   tankerDists, heliDists, fires, fireSizes,
                                   fireLocations, ffdiRanges, configurations,
                                   configsE, configsP, baseConfigsMax,
                                   fireConfigsMax, thresholds, expFiresComp,
                                   expectedE, expectedP, selectedE, weightsP,
                                   tt - start, stepSize, lookahead, path,
                                   control, lambdas, method, expectedTemp)

                    saveState(aircraftAssignments, resourceTypes,
                              resourceSpeeds, maxHours, aircraftLocations,
                              accumulatedHours, patchLocations, fires,
                              fireSizes, fireLocations, expectedE, expectedP,
                              states, configurations, selectedE, weightsP,
                              tt - start, lookahead, path)

                    """ Get the expected cost 2 go for this control at this
                    time for the prevailing state """
                    currC2G = interpolateCost2Go(states, regressionX,
                                                 regressionY, tt - start, path,
                                                 control)

                    if currC2G < bestC2G:
                        bestC2G = currC2G
                        bestControl = control

                controls[path][tt] = bestControl

            else:
                if static < 0:
                    bestControl = int(6*xoroshiro128p_uniform_float32(
                        rng_states, path))
                    controls[path][tt] = bestControl

                else:
                    bestControl = static
                    controls[path][tt] = static

            # AssignAircraft
            # This could potentially be very slow. Just use a naive assignment
            # for now
            assignAircraft(aircraftAssignments, resourceSpeeds, resourceTypes,
                           maxHours, aircraftLocations, accumulatedHours,
                           baseLocations, tankerDists, heliDists, fires,
                           fireSizes, fireLocations, ffdiRanges,
                           configurations, configsE, configsP, baseConfigsMax,
                           fireConfigsMax, thresholds, expFiresComp, expectedE,
                           expectedP, selectedE, weightsP, tt - start,
                           stepSize, lookahead, path, bestControl, lambdas,
                           method, expectedTemp)

#            saveState(aircraftAssignments, resourceTypes, resourceSpeeds,
#                      maxHours, aircraftLocations, accumulatedHours,
#                      patchLocations, fires, fireSizes, fireLocations,
#                      expectedE, expectedP, states, configurations, selectedE,
#                      weightsP, tt - start, lookahead, path)
#
#            # SimulateNextStep
#            simulateNextStep(aircraftAssignments, resourceTypes,
#                             resourceSpeeds, aircraftLocations,
#                             accumulatedHours, baseLocations, fires,
#                             patchVegetations, patchLocations, ffdiRanges,
#                             fireLocations, firePatches, expectedFFDI,
#                             rocA2PHMeans, rocA2PHSDs, fireSizes, configsE,
#                             configsP, selectedE, weightsP, initSizeM,
#                             initSizeSD, initSuccess, extSuccess, occurrence,
#                             accumulatedDamages, tt - start, stepSize,
#                             rng_states, path)

#@jit(nopython=True, fastmath=True)
@cuda.jit(device=True)
def expectedDamageExisting(ffdi_path, fire_patches, no_fires, fire_sizes,
                           patch_vegetations, ffdi_ranges, roc_a2_ph_means,
                           roc_a2_ph_sds, ext_success, configs, lookahead,
                           expected, rng_states, thread_id, random):

    for fire in range(no_fires):
        patch = int(fire_patches[fire])
        vegetation = int(patch_vegetations[patch])
        ffdi_range = ffdi_ranges[vegetation]
        roc_a2_ph_mean = roc_a2_ph_means[vegetation]
        roc_a2_ph_sd = roc_a2_ph_sds[vegetation]
        ext_succ = ext_success[vegetation]

        for c in range(len(configs)):
            size = fire_sizes[fire]
            sizeTemp = size
            config = configs[c]

            for tt in range(lookahead):
                ffdi = ffdi_path[patch, tt]
                success = interpolate1D(ffdi, ffdi_range, ext_succ[config - 1])
                growth = growFire(ffdi, config - 1, ffdi_range, roc_a2_ph_mean,
                                  roc_a2_ph_sd, sizeTemp, rng_states,
                                  thread_id, random) - sizeTemp

                sizeTemp += growth
                size += growth * (1 - success) ** (tt + 1)

            expected[fire, c] = size - fire_sizes[fire]


@cuda.jit(device=True)
def expectedDamagePotential(ffdi_path, patch_vegetations, patch_areas,
                            ffdi_ranges, roc_a2_ph_means, roc_a2_ph_sds,
                            occurrence, init_size_m, init_size_sd,
                            init_success, ext_success, configs, time,
                            lookahead, expected, rng_states, thread_id,
                            random):

    for patch in range(len(patch_vegetations)):
        vegetation = int(patch_vegetations[patch])
        ffdi_range = ffdi_ranges[vegetation]
        roc_a2_ph_mean = roc_a2_ph_means[vegetation]
        roc_a2_ph_sd = roc_a2_ph_sds[vegetation]
        occur_veg = occurrence[vegetation]
        initial_m = init_size_m[vegetation]
        initial_sd = init_size_sd[vegetation]
        initial_success = init_success[vegetation]
        exist_success = ext_success[vegetation]

        for c in range(len(configs)):
            damage = 0
            config = configs[c]

            for tt in range(lookahead):
                # Only look at the expected damage of fires started at this
                # time period to the end of the horizon
                ffdi = ffdi_path[patch, tt]
                occ = max(0, interpolate1D(ffdi, ffdi_range,
                                           occur_veg[time + tt]))
                sizeM = interpolate1D(ffdi, ffdi_range,
                                     initial_m[config - 1])
                sizeSD = interpolate1D(ffdi, ffdi_range,
                                     initial_sd[config - 1])
                success = interpolate1D(ffdi, ffdi_range,
                                        initial_success[config - 1])
                sizeTemp = math.exp(sizeM + sizeSD ** 2 /2)

                # Add the damage caused by fires formed this period but
                # extinguished immediately
                damage += occ * sizeTemp * success

                # Now add the damage caused by fires that escape initial
                # attack success
                for t2 in range(tt, lookahead):
                    ffdi = ffdi_path[patch, t2]
                    success2 = interpolate1D(ffdi, ffdi_range,
                                             exist_success[config - 1])
                    growth = growFire(ffdi, config - 1, ffdi_range,
                                      roc_a2_ph_mean, roc_a2_ph_sd, sizeTemp,
                                      rng_states, thread_id, random) - sizeTemp

                    sizeTemp += growth

                    damage += occ * growth * (1 - success2) ** (t2 - tt + 1)

            expected[patch, c] = damage * patch_areas[patch]


#@jit(nopython=True, fastmath=True)
@cuda.jit(device=True)
def growFire(ffdi, config, ffdi_range, roc_a2_ph_mean, roc_a2_ph_sd, size,
             rng_states, thread_id, random):

    gr_mean = interpolate1D(ffdi, ffdi_range, roc_a2_ph_mean[config])
    gr_sd = max(0, interpolate1D(ffdi, ffdi_range, roc_a2_ph_sd[config]))
    rad_curr = math.sqrt(size*10000/math.pi)

    if random:
        rand_no = xoroshiro128p_normal_float32(rng_states, thread_id)
        rad_new = rad_curr + math.exp(gr_mean + rand_no * gr_sd)
    else:
        rad_new = rad_curr + math.exp(gr_mean + gr_sd ** 2 /2)

    return (math.pi * rad_new**2)/10000

#@jit(nopython=True)
@cuda.jit(device=True)
def interpolate1D(xval, xrange, yrange):
    """ This assumes that the xrange is already sorted and that FFDI values
    are evenly spaced in the FFDI range. Linear extrapolation used at
    boundaries """
    xspan = xrange[1] - xrange[0]
    idxMin = int(max(0, (xval - xrange[0]) / xspan))

    if idxMin < (len(xrange) - 1):
        idxMax = idxMin + 1
    else:
        idxMin = len(xrange) - 2
        idxMax = len(xrange) - 1

    remainder = xval - xrange[idxMin]
    value = (yrange[idxMin] + (remainder / xspan) *
             (yrange[idxMax] - yrange[idxMin]))

    return value

@cuda.jit(device=True)
def interpolateCost2Go(state, regressionX, regressionY, time, path, control):

    global epsilon

    """ Get the global upper and lower bounds for each dimension """
    lower = cuda.local.array(3, dtype=float32)
    upper = cuda.local.array(3, dtype=float32)
    coeffs = cuda.local.array(8, dtype=float32)
    lowerInd = cuda.local.array(3, dtype=int32)

    # Indices for each state dimension value
    for dim in range(3):
        lower[dim] = regressionX[time][control][dim][0]
        upper[dim] = regressionX[time][control][dim][-1]
        lowerInd[dim] = int(len(regressionX[time][control][dim]) *
                state[path][time][dim] / (upper[dim] - lower[dim]))

        if lowerInd[dim] < 0:
            lowerInd[dim] = 0
        elif lowerInd[dim] >= len(regressionX[time][control][dim]):
            lowerInd[dim] = len(regressionX[time][control][dim]) - 2

    # Now that we have all the index requirements, let's interpolate
    # Uppermost dimension X value
    x0 = regressionX[time][control][0][lowerInd[0]]
    x1 = regressionX[time][control][0][lowerInd[0] + 1]

    if abs(x1 - x0) < epsilon:
        xd = 0.0
    else:
        xd = (state[path][time][0] - x0) / (x1 - x0)

    # Assign y values to the coefficient matrix
    coeffs[0] = (
        (1 - xd)*regressionY[time][control][lowerInd[0]][lowerInd[1]][lowerInd[2]]
        + xd*regressionY[time][control][lowerInd[0]+1][lowerInd[1]][lowerInd[2]])
    coeffs[1] = (
        (1 - xd)*regressionY[time][control][lowerInd[0]][lowerInd[1]][lowerInd[2]+1]
        + xd*regressionY[time][control][lowerInd[0]+1][lowerInd[1]][lowerInd[2]+1])
    coeffs[2] = (
        (1 - xd)*regressionY[time][control][lowerInd[0]][lowerInd[1]+1][lowerInd[2]]
        + xd*regressionY[time][control][lowerInd[0]+1][lowerInd[1]+1][lowerInd[2]])
    coeffs[3] = (
        (1 - xd)*regressionY[time][control][lowerInd[0]][lowerInd[1]][lowerInd[2]]
        + xd*regressionY[time][control][lowerInd[0]+1][lowerInd[1]+1][lowerInd[2]+1])

    # Now progress down
    x0 = regressionX[time][control][1][lowerInd[0]]
    x1 = regressionX[time][control][1][lowerInd[0] + 1]

    if abs(x1 - x0) < epsilon:
        xd = 0.0
    else:
        xd = (state[path][time][1] - x0) / (x1 - x0)

    coeffs[0] = (1 - xd)*coeffs[0] + xd*coeffs[2]
    coeffs[1] = (1 - xd)*coeffs[1] + xd*coeffs[3]

    # Final dimension
    x0 = regressionX[time][control][2][lowerInd[0]]
    x1 = regressionX[time][control][2][lowerInd[0] + 1]

    if abs(x1 - x0) < epsilon:
        xd = 0.0
    else:
        xd = (state[path][time][2] - x0) / (x1 - x0)

    return (1 - xd)*coeffs[0] + xd*coeffs[1]


@cuda.jit(device=True)
def getAllConfigsSorted(configurations, configs, baseConfigsPossible,
                        expectedP, numbers):

    tempSortList = cuda.local.array(noConfigs, dtype=int32)

    # Zero the output list
    for config in range(len(baseConfigsPossible)):
        baseConfigsPossible[config] = 0

    # Collect
    found = 0
    for config in configs:
        viable = True

        for c in range(4):
            if configurations[config][c] > numbers[c]:
                viable = False
                break

        if viable:
            tempSortList[found] = config
            found += 1

    # Simple selection sort for now as the list will likely be very small
    for i in range(found):
        iMin = i

        for j in range(i, found):
            if expectedP[tempSortList[j]] < expectedP[tempSortList[iMin]]:
                iMin = j

        baseConfigsPossible[i] = tempSortList[iMin]
        tempSortList[iMin] = tempSortList[i]


@cuda.jit(device=True)
def maxComponentNo():
    pass


@cuda.jit(device=True)
def configWeight():
    pass


@cuda.jit(device=True)
def simulateNextStep(aircraftAssignments, aircraftTypes, aircraftSpeeds,
                     aircraftLocations, accumulatedHours, baseLocations,
                     noFires, patchVegetations, patchLocations, ffdiRanges,
                     fireLocations, firePatches, ffdis, roc_a2_ph_means,
                     roc_a2_ph_sds, fireSizes, fireConfigs, patchConfigs,
                     selectedE, configWeights, initM, initSD, init_succ,
                     occurrence, accumulatedDamage, time, stepSize, rng_states,
                     thread_id):

    """ Update aircraft locations """
    for resource in range(len(aircraftLocations)):
        baseAssignment = int(aircraftAssignments[thread_id][time][resource][0])
        fireAssignment = int(aircraftAssignments[thread_id][time][resource][1])
        speed = aircraftSpeeds[resource]
        acX = aircraftLocations[thread_id][time][resource][0]
        acY = aircraftLocations[thread_id][time][resource][1]

        if fireAssignment > 0:
            [fireX, fireY] = fireLocations[thread_id][time][fireAssignment]

            distance = geoDist(aircraftLocations[thread_id][time][resource],
                               fireLocations[thread_id][time][fireAssignment])

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

            distance = geoDist(aircraftLocations[thread_id][time][resource],
                               baseLocations[baseAssignment])

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

        ffdi = ffdis[patch, 0]
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
        ffdi = ffdis[patch, 0]
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
                weight = configWeights[patch, config]

                if weight > 0.0:
                    sizeMean += weight * interpolate1D(
                            ffdi, ffdi_range,
                            initial_size_M[patchConfigs[config]])
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
                            patchLocations[patch][0])
                    fireLocations[thread_id][time + 1][newFires][1] = (
                            patchLocations[patch][1])
                    firePatches[thread_id][time + 1][newFires] = firePatches[
                            thread_id][time][fire]
                    noFires[thread_id][time + 1] += 1

@cuda.jit(device=True)
def assignAircraft(aircraftAssignments, resourceSpeeds, resourceTypes,
                   maxHours, aircraftLocations, accumulatedHours,
                   baseLocations, tankerCovers, heliCovers, noFires,
                   fireSizes, fireLocations, ffdiRanges, configurations,
                   configsE, configsP, baseConfigsMax, fireConfigsMax,
                   thresholds, expFiresComp, expectedE, expectedP, selectedE,
                   weightsP, time, stepSize, lookahead, thread_id, control,
                   lambdas, method, expectedTemp):

    """ This is a simple fast heuristic for approximately relocating the
    aircraft. An optimal algorithm would be too slow and may not be
    necessary given the global optimisation """

    global noBases
    global noPatches
    global noAircraft
    global noFiresMax
    global noConfigs
    global noConfE
    global noConfP

    """ Whether aircraft assignments are possible """
    canBase = cuda.local.array((noAircraft, noBases), dtype=b1)
    canFire = cuda.local.array((noAircraft, noFiresMax), dtype=b1)
    baseTankersE = cuda.local.array(noBases, dtype=int32)
    baseHelisE = cuda.local.array(noBases, dtype=int32)
    baseTankersL = cuda.local.array(noBases, dtype=int32)
    baseHelisL = cuda.local.array(noBases, dtype=int32)
    fireTankersE = cuda.local.array(noFiresMax, dtype=int32)
    fireHelisE = cuda.local.array(noFiresMax, dtype=int32)
    fireTankersL = cuda.local.array(noFiresMax, dtype=int32)
    fireHelisL = cuda.local.array(noFiresMax, dtype=int32)
    fireImproveTankerE = cuda.local.array(noFiresMax, dtype=float32)
    fireImproveHeliE = cuda.local.array(noFiresMax, dtype=float32)
    fireImproveTankerL = cuda.local.array(noFiresMax, dtype=float32)
    fireImproveHeliL = cuda.local.array(noFiresMax, dtype=float32)
    baseImproveTankerE = cuda.local.array(noBases, dtype=float32)
    baseImproveHeliE = cuda.local.array(noBases, dtype=float32)
    fireMaxTankersE = cuda.local.array(noFiresMax, dtype=int32)
    fireMaxTankersL = cuda.local.array(noFiresMax, dtype=int32)
    baseMaxTankersE = cuda.local.array(noFiresMax, dtype=int32)
    baseMaxTankersL = cuda.local.array(noFiresMax, dtype=int32)
    fireMaxHelisE = cuda.local.array(noBases, dtype=int32)
    fireMaxHelisL = cuda.local.array(noBases, dtype=int32)
    baseMaxHelisE = cuda.local.array(noBases, dtype=int32)
    baseMaxHelisL = cuda.local.array(noBases, dtype=int32)
    updatePatches = cuda.local.array(noPatches, dtype=b1)
    updateBases = cuda.local.array(noBases, dtype=b1)
    baseConfigsPossible = cuda.local.array(noConfP, dtype=int32)
    configNos = cuda.local.array(4, dtype=int32)
    fireCumulativeSavings = cuda.local.array(noFiresMax, dtype=float32)
    patchExpectedDamage = cuda.local.array(noPatches, dtype=float32)
    nearest = cuda.local.array(noAircraft, dtype=float32)

    """ First make sure that the next period assignments are zeroed """
    for resource in range(noAircraft):
        aircraftAssignments[thread_id][time+1][resource][0] = 0
        aircraftAssignments[thread_id][time+1][resource][1] = 0

    """ Whether patches are covered within threshold time for different
    aircraft types """

    for base in range(noBases):
        baseTankersE[base] = 0
        baseTankersL[base] = 0
        baseHelisE[base] = 0
        baseHelisL[base] = 0
        baseMaxTankersE[base] = 0
        baseMaxTankersL[base] = 0
        baseMaxHelisE[base] = 0
        baseMaxHelisL[base] = 0

    for patch in range(noPatches):
        patchExpectedDamage[patch] = 0

    for fire in range(noFires[thread_id][time]):
        fireTankersE[fire] = 0
        fireTankersL[fire] = 0
        fireHelisE[fire] = 0
        fireHelisL[fire] = 0
        fireMaxTankersE[fire] = 0
        fireMaxTankersL[fire] = 0
        fireMaxHelisE[fire] = 0
        fireMaxHelisL[fire] = 0
        fireCumulativeSavings[fire] = 0

    """ Threshold distances """
    if method == 2:
        fw = 1
        pw = 1

        if control == 0:
            maxFire = thresholds[0]
            maxBase = 0
        elif control == 1:
            maxFire = math.inf
            maxBase = 0
        elif control == 2:
            maxFire = thresholds[0]
            maxBase = thresholds[1]
        elif control == 3:
            maxFire = math.inf
            maxBase = thresholds[1]
        elif control == 4:
            maxFire = thresholds[0]
            maxBase = math.inf
        elif control == 5:
            maxFire = math.inf
            maxBase = math.inf
    else:
        fw = lambdas[control][0]
        pw = lambdas[control][1]
        maxFire = math.inf
        maxBase = math.inf

    """ Pre calcs """
    # Nearest base to each aircraft
    for resource in range(len(resourceTypes)):
        nearest[resource] = 0
        nearestDist = math.inf

        for base in range(len(baseLocations)):
            dist = geoDist(aircraftLocations[thread_id][time][resource],
                           baseLocations[base])
            if dist < nearestDist:
                nearestDist = dist
                nearest[resource] = base + 1

    # Possible aircraft to base assignments based on control
    for resource in range(len(resourceTypes)):
        for base in range(len(baseLocations)):
            dist = geoDist(aircraftLocations[thread_id][time][resource],
                           baseLocations[base])
            travTime = dist / resourceSpeeds[resource]
            accTime = accumulatedHours[thread_id][resource][time]
            maxTime = maxHours[resource]

            canBase[resource, base] = (
                True if (travTime <= maxBase
                         and travTime +  accTime <= maxTime)
                        or base == nearest[resource]
                else False)

            if canBase[resource, base]:
                if resourceTypes[resource] == 0:
                    baseMaxTankersE[base] += 1

                elif resourceTypes[resource] == 1:
                    baseMaxHelisE[base] += 1

    # Possible aircraft to fire assignments based on control
    for resource in range(len(resourceTypes)):
        for fire in range(noFires[thread_id][time]):
            dist = geoDist(aircraftLocations[thread_id][time][resource],
                           fireLocations[thread_id][time][fire])
            travTime = dist / resourceSpeeds[resource]
            accTime = accumulatedHours[thread_id][resource][time]
            maxTime = maxHours[resource]

            canFire[resource, fire] = (
                True if (travTime <= maxFire
                         and travTime + accTime <= maxTime)
                else False)

            if canFire[resource, fire]:
                if resourceTypes[resource] == 0:
                    if travTime < 1/3:
                        fireMaxTankersE[fire] += 1
                    else:
                        fireMaxTankersL[fire] += 1
                elif resourceTypes[resource] == 1:
                    if travTime < 1/3:
                        fireMaxHelisE[fire] += 1
                    else:
                        fireMaxHelisL[fire] += 1

    # While remaining aircraft:
    """ Track latest update. Indexing of types and bases/fires starts at 1.0
    indicates that the process has not started yet """
    lastUpdateDType = 0
    thisUpdateDType = 0
    thisUpdateACType = 0
    thisUpdateIdx = 0
    remaining = noAircraft

    while remaining > 0:
        thisUpdateImprovement = -10

        if int(remaining) == noAircraft:
            # Initialise incremental benefit for each base and fire of one more
            # (early and late):
            for fire in range(noFires[thread_id][time]):
                fireTankersE[fire] = 0
                fireTankersL[fire] = 0
                fireHelisE[fire] = 0
                fireHelisL[fire] = 0

                # Tankers
                # Early
                if fireMaxTankersE[fire] > 0:
                    configNos[0] = 1
                    configNos[1] = 0
                    configNos[2] = 0
                    configNos[3] = 0

                    getAllConfigsSorted(configurations, configsE,
                                        baseConfigsPossible,
                                        expectedE[fire], configNos)
                    config = baseConfigsPossible[0]

                    fireImproveTankerE[fire] = fw * (expectedE[fire][0]
                        - expectedE[fire][config])

                    if fireImproveTankerE[fire] > thisUpdateImprovement:
                        thisUpdateImprovement = fireImproveTankerE[fire]
                        thisUpdateDType = 0
                        thisUpdateACType = 1
                        thisUpdateIdx = fire

                else:
                    fireImproveTankerE[fire] = 0

                # Late
                if fireMaxTankersL[fire] > 0:
                    configNos[0] = 0
                    configNos[1] = 0
                    configNos[2] = 1
                    configNos[3] = 0

                    getAllConfigsSorted(configurations, configsE,
                                        baseConfigsPossible,
                                        expectedE[fire], configNos)
                    config = baseConfigsPossible[0]

                    fireImproveTankerL[fire] = fw * (expectedE[fire][0]
                        - expectedE[fire][config])

                    if fireImproveTankerL[fire] > thisUpdateImprovement:
                        thisUpdateImprovement = fireImproveTankerL[fire]
                        thisUpdateDType = 0
                        thisUpdateACType = 3
                        thisUpdateIdx = fire

                else:
                    fireImproveTankerL[fire] = 0

                # Helis
                # Early
                if fireMaxHelisE[fire] > 0:
                    configNos[0] = 0
                    configNos[1] = 1
                    configNos[2] = 0
                    configNos[3] = 0

                    getAllConfigsSorted(configurations, configsE,
                                        baseConfigsPossible,
                                        expectedE[fire], configNos)
                    config = baseConfigsPossible[0]

                    fireImproveHeliE[fire] = fw * (expectedE[fire][0]
                        - expectedE[fire][config])

                    if fireImproveHeliE[fire] > thisUpdateImprovement:
                        thisUpdateImprovement = fireImproveHeliE[fire]
                        thisUpdateDType = 0
                        thisUpdateACType = 2
                        thisUpdateIdx = fire

                else:
                    fireImproveHeliE[fire] = 0

                # Late
                if fireMaxHelisL[fire] > 0:
                    configNos[0] = 0
                    configNos[1] = 0
                    configNos[2] = 0
                    configNos[3] = 1

                    getAllConfigsSorted(configurations, configsE,
                                        baseConfigsPossible,
                                        expectedE[fire], configNos)
                    config = baseConfigsPossible[0]

                    fireImproveHeliL[fire] = fw * (expectedE[fire][0]
                        - expectedE[fire][config])

                    if fireImproveHeliL[fire] > thisUpdateImprovement:
                        thisUpdateImprovement = fireImproveHeliL[fire]
                        thisUpdateDType = 0
                        thisUpdateACType = 4
                        thisUpdateIdx = fire

                else:
                    fireImproveHeliL[fire] = 0

            for base in range(noBases):
                baseTankersE[base] = 0
                baseTankersL[base] = 0
                baseHelisE[base] = 0
                baseHelisL[base] = 0
                # We ignore early and late here and assume all aircraft
                # lateness/earliness are just a function of the distance
                # between the base and patch

                # Tanker
                if baseMaxTankersE[base] > 0:
                    expectedDamageBase = 0.0
                    baseDamageBaseline = 0.0

                    for patch in range(noPatches):
                        if tankerCovers[patch, base] <= min(maxFire, 1/3):
                            # If the patch is close to the base
                            configNos[0] = 1
                            configNos[1] = 0
                            configNos[2] = 0
                            configNos[3] = 0

                            getAllConfigsSorted(configurations, configsP,
                                                baseConfigsPossible,
                                                expectedP[patch], configNos)
                            config = baseConfigsPossible[0]

                            expFires = 0.0

                            for tt in range(time, time + lookahead):
                                expFires += expFiresComp[tt, 0, base]

                            n_c = max(1, expFires)
                            weight_c = 1/n_c

                            expectedDamageBase += pw * (
                                    expectedP[patch, 0] * (1 - weight_c) -
                                    expectedP[patch, config] * weight_c)

                            baseDamageBaseline += pw * expectedP[patch, 0]

                        elif tankerCovers[patch, base] <= maxFire:
                            # If the patch is further away but still allowed
                            configNos[0] = 0
                            configNos[1] = 0
                            configNos[2] = 1
                            configNos[3] = 0

                            getAllConfigsSorted(configurations, configsP,
                                                baseConfigsPossible,
                                                expectedP[patch], configNos)
                            config = baseConfigsPossible[0]

                            expFires = 0.0

                            for tt in range(time, time + lookahead):
                                expFires += (expFiresComp[tt, 0, base]
                                             + expFiresComp[tt, 2, base])

                            n_c = max(1, expFires)
                            weight_c = 1/n_c

                            expectedDamageBase += pw * (
                                    expectedP[patch, 0] * (1 - weight_c) -
                                    expectedP[patch, config] * weight_c)

                            baseDamageBaseline += pw * expectedP[patch, 0]

                    baseImproveTankerE[base] = (baseDamageBaseline
                        - expectedDamageBase)

                    if baseImproveTankerE[base] > thisUpdateImprovement:
                            thisUpdateImprovement = baseImproveTankerE[base]
                            thisUpdateDType = 1
                            thisUpdateACType = 1
                            thisUpdateIdx = base
                else:
                    baseImproveTankerE[base] = 0.0

                # Heli
                if baseMaxHelisE[base] > 0:
                    expectedDamageBase = 0.0
                    baseDamageBaseline = 0.0

                    for patch in range(noPatches):
                        if heliCovers[patch, base] <= min(maxFire, 1/3):
                            configNos[0] = 0
                            configNos[1] = 1
                            configNos[2] = 0
                            configNos[3] = 0

                            getAllConfigsSorted(configurations, configsP,
                                                baseConfigsPossible,
                                                expectedP[patch], configNos)
                            config = baseConfigsPossible[0]

                            # If the patch is close to the base
                            expFires = 0.0

                            for tt in range(time, time + lookahead):
                                expFires += expFiresComp[tt, 1, base]

                            n_c = max(1, expFires)
                            weight_c = 1/n_c

                            expectedDamageBase += pw * (
                                    expectedP[patch, 0] * (1 - weight_c) -
                                    expectedP[patch, config] * weight_c)

                            baseDamageBaseline += pw * expectedP[patch, 0]

                        elif heliCovers[patch, base] <= maxFire:
                            configNos[0] = 0
                            configNos[1] = 0
                            configNos[2] = 0
                            configNos[3] = 1

                            getAllConfigsSorted(configurations, configsP,
                                                baseConfigsPossible,
                                                expectedP[patch], configNos)
                            config = baseConfigsPossible[0]

                            # If the patch is close to the base
                            expFires = 0.0

                            for tt in range(time, time + lookahead):
                                expFires += (expFiresComp[tt, 1, base]
                                             + expFiresComp[tt, 3, base])

                            n_c = max(1, expFires)
                            weight_c = 1/n_c

                            expectedDamageBase += pw * (
                                    expectedP[patch, 0] * (1 - weight_c) -
                                    expectedP[patch, config] * weight_c)

                            baseDamageBaseline += pw * expectedP[patch, 0]

                    baseImproveHeliE[base] = (baseDamageBaseline
                        - expectedDamageBase)

                    if baseImproveTankerE[base] > thisUpdateImprovement:
                            thisUpdateImprovement = baseImproveTankerE[base]
                            thisUpdateDType = 1
                            thisUpdateACType = 2
                            thisUpdateIdx = base

                else:
                    baseImproveHeliE[base] = 0.0

        else:
            # Only update recently changed values (fires just updated or
            # patches affected by base assignments)
            if int(lastUpdateDType) == 0:
                # Fire
                # Current benefit
                currentImprove = fireCumulativeSavings[thisUpdateIdx]

                # 1 more early tanker
                if fireMaxTankersE[thisUpdateIdx] > fireTankersE[thisUpdateIdx]:
                    configNos[0] = fireTankersE[thisUpdateIdx] + 1
                    configNos[1] = fireHelisE[thisUpdateIdx]
                    configNos[2] = fireTankersL[thisUpdateIdx]
                    configNos[3] = fireHelisL[thisUpdateIdx]

                    getAllConfigsSorted(configurations, configsE,
                                        baseConfigsPossible,
                                        expectedE[thisUpdateIdx], configNos)
                    config = baseConfigsPossible[0]

                    fireImproveTankerE[thisUpdateIdx] = fw * (
                        expectedE[thisUpdateIdx][0] - currentImprove
                        - expectedE[thisUpdateIdx][config])
                else:
                    fireImproveTankerE[thisUpdateIdx] = 0

                # 1 more early heli
                if fireMaxHelisE[thisUpdateIdx] > fireHelisE[thisUpdateIdx]:
                    configNos[0] = fireTankersE[thisUpdateIdx]
                    configNos[1] = fireHelisE[thisUpdateIdx] + 1
                    configNos[2] = fireTankersL[thisUpdateIdx]
                    configNos[3] = fireHelisL[thisUpdateIdx]

                    getAllConfigsSorted(configurations, configsE,
                                        baseConfigsPossible,
                                        expectedE[thisUpdateIdx], configNos)
                    config = baseConfigsPossible[0]

                    fireImproveHeliE[thisUpdateIdx] = fw * (
                        expectedE[thisUpdateIdx][0] - currentImprove
                        - expectedE[thisUpdateIdx][config])
                else:
                    fireImproveHeliE[thisUpdateIdx] = 0

                # 1 more late tanker
                if fireMaxTankersL[thisUpdateIdx] > fireTankersL[thisUpdateIdx]:
                    configNos[0] = fireTankersE[thisUpdateIdx]
                    configNos[1] = fireHelisE[thisUpdateIdx]
                    configNos[2] = fireTankersL[thisUpdateIdx] + 1
                    configNos[3] = fireHelisL[thisUpdateIdx]

                    getAllConfigsSorted(configurations, configsE,
                                        baseConfigsPossible,
                                        expectedE[thisUpdateIdx], configNos)
                    config = baseConfigsPossible[0]

                    fireImproveTankerL[thisUpdateIdx] = fw * (
                        expectedE[thisUpdateIdx][0] - currentImprove
                        - expectedE[thisUpdateIdx][config])
                else:
                    fireImproveTankerL[thisUpdateIdx] = 0

                # 1 more late heli
                if fireMaxHelisL[thisUpdateIdx] > fireHelisL[thisUpdateIdx]:
                    configNos[0] = fireTankersE[thisUpdateIdx]
                    configNos[1] = fireHelisE[thisUpdateIdx]
                    configNos[2] = fireTankersL[thisUpdateIdx]
                    configNos[3] = fireHelisL[thisUpdateIdx] + 1

                    getAllConfigsSorted(configurations, configsE,
                                        baseConfigsPossible,
                                        expectedE[thisUpdateIdx], configNos)
                    config = baseConfigsPossible[0]

                    fireImproveHeliL[thisUpdateIdx] = fw * (
                        expectedE[thisUpdateIdx][0] - currentImprove
                        - expectedE[thisUpdateIdx][config])
                else:
                    fireImproveHeliL[thisUpdateIdx] = 0

            elif lastUpdateDType == 1:
                for patch in range(noPatches):
                    updatePatches[patch] = 0

                for base in range(noBases):
                    updateBases[base] = False

                # Current benefit
                currentImprove = 0.0

                for patch in range(noPatches):
                    currentImprove += patchExpectedDamage[patch]

                ###############################################################
                # First recalculate the base that was just updated
                # An extra tanker ################
                if baseMaxTankersE[thisUpdateIdx] > baseTankersE[thisUpdateIdx]:
                    expectedDamageBase = 0.0
                    baseTankersE[thisUpdateIdx] += 1

                    # Calculate the improvement of one more tanker at this
                    # base for each patch
                    for patch in range(noPatches):
                        componentNos(configNos, tankerCovers, heliCovers,
                                     baseTankersE, baseHelisE, updateBases,
                                     patch, maxFire, True)

                        if tankerCovers[patch, thisUpdateIdx] <= maxFire:

                            updatePatches[patch] = True

                            if tankerCovers[patch, thisUpdateIdx] <= 1/3:
                                configNos[0] += 1

                            else:
                                configNos[2] += 1

                            # Get all possible configs and sort them by
                            # benefit to this patch
                            getAllConfigsSorted(
                                    configurations, configsP,
                                    baseConfigsPossible, expectedP[patch],
                                    configNos)

                            expectedDamageBase += expectedDamage(
                                baseConfigsPossible, configurations,
                                tankerCovers, heliCovers, baseTankersE,
                                baseHelisE, expectedP, expFiresComp, weightsP,
                                patch, time, lookahead, pw, maxFire)

                        else:
                            expectedDamageBase += patchExpectedDamage[patch]

                    baseImproveTankerE[thisUpdateIdx] = (
                            currentImprove - expectedDamageBase)
                    baseTankersE[thisUpdateIdx] -= 1

                else:
                    baseImproveTankerE[thisUpdateIdx] = 0.0

                # An extra helicopter ############
                if baseMaxHelisE[thisUpdateIdx] > baseHelisE[thisUpdateIdx]:
                    expectedDamageBase = 0.0
                    baseHelisE[thisUpdateIdx] += 1

                    # Calculate the improvement of one more tanker at this
                    # base for each patch
                    for patch in range(noPatches):
                        componentNos(configNos, tankerCovers, heliCovers,
                                     baseTankersE, baseHelisE, updateBases,
                                     patch, maxFire, True)

                        if heliCovers[patch, thisUpdateIdx] <= maxFire:

                            updatePatches[patch] = True

                            if heliCovers[patch, thisUpdateIdx] <= 1/3:
                                configNos[1] += 1

                            else:
                                configNos[3] += 1

                            # Get all possible configs and sort them by
                            # benefit to this patch
                            getAllConfigsSorted(
                                    configurations, configsP,
                                    baseConfigsPossible, expectedP[patch],
                                    configNos)

                            expectedDamageBase += expectedDamage(
                                baseConfigsPossible, configurations,
                                tankerCovers, heliCovers, baseTankersE,
                                baseHelisE, expectedP, expFiresComp, weightsP,
                                patch, time, lookahead, pw, maxFire)

                        else:
                            expectedDamageBase += patchExpectedDamage[patch]

                    baseImproveHeliE[thisUpdateIdx] = (
                        currentImprove - expectedDamageBase)
                    baseHelisE[thisUpdateIdx] -= 1

                else:
                    baseImproveHeliE[thisUpdateIdx] = 0.0

                ###############################################################
                # Bases that are affected by adjusted patches
                for baseOther in range(noBases):
                    if updateBases[baseOther]:

                        # First recalculate the base that was just updated
                        # An extra tanker ################
                        if baseMaxTankersE[baseOther] > baseTankersE[baseOther]:
                            expectedDamageBase = 0.0
                            baseTankersE[baseOther] += 1

                            # Calculate the improvement of one more tanker at
                            # this base for each patch
                            for patch in range(noPatches):
                                componentNos(
                                        configNos, tankerCovers, heliCovers,
                                        baseTankersE, baseHelisE, updateBases,
                                        patch, maxFire, False)

                                if tankerCovers[patch, baseOther] <= maxFire:

                                    if tankerCovers[patch, baseOther] <= 1/3:
                                        configNos[0] += 1
                                    else:
                                        configNos[2] += 1

                                    # Get all possible configs and sort them by
                                    # benefit to this patch
                                    getAllConfigsSorted(
                                            configurations, configsP,
                                            baseConfigsPossible,
                                            expectedP[patch], configNos)

                                    expectedDamageBase += expectedDamage(
                                            baseConfigsPossible,
                                            configurations, tankerCovers,
                                            heliCovers, baseTankersE,
                                            baseHelisE, expectedP,
                                            expFiresComp, weightsP, patch,
                                            time, lookahead, pw, maxFire)

                                else:
                                    expectedDamageBase += (
                                            patchExpectedDamage[patch])

                            baseImproveTankerE[baseOther] = (
                                currentImprove - expectedDamageBase)
                            baseTankersE[baseOther] -= 1

                        else:
                            baseImproveTankerE[baseOther] = 0.0

                        # An extra helicopter ############
                        if baseMaxHelisE[baseOther] > baseHelisE[baseOther]:
                            expectedDamageBase = 0.0
                            baseHelisE[baseOther] += 1

                            # Calculate the improvement of one more tanker at
                            # this base for each patch
                            for patch in range(noPatches):
                                componentNos(
                                        configNos, tankerCovers, heliCovers,
                                        baseTankersE, baseHelisE, updateBases,
                                        patch, maxFire, False)

                                if heliCovers[patch, baseOther] <= maxFire:

                                    if heliCovers[patch, baseOther] <= 1/3:
                                        configNos[1] += 1
                                    else:
                                        configNos[3] += 1

                                    # Get all possible configs and sort them by
                                    # benefit to this patch
                                    getAllConfigsSorted(
                                            configurations, configsP,
                                            baseConfigsPossible,
                                            expectedP[patch], configNos)

                                    expectedDamageBase += expectedDamage(
                                            baseConfigsPossible,
                                            configurations, tankerCovers,
                                            heliCovers, baseTankersE,
                                            baseHelisE, expectedP,
                                            expFiresComp, weightsP, patch,
                                            time, lookahead, pw, maxFire)

                                else:
                                    expectedDamageBase += (
                                            patchExpectedDamage[patch])

                            baseImproveHeliE[baseOther] = (
                                currentImprove - expectedDamageBase)
                            baseHelisE[baseOther] -= 1

                        else:
                            baseImproveHeliE[baseOther] = 0.0

        # Pick the best assignment ############################################
        for fire in range(noFires[thread_id][time]):
            if fireImproveTankerE[fire] > thisUpdateImprovement:
                    thisUpdateImprovement = fireImproveTankerE[fire]
                    thisUpdateDType = 0
                    thisUpdateACType = 1
                    thisUpdateIdx = fire

            if fireImproveTankerL[fire] > thisUpdateImprovement:
                    thisUpdateImprovement = fireImproveTankerL[fire]
                    thisUpdateDType = 0
                    thisUpdateACType = 3
                    thisUpdateIdx = fire

            if fireImproveHeliE[fire] > thisUpdateImprovement:
                    thisUpdateImprovement = fireImproveHeliE[fire]
                    thisUpdateDType = 0
                    thisUpdateACType = 2
                    thisUpdateIdx = fire

            if fireImproveHeliL[fire] > thisUpdateImprovement:
                    thisUpdateImprovement = fireImproveHeliL[fire]
                    thisUpdateDType = 0
                    thisUpdateACType = 4
                    thisUpdateIdx = fire

        for base in range(noBases):
            if baseImproveTankerE[base] > thisUpdateImprovement:
                    thisUpdateImprovement = baseImproveTankerE[base]
                    thisUpdateDType = 1
                    thisUpdateACType = 1
                    thisUpdateIdx = base

            if baseImproveHeliE[base] > thisUpdateImprovement:
                    thisUpdateImprovement = baseImproveHeliE[base]
                    thisUpdateDType = 1
                    thisUpdateACType = 2
                    thisUpdateIdx = base

        #######################################################################
        # Now update the number of aircraft assigned to the chosen base/fire
        # and the cumulative damage saved
        if int(thisUpdateDType) == 0:
            if int(thisUpdateACType) == 1:
                fireTankersE[thisUpdateIdx] += 1
                fireCumulativeSavings[thisUpdateIdx] += (
                    fireImproveTankerE[thisUpdateIdx])

            elif int(thisUpdateACType) == 2:
                fireHelisE[thisUpdateIdx] += 1
                fireCumulativeSavings[thisUpdateIdx] += (
                    fireImproveHeliE[thisUpdateIdx])

            elif int(thisUpdateACType) == 3:
                fireTankersL[thisUpdateIdx] += 1
                fireCumulativeSavings[thisUpdateIdx] += (
                    fireImproveTankerL[thisUpdateIdx])

            elif int(thisUpdateACType) == 4:
                fireHelisL[thisUpdateIdx] += 1
                fireCumulativeSavings[thisUpdateIdx] += (
                    fireImproveHeliL[thisUpdateIdx])

        elif thisUpdateDType == 1:
            if thisUpdateACType == 1:
                baseTankersE[thisUpdateIdx] += 1

            elif thisUpdateACType == 2:
                baseHelisE[thisUpdateIdx] += 1

            for patch in range(noPatches):
                if updatePatches[patch]:
                    componentNos(
                            configNos, tankerCovers, heliCovers,
                            baseTankersE, baseHelisE, updateBases,
                            patch, maxFire, False)

                    # Get all possible configs and sort them by benefit to this
                    # patch
                    getAllConfigsSorted(
                            configurations, configsP, baseConfigsPossible,
                            expectedP[patch], configNos)

                    patchExpectedDamage[patch] = expectedDamage(
                            baseConfigsPossible, configurations, tankerCovers,
                            heliCovers, baseTankersE, baseHelisE, expectedP,
                            expFiresComp, weightsP, patch, time, lookahead,
                            pw, maxFire)

        #######################################################################
        # For given assignment, pick best A/C #################################
        minMax = math.inf
        nextAC = 0
        nextBest = 0

        # Next assignment is to a fire
        if int(thisUpdateDType) == 0:
            if int(thisUpdateACType) == 1:
                for resource in range(noAircraft):
                    nextBest = 0

                    if (int(resourceTypes[resource]) == 0
                        and int(aircraftAssignments[thread_id][time+1][
                                resource][0]) == 0):

                        if canFire[resource, thisUpdateIdx]:
                            dist = geoDist(
                                aircraftLocations[thread_id][time][resource],
                                fireLocations[thread_id][time][thisUpdateIdx])
                            travTime = dist / resourceSpeeds[resource]

                            if travTime <= 1/3:
                                # This resource is a candidate

                                # Check potential benefit to other fires
                                for fire in range(noFires[thread_id][time]):
                                    if (fire != int(thisUpdateIdx)
                                        and canFire[resource, fire]):

                                        dist = geoDist(
                                            aircraftLocations[thread_id][time][
                                                resource],
                                            fireLocations[thread_id][time][
                                                fire])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime <= 1/3:
                                            if (fireImproveTankerE[fire] >
                                                nextBest):

                                                nextBest = fireImproveTankerE[
                                                    fire]
                                        else:
                                            if (fireImproveTankerL[fire] >
                                                nextBest):
                                                nextBest = fireImproveTankerL[
                                                    fire]

                                # Check potential benefit to bases
                                for base in range(noBases):
                                    if canBase[resource, base]:

                                        if baseImproveTankerE[base] > nextBest:
                                            nextBest = baseImproveTankerE[base]

                                if nextBest < minMax:
                                    minMax = nextBest
                                    nextAC = resource

            elif thisUpdateACType == 2:
                for resource in range(noAircraft):
                    nextBest = 0

                    if (int(resourceTypes[resource]) == 1
                        and int(aircraftAssignments[thread_id][time+1][
                                resource][0]) == 0):

                        if canFire[resource, thisUpdateIdx]:
                            dist = geoDist(
                                aircraftLocations[thread_id][time][resource],
                                fireLocations[thread_id][time][thisUpdateIdx])
                            travTime = dist / resourceSpeeds[resource]

                            if travTime <= 1/3:
                                # This resource is a candidate

                                # Check potential benefit to other fires
                                for fire in range(noFires[thread_id][time]):
                                    if (fire != int(thisUpdateIdx)
                                        and canFire[resource, fire]):

                                        dist = geoDist(
                                            aircraftLocations[thread_id][time][
                                                resource],
                                            fireLocations[thread_id][time][
                                                fire])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime < 1/3:
                                            if (fireImproveHeliE[fire] >
                                                nextBest):

                                                nextBest = fireImproveHeliE[
                                                    fire]
                                        else:
                                            if (fireImproveHeliL[fire] >
                                                nextBest):

                                                nextBest = fireImproveHeliL[
                                                fire]

                                # Check potential benefit to bases
                                for base in range(noBases):
                                    if canBase[resource, base]:

                                        if baseImproveHeliE[base] > nextBest:
                                            nextBest = baseImproveHeliE[base]

                                if nextBest < minMax:
                                    minMax = nextBest
                                    nextAC = resource

            if thisUpdateACType == 3:
                for resource in range(noAircraft):
                    nextBest = 0

                    if (int(resourceTypes[resource]) == 0
                        and int(aircraftAssignments[thread_id][time+1][
                                resource][0]) == 0):

                        if canFire[resource, thisUpdateIdx]:
                            dist = geoDist(
                                aircraftLocations[thread_id][time][resource],
                                fireLocations[thread_id][time][thisUpdateIdx])
                            travTime = dist / resourceSpeeds[resource]

                            if travTime > 1/3:
                                # This resource is a candidate

                                # Check potential benefit to other fires
                                for fire in range(noFires[thread_id][time]):
                                    if (fire != int(thisUpdateIdx)
                                        and canFire[resource, fire]):

                                        dist = geoDist(
                                            aircraftLocations[thread_id][time][
                                                resource],
                                            fireLocations[thread_id][time][
                                                fire])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime < 1/3:
                                            if (fireImproveTankerE[fire] >
                                                nextBest):

                                                nextBest = fireImproveTankerE[
                                                    fire]
                                        else:
                                            if (fireImproveTankerL[fire] >
                                                nextBest):

                                                nextBest = fireImproveTankerL[
                                                    fire]

                                # Check potential benefit to bases
                                for base in range(noBases):
                                    if canBase[resource, base]:

                                        if baseImproveTankerE[base] > nextBest:
                                            nextBest = baseImproveTankerE[base]

                                if nextBest < minMax:
                                    minMax = nextBest
                                    nextAC = resource

            if thisUpdateACType == 4:
                for resource in range(noAircraft):
                    nextBest = 0

                    if (int(resourceTypes[resource]) == 1
                        and int(aircraftAssignments[thread_id][time+1][
                                resource][0]) == 0):

                        if canFire[resource, thisUpdateIdx]:
                            dist = geoDist(
                                aircraftLocations[thread_id][time][resource],
                                fireLocations[thread_id][time][thisUpdateIdx])

                            travTime = dist / resourceSpeeds[resource]

                            if travTime > 1/3:
                                # This resource is a candidate

                                # Check potential benefit to other fires
                                for fire in range(noFires[thread_id][time]):
                                    if (fire != thisUpdateIdx
                                        and canFire[resource, fire]):

                                        dist = geoDist(
                                            aircraftLocations[thread_id][time][
                                                resource],
                                            fireLocations[thread_id][time][
                                                fire])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime < 1/3:
                                            if (fireImproveHeliE[fire] >
                                                nextBest):

                                                nextBest = fireImproveHeliE[
                                                    fire]
                                        else:
                                            if (fireImproveHeliL[fire] >
                                                nextBest):

                                                nextBest = fireImproveHeliL[
                                                    fire]

                                # Check potential benefit to bases
                                for base in range(noBases):
                                    if canBase[resource, base]:

                                        if baseImproveHeliE[base] > nextBest:
                                            nextBest = baseImproveHeliE[base]

                                if nextBest < minMax:
                                    minMax = nextBest
                                    nextAC = resource

        # Next assignment is a base
        elif thisUpdateDType == 1:
            if (thisUpdateACType == 1 or thisUpdateACType == 3):
                for resource in range(noAircraft):
                    nextBest = 0

                    if (resourceTypes[resource] == 0
                        and int(aircraftAssignments[thread_id][time+1][
                                resource][0] == 0)):

                        if canBase[resource, thisUpdateIdx]:
                            # This resource is a candidate

                            # Check potential benefit to other fires
                            for fire in range(noFires[thread_id][time]):
                                if canFire[resource, fire]:

                                    dist = geoDist(
                                        aircraftLocations[thread_id][time][
                                            resource],
                                        fireLocations[thread_id][time][fire])

                                    travTime = dist / resourceSpeeds[resource]

                                    if travTime <= 1/3:
                                        if fireImproveTankerE[fire] > nextBest:

                                            nextBest = fireImproveTankerE[fire]
                                    else:
                                        if fireImproveTankerL[fire] > nextBest:

                                            nextBest = fireImproveTankerL[fire]

                            # Check potential benefit to bases
                            for base in range(noBases):
                                if (base != thisUpdateIdx
                                    and canBase[resource, fire]):

                                    if baseImproveTankerE[base] > nextBest:
                                        nextBest = baseImproveTankerE[base]

                            if nextBest < minMax:
                                minMax = nextBest
                                nextAC = resource

            if (thisUpdateACType == 2 or thisUpdateACType == 4):
                for resource in range(noAircraft):
                    nextBest = 0

                    if (resourceTypes[resource] == 1
                        and int(aircraftAssignments[thread_id][time][
                                resource][0] == 0)):
                            # This resource is a candidate

                            # Check potential benefit to other fires
                            for fire in range(noFires[thread_id][time]):
                                if canFire[resource, fire]:
                                    dist = geoDist(
                                        aircraftLocations[thread_id][time][
                                            resource],
                                        fireLocations[thread_id][time][fire])

                                    travTime = dist / resourceSpeeds[resource]

                                    if travTime < 1/3:
                                        if fireImproveHeliE[fire] > nextBest:

                                            nextBest = fireImproveHeliE[fire]
                                    else:
                                        if fireImproveHeliL[fire] > nextBest:

                                            nextBest = fireImproveHeliL[fire]

                            # Check potential benefit to bases
                            for base in range(noBases):
                                if (base != thisUpdateIdx
                                    and canBase[resource, fire]):

                                    if baseImproveHeliE[base] > nextBest:
                                        nextBest = baseImproveHeliE[base]

                            if nextBest < minMax:
                                minMax = nextBest
                                nextAC = resource

        # Update the assignments matrix #######################################
        if int(thisUpdateDType == 0):
            aircraftAssignments[thread_id][time+1][nextAC][1] = (
                thisUpdateIdx + 1)

            # Pick nearest base to fire that is within the maximum relocation
            # distance from the aircraft's current base
            bestBase = 0
            bestTrav = math.inf
            for base in range(noBases):
                if canBase[nextAC, base]:
                    dist = geoDist(aircraftLocations[thread_id][time][nextAC],
                                   baseLocations[base])

                    travTime = dist / resourceSpeeds[nextAC]

                    if travTime < bestTrav:
                        bestBase = base
                        bestTrav = travTime

            aircraftAssignments[thread_id][time+1][nextAC][0] = bestBase + 1

        elif int(thisUpdateDType) == 1:
            aircraftAssignments[thread_id][time+1][nextAC][1] = 0
            aircraftAssignments[thread_id][time+1][nextAC][0] = (
                thisUpdateIdx + 1)

        if thread_id == 0:
            if time == 0:
                for fire in range(noFires[thread_id][time]):
                    expectedTemp[0, fire, 17 - remaining] = fireImproveTankerE[fire]
                    expectedTemp[1, fire, 17 - remaining] = fireImproveTankerL[fire]

                for base in range(noBases):
                    expectedTemp[2, base, 17 - remaining] = baseImproveTankerE[base]
                    expectedTemp[3, base, 17 - remaining] = baseImproveHeliE[base]
#                    expectedTemp[0, fire, 17 - remaining] = fireCumulativeSavings[fire]
#                    expectedTemp[2, fire, 17 - remaining] = fireTankersL[fire]

        #######################################################################
        # Reduce max component capacities (and possibly incremental
        # improvement) based on assignment just made
        for fire in range(noFires[thread_id][time]):
            if canFire[nextAC, fire]:
                dist = geoDist(aircraftLocations[thread_id][time][nextAC],
                               fireLocations[thread_id][time][fire])

                travTime = dist / resourceSpeeds[nextAC]

                if thisUpdateACType == 1 or thisUpdateACType == 3:
                    if (travTime <= 1/3):
                        fireMaxTankersE[fire] -= 1

                        if fireMaxTankersE[fire] == fireTankersE[fire]:
                            fireImproveTankerE[fire] = 0

                    else:
                        fireMaxTankersL[fire] -= 1

                        if fireMaxTankersL[fire] == fireTankersL[fire]:
                            fireImproveTankerL[fire] = 0

                elif thisUpdateACType == 2 or thisUpdateACType == 4:
                    if (travTime <= 1/3):
                        fireMaxHelisE[fire] -= 1

                        if fireMaxHelisE[fire] == fireHelisE[fire]:
                            fireImproveHeliE[fire] = 0

                    else:
                        fireMaxHelisL[fire] -= 1

                        if fireMaxHelisL[fire] == fireHelisL[fire]:
                            fireImproveHeliL[fire] = 0

        for base in range(noBases):
            if canBase[nextAC, base]:
                if thisUpdateACType == 1 or thisUpdateACType == 3:
                    baseMaxTankersE[base] -= 1

                    if baseMaxTankersE[base] == baseTankersE[fire]:
                        baseImproveTankerE[base] = 0

                elif thisUpdateACType == 2 or thisUpdateACType == 4:
                    baseMaxHelisE[base] -= 1

                    if baseMaxHelisE[base] == baseHelisE[fire]:
                        baseImproveHeliE[base] = 0

        # Repeat until assignments complete
        lastUpdateDType = thisUpdateDType
        remaining -= 1

    # Compute resulting fire configurations and patch configuration weights
    # given these assignments
    for fire in range(noFiresMax):
        configNos[0] = fireTankersE[fire]
        configNos[1] = fireHelisE[fire]
        configNos[2] = fireTankersL[fire]
        configNos[3] = fireHelisL[fire]

        getAllConfigsSorted(configurations, configsE, baseConfigsPossible,
                            expectedE[fire], configNos)
        selectedE[fire] = baseConfigsPossible[0]

    for patch in range(noPatches):
        componentNos(configNos, tankerCovers, heliCovers, baseTankersE,
                     baseHelisE, updateBases, patch, maxFire, False)

        getAllConfigsSorted(configurations, configsP, baseConfigsPossible,
                            expectedP[patch], configNos)

        patchExpectedDamage[patch] = expectedDamage(
                baseConfigsPossible, configurations, tankerCovers, heliCovers,
                baseTankersE, baseHelisE, expectedP, expFiresComp, weightsP,
                patch, time, lookahead, pw, maxFire)

    # If we have aircraft that have not been assigned to anything (not possible
    # if the program is run in its entirety), we need to make sure that they
    # are assigned at least to the nearest base.
    for resource in range(noAircraft):
        if aircraftAssignments[thread_id][time+1][resource][0] == 0:

            # Pick nearest base to fire that is within the maximum relocation
            # distance from the aircraft's current base
            bestBase = 0
            bestTrav = math.inf
            for base in range(noBases):
                if canBase[resource, base]:
                    dist = geoDist(aircraftLocations[thread_id][time][resource],
                                   baseLocations[base])

                    travTime = dist / resourceSpeeds[resource]

                    if travTime < bestTrav:
                        bestBase = base
                        bestTrav = travTime

            aircraftAssignments[thread_id][time+1][resource][0] = bestBase + 1


@cuda.jit(device=True)
def geoDist(x1d, x2d):

    # First convert to radians
    x1 = cuda.local.array(2, dtype=float32)
    x2 = cuda.local.array(2, dtype=float32)

    x1[0] = x1d[0] * math.pi/180
    x1[1] = x1d[1] * math.pi/180
    x2[0] = x2d[0] * math.pi/180
    x2[1] = x2d[1] * math.pi/180

    a = (math.sin(0.5 * (x2[1] - x1[1])) ** 2
         + math.cos(x1[0]) * math.cos(x2[0])
         * math.sin(0.5 * (x2[0] - x1[0])) ** 2)
    c = math.sqrt(a)
    dist = 2 * 6371 * math.asin(c)

    return dist


@cuda.jit(device=True)
def improvement(currentConf, proposedConfNos, configurations, validConfs,
                baseConfigsPossible, expectedE):
    pass


@cuda.jit(device=True)
def findConfigIdx(configurations, configs, noTE, noHE, noTL, noHL):
    config = cuda.local.array(4, dtype=float64)
    config[0] = noTE
    config[1] = noHE
    config[2] = noTL
    config[3] = noHL

    if config in configurations:
        return configurations.index([noTE, noHE, noTL, noHL])
    else:
        return 0


@cuda.jit(device=True)
def travTime(X1, X2, speed):

    dist = math.sqrt(((X1[0] - X2[0]) * 40000*math.cos((X1[1] + X2[1])
         * math.pi/360)/360) ** 2 + ((X2[1] - X1[1]) * 40000/360)**2)

    return dist / speed


@cuda.jit(device=True)
def expectedDamage(baseConfigsPossible, configurations, tankerCovers,
                   heliCovers, baseTankersE, baseHelisE, expectedP,
                   expFiresComp, weightsP, patch, time, lookahead, pw,
                   maxFire):

    expectedDamageBase = 0.0
    configIdx = 0
    weight_total = 0.0

    while weight_total < 1:
        config = baseConfigsPossible[configIdx]

        weight_c = potentialWeight(
                configurations, tankerCovers, heliCovers, baseTankersE,
                baseHelisE, expFiresComp, patch, config, time, lookahead,
                maxFire)

        weight_c = 1
        weight_c = min(weight_c, 1 - weight_total)
        weightsP[patch, config] = weight_c

        expectedDamageBase += pw * (expectedP[patch, config] * weight_c)

        weight_total += weight_c
        configIdx += 1

    return expectedDamageBase


@cuda.jit(device=True)
def componentNos(configNos, tankerCovers, heliCovers, baseTankersE, baseHelisE,
                 updateBases, patch, maxFire, initial):

    for c in range(4):
        configNos[c] = 0

    # Get best possible config
    for base in range(noBases):
        if tankerCovers[patch, base] <= min(
            maxFire, 1/3):

            if initial:
                updateBases[base] = True

            configNos[0] += baseTankersE[base]

        elif tankerCovers[patch, base] <= maxFire:

            if initial:
                updateBases[base] = True

            configNos[1] += baseTankersE[base]

        if heliCovers[patch, base] <= min(
            maxFire, 1/3):

            if initial:
                updateBases[base] = True

            configNos[2] += baseHelisE[base]

        elif heliCovers[patch, base] <= maxFire:

            if initial:
                updateBases[base] = True

            configNos[3] += baseHelisE[base]


@cuda.jit(device=True)
def potentialWeight(configurations, tankerCovers, heliCovers, baseTankersE,
                    baseHelisE, expFiresComp, patch, config, time, lookahead,
                    maxFire):

    weight_c = 0.0

    # Early tankers
    a_cb = 0.0

    for base in range(noBases):
        expFires = 0.0

        for tt in range(time, time + lookahead):
            expFires += expFiresComp[tt, 0, base]

        n_c = max(1, expFires)

        a_cb += ((baseTankersE[base]
                  if tankerCovers[patch, base] <= min(1/3, maxFire)
                  else 0) / n_c)

    q_c = max(1, configurations[config, 0])
    weight_c = a_cb / q_c

    # Early helis
    a_cb = 0.0

    for base in range(noBases):
        expFires = 0.0

        for tt in range(time, time + lookahead):
            expFires += expFiresComp[tt, 1, base]

        n_c = max(1, expFires)

        a_cb += ((baseHelisE[base]
                  if heliCovers[patch, base] <= min(1/3, maxFire)
                  else 0) / n_c)

    q_c = max(1, configurations[config, 1])
    weight_c = min(weight_c, (a_cb / q_c))

    # Late tankers
    a_cb = 0.0

    for base in range(noBases):
        expFires = 0.0

        for tt in range(time, time + lookahead):
            expFires += expFiresComp[tt, 2, base]

        n_c = max(1, expFires)

        a_cb += ((baseTankersE[base]
                  if (heliCovers[patch, base] <= maxFire and
                      heliCovers[patch, base] > 1/3)
                  else 0) / n_c)

    q_c = max(1, configurations[config, 2])
    weight_c = min(weight_c, (a_cb / q_c))

    # Late helis
    a_cb = 0.0

    for base in range(noBases):
        expFires = 0.0

        for tt in range(time, time + lookahead):
            expFires += expFiresComp[tt, 3, base]

        n_c = max(1, expFires)

        a_cb += ((baseHelisE[base]
                  if (heliCovers[patch, base] <= maxFire and
                      heliCovers[patch, base] > 1/3)
                  else 0) / n_c)

    q_c = max(1, configurations[config, 3])
    weight_c = min(weight_c, (a_cb / q_c))

    return weight_c


@cuda.jit(device=True)
def saveState(resourceAssignments, resourceTypes, resourceSpeeds, maxHours,
              aircraftLocations, accumulatedHours, patchLocations, fires,
              fireSizes, fireLocations, expectedE, expectedP, states,
              configurations, selectedE, weightsP, time, lookahead, thread_id):

    """ Remaining hours: Simple """
    stateVal = 0
    for resource in range(noAircraft):
        stateVal += maxHours[resource] - accumulatedHours[thread_id][time][
                resource]

    states[thread_id, time, 0] = stateVal

    """ Now save the expected damages for this assignment """
    shortTermE = 0
    for patch in range(noPatches):
        shortTermE += expectedE[patch, selectedE[patch]]

    states[thread_id, time, 1] = shortTermE

    shortTermP = 0
    for config in range(len(weightsP)):
        for patch in range(noPatches):
            shortTermP += weightsP[patch, config] * expectedP[patch, config]

    states[thread_id, time, 2] = shortTermP

    """ Other state values to try """

    """ Phase 2: Sum of Weighted Distance to Fires and Patches AFTER
    assignment """
    stateVal = 0
    # Fires
    for resource in range(noAircraft):
        for fire in range(fires[thread_id][time]):
            dist = geoDist(aircraftLocations[thread_id][time+1][resource],
                           fireLocations[thread_id][time+1][fire])

            stateVal += dist * expectedE[fire, 0]

    states[thread_id, time, 3] = stateVal

    stateVal = 0
    # Patches
    for resource in range(noAircraft):
        for patch in range(noPatches):
            dist = geoDist(aircraftLocations[thread_id][time+1][resource],
                           patchLocations[patch])

            stateVal += dist * expectedP[patch, 0]

    states[thread_id, time, 4] = stateVal

    """" Remaining hours: Weighted """
    stateVal = 0

    for resource in range(noAircraft):
        weight = 0
        for fire in range(fires[thread_id, time]):
            weight += expectedE[fire, 0]

        for patch in range(noPatches):
            weight += expectedP[patch, 0]

        stateVal += weight * (maxHours[resource]
            - accumulatedHours[thread_id][time][resource])

    """ OPTION 1 """
    """ Phase 1: Sum of Weighted Distances to Fires and Patches BEFORE
    assignments"""
    stateVal = 0
    # Fires
    for resource in range(noAircraft):
        for fire in range(fires[thread_id][time]):
            dist = geoDist(aircraftLocations[thread_id][time][resource],
                           fireLocations[thread_id][time][fire])

            stateVal += dist * expectedE[fire, 0]

    states[thread_id, time, 6] = stateVal

    stateVal = 0
    # Patches
    for resource in range(noAircraft):
        for patch in range(noPatches):
            dist = geoDist(aircraftLocations[thread_id][time][resource],
                           patchLocations[patch])

            stateVal += dist * expectedP[patch, 0]

    states[thread_id, time, 7] = stateVal

    """ OPTION 2 """
    """ Phase 3: Short-Term Expected Damage from Fires and Patches BEFORE
    assignments"""
    if time > 0:
        # Fires
        states[thread_id, time, 8] = shortTermE

        # Patches
        states[thread_id, time, 9] = shortTermP
    else:
        # Compute the no-relocation option to determine the baseline expected
        # damage for time 0
        states[thread_id, time, 8] = 0
        states[thread_id, time, 9] = 0


"""/////////////////////////////// WRAPPERS ////////////////////////////////"""
""" Main Wrapper """
def simulateROV(paths, sampleFFDIs, patchVegetations, patchAreas,
                patchLocations, baseLocations, resourceTypes, resourceSpeeds,
                maxHours, configurations, configsE, configsP, thresholds,
                ffdiRanges, rocA2PHMeans, rocA2PHSDs, occurrence, initSizeM,
                initSizeSD, initSuccess, extSuccess, tankerDists, heliDists,
                fireConfigsMax, baseConfigsMax, expFiresComp, totalSteps,
                lookahead, stepSize, accumulatedDamages, accumulatedHours,
                fires, fireSizes, fireLocations, firePatches,
                aircraftLocations, aircraftAssignments, controls, regressionX,
                regressionY, states, costs2Go, lambdas, method, noCont):

#    print(initSizeSD[0])
#    sys.exit()

    """ Set global values """
    global noBases
    global noPatches
    global noAircraft
    global noFiresMax
    global noConfigs
    global noConfE
    global noConfP
    global noControls
    global epsilon

    noBases = len(baseLocations)
    noPatches = len(patchLocations)
    noAircraft = len(resourceTypes)
    noFiresMax = 500
    noConfigs = len(configurations)
    noConfE = len(configsE)
    noConfP = len(configsP)
    noControls = noCont
    epsilon = sys.float_info.epsilon

    """ Copy data to the device """
    # Copy universal values to device. Path values are copied on-the-fly in
    # batches as too much memory will be required on the GPU if we copy it all
    # at once.
    d_sampleFFDIs = cuda.to_device(sampleFFDIs)
    d_patchVegetations = cuda.to_device(patchVegetations)
    d_patchAreas = cuda.to_device(patchAreas)
    d_patchLocations = cuda.to_device(patchLocations)
    d_baseLocations = cuda.to_device(baseLocations)
    d_ffdiRanges = cuda.to_device(ffdiRanges)
    d_rocA2PHMeans = cuda.to_device(rocA2PHMeans)
    d_rocA2PHSDs = cuda.to_device(rocA2PHSDs)
    d_occurrence = cuda.to_device(occurrence)
    d_initSizeM = cuda.to_device(initSizeM)
    d_initSizeSD = cuda.to_device(initSizeSD)
    d_initSuccess = cuda.to_device(initSuccess)
    d_extSuccess = cuda.to_device(extSuccess)
    d_resourceTypes = cuda.to_device(resourceTypes)
    d_resourceSpeeds = cuda.to_device(resourceSpeeds)
    d_configsE = cuda.to_device(configsE)
    d_configsP = cuda.to_device(configsP)
    d_thresholds = cuda.to_device(thresholds)
    d_regressionX = cuda.to_device(regressionX)
    d_regressionY = cuda.to_device(regressionY)
    d_tankerDists = cuda.to_device(tankerDists)
    d_heliDists = cuda.to_device(heliDists)
    d_maxHours = cuda.to_device(maxHours)
    d_configurations = cuda.to_device(configurations)
    d_baseConfigsMax = cuda.to_device(baseConfigsMax)
    d_fireConfigsMax = cuda.to_device(fireConfigsMax)
    d_expFiresComp = cuda.to_device(expFiresComp)
    d_lambdas = cuda.to_device(lambdas)

    """ Initial Monte Carlo Paths """
    simulateMC(
            paths, d_sampleFFDIs, d_patchVegetations, d_patchAreas,
            d_patchLocations, d_baseLocations, d_resourceTypes,
            d_resourceSpeeds, d_maxHours, d_configurations, d_configsE,
            d_configsP, d_thresholds, d_ffdiRanges, d_rocA2PHMeans,
            d_rocA2PHSDs, d_occurrence, d_initSizeM, d_initSizeSD,
            d_initSuccess, d_extSuccess, d_tankerDists, d_heliDists,
            d_fireConfigsMax, d_baseConfigsMax, d_expFiresComp, d_lambdas,
            totalSteps, lookahead, stepSize, accumulatedDamages,
            accumulatedHours, fires, fireSizes, fireLocations, firePatches,
            aircraftLocations, aircraftAssignments, controls, d_regressionX,
            d_regressionY, states, costs2Go, method, noControls)

#    sys.exit()
#
#    """ BACKWARD INDUCTION """
#    """ Regressions and Forward Path Re-Computations"""
#    for tt in range(totalSteps - 1, -1, -1):
#        """ Regressions """
#        for control in range(noControls):
#            """ Compute the regression using the relevant states and
#            costs2Go """
#            xs = states[:][tt]
#            ys = costs2Go[:][tt]
#
#            reg = smooth.NonParamRegression(
#                xs, ys, method=npr_methods.LocalPolynomialKernel(q=2))
#            reg.fit()
#
#            tempGrid = numpy.meshgrid(
#                numpy.linspace(min(xs[0]), max(xs[0], 50)),
#                numpy.linspace(min(xs[1]), max(xs[1], 50)),
#                numpy.linspace(min(xs[2]), max(xs[2], 50)))
#
#            regressionY[tt][control] = reg(tempGrid)
#
#            regressionX[tt][control] = numpy.array([
#                numpy.linspace(min(xs[0]), max(xs[0], 50)),
#                numpy.linspace(min(xs[1]), max(xs[1], 50)),
#                numpy.linspace(min(xs[2]), max(xs[2], 50))])
#
#            """ Push the regressions back onto the GPU for reuse in the forward
#            path recomputations """
#            d_regressionX[tt][control] = cuda.to_device(
#                    regressionX[tt][control])
#            d_regressionY[tt][control] = cuda.to_device(
#                    regressionY[tt][control])
#
#        simulateMC(
#                paths, d_sampleFFDIs, d_patchVegetations, d_patchAreas,
#                d_patchLocations, d_baseLocations, d_resourceTypes,
#                d_resourceSpeeds, d_maxHours, d_configurations, d_configsE,
#                d_configsP, d_thresholds, d_ffdiRanges, d_rocA2PHMeans,
#                d_rocA2PHSDs, d_occurrence, d_initSizeM, d_initSizeSD,
#                d_initSuccess, d_extSuccess, d_tankerDists, d_heliDists,
#                d_fireConfigsMax, d_baseConfigsMax, d_expFiresComp, d_lambdas,
#                totalSteps, lookahead, stepSize, accumulatedDamages,
#                accumulatedHours, fires, fireSizes, fireLocations, firePatches,
#                aircraftLocations, aircraftAssignments, controls,
#                d_regressionX, d_regressionY, states, costs2Go, method,
#                noControls, tt, optimal=True)

    """ Pull the final states and costs 2 go from the GPU and save to an output
    file. For analysis purposes, we need to print our paths to output csv files
    or data dumps (use Pandas?/Spark?)
    The extraction is already done but the saving of the data is not. We save
    the data in the calling routine. """


""" Monte Carlo Routine """
#@jit(parallel=True, fastmath=True)
def simulateMC(paths, d_sampleFFDIs, d_patchVegetations, d_patchAreas,
               d_patchLocations, d_baseLocations, d_resourceTypes,
               d_resourceSpeeds, d_maxHours, d_configurations, d_configsE,
               d_configsP, d_thresholds, d_ffdiRanges, d_rocA2PHMeans,
               d_rocA2PHSDs, d_occurrence, d_initSizeM, d_initSizeSD,
               d_initSuccess, d_extSuccess, d_tankerDists, d_heliDists,
               d_fireConfigsMax, d_baseConfigsMax, d_expFiresComp, d_lambdas,
               totalSteps, lookahead, stepSize, accumulatedDamages,
               accumulatedHours, fires, fireSizes, fireLocations, firePatches,
               aircraftLocations, aircraftAssignments, controls, d_regressionX,
               d_regressionY, states, costs2Go, method, noControls, start=0,
               optimal=False, static=-1):

    # Input values prefixed with 'd_' are already on the device and will not
    # be copied across. Values without the prefix need to be copied across.

    batches = math.ceil(paths / 1000)
    batchAmounts = [1000 for batch in range(batches - 1)]
    batchAmounts.append(paths - sum(batchAmounts))

    expectedTemp = numpy.zeros([4, 17, 17])
#    expectedTemp = numpy.zeros([15, 16])
    d_expectedTemp = cuda.to_device(expectedTemp)

    # Run this in chunks (i.e. multiple paths at a time)
    for b, batchSize in enumerate(batchAmounts):
        # We may even need to batch this on the CPU using Big-Data structures
        # CUDA requirements
        batchStart = sum(batchAmounts[:b])
        batchEnd = batchStart + batchAmounts[b]
        threadsperblock = 32
        blockspergrid = (batchSize + (threadsperblock - 1)) // threadsperblock

        # Copy batch-relevant memory to the device
        d_accumulatedDamages = cuda.to_device(accumulatedDamages[
                batchStart:batchEnd, start:(totalSteps+1), :])
        d_accumulatedHours = cuda.to_device(accumulatedHours[
                batchStart:batchEnd, start:(totalSteps+1), :])
        d_fires = cuda.to_device(fires[batchStart:batchEnd,
                                       start:(totalSteps+1)])
        d_fireSizes = cuda.to_device(fireSizes[batchStart:batchEnd,
                                               start:(totalSteps+1), :])
        d_fireLocations = cuda.to_device(fireLocations[batchStart:batchEnd,
                                                       start:(totalSteps+1),
                                                       :])
        d_firePatches = cuda.to_device(firePatches[batchStart:batchEnd,
                                                   start:(totalSteps+1), :])
        d_aircraftLocations = cuda.to_device(aircraftLocations[
                batchStart:batchEnd, start:(totalSteps+1), :])
        d_aircraftAssignments = cuda.to_device(aircraftAssignments[
                batchStart:batchEnd, start:(totalSteps+1), :])
        d_controls = cuda.to_device(controls[batchStart:batchEnd,
                                             start:(totalSteps+1)])
        d_states = cuda.to_device(states[batchStart:batchEnd,
                                         start:(totalSteps+1), :])
        d_costs2Go = cuda.to_device(costs2Go[batchStart:batchEnd,
                                             start:(totalSteps+1)])

        # Initialise all random numbers state to use for each thread
        rng_states = create_xoroshiro128p_states(batchSize, seed=1)

        # Compute the paths in batches to preserve memory (see if we can
        # exploit both GPUs to share the computational load)

        simulateSinglePath[blockspergrid, threadsperblock](
                batchSize, totalSteps, lookahead, d_sampleFFDIs,
                d_expFiresComp, d_lambdas, d_patchVegetations, d_patchAreas,
                d_patchLocations, d_baseLocations, d_tankerDists, d_heliDists,
                d_ffdiRanges, d_rocA2PHMeans, d_rocA2PHSDs, d_occurrence,
                d_initSizeM, d_initSizeSD, d_initSuccess, d_extSuccess,
                d_resourceTypes, d_resourceSpeeds, d_maxHours,
                d_configurations, d_configsE, d_configsP, d_baseConfigsMax,
                d_fireConfigsMax, d_thresholds, d_accumulatedDamages,
                d_accumulatedHours, d_fires, d_fireSizes, d_fireLocations,
                d_firePatches, d_aircraftLocations, d_aircraftAssignments,
                rng_states, d_states, d_controls, d_regressionX, d_regressionY,
                d_costs2Go, start, stepSize, method, optimal, static,
                d_expectedTemp)

        cuda.synchronize()

        # Return memory to the host. We unfortunately have to do this all the
        # time due to the batching requirement to prevent excessive memory
        # use on the GPU
        d_accumulatedDamages.copy_to_host(accumulatedDamages[
                batchStart:batchEnd, start:(totalSteps+1), :])
        d_accumulatedHours.copy_to_host(accumulatedHours[
                batchStart:batchEnd, start:(totalSteps+1), :])
        d_fires.copy_to_host(fires[batchStart:batchEnd, start:(totalSteps+1)])
        d_fireSizes.copy_to_host(fireSizes[batchStart:batchEnd,
                                           start:(totalSteps+1), :])
        d_fireLocations.copy_to_host(fireLocations[batchStart:batchEnd,
                                                   start:(totalSteps+1), :])
        d_firePatches.copy_to_host(firePatches[batchStart:batchEnd,
                                               start:(totalSteps+1), :])
        d_aircraftLocations.copy_to_host(aircraftLocations[
                batchStart:batchEnd, start:(totalSteps+1), :])
        d_aircraftAssignments.copy_to_host(aircraftAssignments[
                batchStart:batchEnd, start:(totalSteps+1), :])
        d_controls.copy_to_host(controls[batchStart:batchEnd,
                                         start:(totalSteps+1)])
        d_states.copy_to_host(states[batchStart:batchEnd,
                                     start:(totalSteps+1), :])
        d_costs2Go.copy_to_host(costs2Go[batchStart:batchEnd,
                                         start:(totalSteps+1)])

    d_expectedTemp.copy_to_host(expectedTemp)
    print(expectedTemp[0, :, :])
    print(expectedTemp[1, :, :])
    print(expectedTemp[2, :, :])
    print(expectedTemp[3, :, :])
    print(aircraftAssignments[0, 1, :])

def analyseMCPaths():
    pass
