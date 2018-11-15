# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:35:12 2018

@author: nicholas
"""

import numpy
import math
from numba import cuda, float32, float64, int32
from numba.types import b1
#from pyculib import rand
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_normal_float32
from numba.cuda.random import xoroshiro128p_uniform_float32

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


@cuda.jit
def simulateSinglePath(paths, totalSteps, lookahead, sampleFFDIs, expFiresComp,
                       patchVegetations, patchAreas, patchLocations,
                       baseLocations, tankerDists, heliDists, ffdiRanges,
                       rocA2PHMeans, rocA2PHSDs, occurrence, initSizeM,
                       initSizeSD, initSuccess, resourceTypes, resourceSpeeds,
                       maxHours, configurations, configsE, configsP,
                       baseConfigsMax, fireConfigsMax, thresholds,
                       accumulatedDamages, accumulatedHours, fires, fireSizes,
                       fireLocations, firePatches, aircraftLocations,
                       aircraftAssignments, rng_states, controls, regressionX,
                       regressionY, start, stepSize, optimal):

    global noBases
    global noPatches
    global noAircraft
    global noFiresMax
    global noConfigs
    global noConfE
    global noConfP

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

        for ii in range(noPatches):
            for jj in range(noConfP):
                weightsP[ii, jj] = 0.1

        for tt in range(start, totalSteps):
            expectedFFDI = sampleFFDIs[:, tt:(totalSteps + lookahead + 1)]

            expectedDamageExisting(
                    expectedFFDI, firePatches[path][tt], fires[path][tt],
                    fireSizes[path][tt], patchVegetations, ffdiRanges,
                    rocA2PHMeans, rocA2PHSDs, occurrence, configsE, lookahead,
                    expectedE, rng_states, path, False)

            expectedDamagePotential(
                    expectedFFDI, patchVegetations, patchAreas, ffdiRanges,
                    rocA2PHMeans, occurrence, initSizeM, initSuccess, configsP,
                    tt, lookahead, expectedP)

            if optimal:
                # Compute the optimal control using regressions
                pass
            else:
                control = int(6*xoroshiro128p_uniform_float32(rng_states, path))

            # AssignAircraft
            # This could potentially be very slow. Just use a naive assignment
            # for now
            assignAircraft(aircraftAssignments, resourceSpeeds, resourceTypes,
                           maxHours, aircraftLocations, accumulatedHours,
                           baseLocations, tankerDists, heliDists, fires,
                           fireSizes, fireLocations, ffdiRanges, configurations,
                           configsE, configsP, baseConfigsMax, fireConfigsMax,
                           thresholds, sampleFFDIs, expFiresComp, expectedE,
                           expectedP, weightsP, tt, stepSize, lookahead, path,
                           control)

            # SimulateNextStep
            simulateNextStep(aircraftAssignments, resourceTypes, resourceSpeeds,
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
def expectedDamageExisting(ffdi_path, fire_patches, no_fires, fire_sizes,
                           patch_vegetations, ffdi_ranges, roc_a2_ph_means,
                           roc_a2_ph_sds, occurrence, configs, lookahead,
                           expected, rng_states, thread_id, random):

    for fire in range(no_fires):
        patch = int(fire_patches[fire])
        vegetation = int(patch_vegetations[patch])
        ffdi_range = ffdi_ranges[vegetation]
        roc_a2_ph_mean = roc_a2_ph_means[vegetation]
        roc_a2_ph_sd = roc_a2_ph_sds[vegetation]

        for c in range(len(configs)):
            size = fire_sizes[fire]
            config = configs[c]

            for tt in range(lookahead):
                ffdi = ffdi_path[patch, tt]
                size = growFire(ffdi, config - 1, ffdi_range, roc_a2_ph_mean,
                                roc_a2_ph_sd, size, rng_states, thread_id,
                                random)

            expected[fire, c] = max(0, size - fire_sizes[fire])

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

        for c in range(len(configs)):
            damage = 0
            config = configs[c]

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

                expected[patch, c] = damage*patch_areas[patch]

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

@cuda.jit(device=True)
def getAllConfigsSorted(configurations, configsP, baseConfigsPossible,
                        expectedP, numbers):

    tempSortList = cuda.local.array(noConfP, dtype=int32)

    # Collect
    found = 0
    for config in configsP:
        viable = True

        for c in range(4):
            if configurations[config][c] < numbers[c]:
                viable = False
                break

        if viable:
            tempSortList[found] = config
            found += 1

    # Simple selection sort for now as the list will likely be very small
    for i in range(found):
        iMin = i
        for j in range(i, found):
            if expectedP[j] < expectedP[i]:
                iMin = j

        baseConfigsPossible[i] = iMin + 1

@cuda.jit(device=True)
def maxComponentNo():
    pass

@cuda.jit(device=True)
def configWeight():
    pass

#@cuda.jit(device=True)
#def simulateNextStep(aircraftAssignments, aircraftTypes, aircraftSpeeds,
#                     aircraftLocations, accumulatedHours, baseLocations,
#                     fires, patchVegetations, patchLocations, ffdiRanges,
#                     fireLocations, firePatches, sampleFFDIs, rocA2PHMeans,
#                     rocA2PHSDs, fireSizes, configsE, configsP, weightsP):
#    pass

@cuda.jit(device=True)
def simulateNextStep(aircraftAssignments, aircraftTypes, aircraftSpeeds,
                     aircraftLocations, accumulatedHours, baseLocations,
                     noFires, patchVegetations, patchLocations, ffdiRanges,
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
                   thresholds, sampleFFDIs, expFiresComp, expectedE, expectedP,
                   weightsP, time, stepSize, lookahead, thread_id, control):

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
    baseCumulativeSavings = cuda.local.array(noBases, dtype=float32)
    nearest = cuda.local.array(noAircraft, dtype=float32)

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
        baseCumulativeSavings[base] = 0

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
    if control == 0:
        maxFire = thresholds[1]
        maxBase = 0
    elif control == 1:
        maxFire = math.inf
        maxBase = 0
    elif control == 2:
        maxFire = thresholds[1]
        maxBase = thresholds[0]
    elif control == 3:
        maxFire = math.inf
        maxBase = thresholds[0]
    elif control == 4:
        maxFire = thresholds[1]
        maxBase = math.inf
    elif control == 5:
        maxFire = math.inf
        maxBase = math.inf

    """ Pre calcs """
    # Nearest base to each aircraft
    for resource in range(len(resourceTypes)):
        nearest[resource] = 0
        nearestDist = math.inf

        for base in range(len(baseLocations)):
            dist = math.sqrt(((aircraftLocations[thread_id][time][resource][0]
                             - baseLocations[base][0])*40000*math.cos(
                                 (aircraftLocations[thread_id][time][resource][1]
                                  + baseLocations[base][1])
                                 * math.pi/360)/360) ** 2
                             + ((baseLocations[base][1]
                                 - aircraftLocations[thread_id][time][resource][1])*
                                 40000/360)**2)
            if dist < nearestDist:
                nearestDist = dist
                nearest[resource] = base

    # Possible aircraft to base assignments based on control
    for resource in range(len(resourceTypes)):
        for base in range(len(baseLocations)):
            dist = math.sqrt(((aircraftLocations[thread_id][time][resource][0]
                             - baseLocations[base][0])*40000*math.cos(
                                 (aircraftLocations[thread_id][time][resource][1]
                                  + baseLocations[base][1])
                                 * math.pi/360)/360) ** 2
                             + ((baseLocations[base][1]
                                 - aircraftLocations[thread_id][time][resource][1])*
                                 40000/360)**2)
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
            dist = math.sqrt(((aircraftLocations[thread_id][time][resource][0]
                             - fireLocations[thread_id][time][fire][0])*
                             40000*math.cos(
                                 (aircraftLocations[thread_id][time][resource][1]
                                  + fireLocations[thread_id][time][fire][1])
                                 * math.pi/360)/360) ** 2
                             + ((fireLocations[thread_id][time][fire][1]
                                 - aircraftLocations[thread_id][time][resource][1])*
                                 40000/360)**2)
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

        if remaining == noAircraft:
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

                    fireImproveTankerE[fire] = (expectedE[fire][0]
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

                    fireImproveTankerL[fire] = (expectedE[fire][0]
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

                    fireImproveHeliE[fire] = (expectedE[fire][0]
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

                    fireImproveHeliL[fire] = (expectedE[fire][0]
                        - expectedE[fire][config])

                    if fireImproveHeliL[fire] > thisUpdateImprovement:
                        thisUpdateImprovement = fireImproveHeliL[fire]
                        thisUpdateDType = 0
                        thisUpdateACType = 4
                        thisUpdateIdx = fire

                else:
                    fireImproveHeliL[fire] = 0

            for base in range(noBases):
                baseTankersE[fire] = 0
                baseTankersL[fire] = 0
                baseHelisE[fire] = 0
                baseHelisL[fire] = 0
                # We ignore early and late here and assume all aircraft
                # lateness/earliness are just a function of the distance
                # between the base and patch

                # Tanker
                if baseMaxTankersE[base] > 0:
                    expectedDamageBase = 0.0
                    baseDamageBaseline = 0.0

                    for patch in range(noPatches):
                        if tankerCovers[patch, base] <= min(maxFire, 1/3):
                            configNos[0] = 1
                            configNos[1] = 0
                            configNos[2] = 0
                            configNos[3] = 0

                            getAllConfigsSorted(configurations, configsE,
                                                baseConfigsPossible,
                                                expectedE[fire], configNos)
                            config = baseConfigsPossible[0]

                            # If the patch is close to the base
                            expFires = 0.0

                            for tt in range(time, time + lookahead):
                                expFires += expFiresComp[tt][base][0]

                            n_c = max(1, expFires)
                            weight_c = 1/n_c

                            expectedDamageBase += (
                                    expectedP[patch, 0] * (1 - weight_c) -
                                    expectedP[patch, config] * weight_c)

                            baseDamageBaseline += expectedP[patch, 0]

                        elif tankerCovers[patch, base] <= maxFire:
                            configNos[0] = 0
                            configNos[1] = 0
                            configNos[2] = 1
                            configNos[3] = 0

                            getAllConfigsSorted(configurations, configsE,
                                                baseConfigsPossible,
                                                expectedE[fire], configNos)
                            config = baseConfigsPossible[0]

                            # If the patch is close to the base
                            expFires = 0.0

                            for tt in range(time, time + lookahead):
                                expFires += (expFiresComp[tt][base][0]
                                             + expFiresComp[tt][base][2])

                            n_c = max(1, expFires)
                            weight_c = 1/n_c

                            expectedDamageBase += (
                                    expectedP[patch, 0] * (1 - weight_c) -
                                    expectedP[patch, config] * weight_c)

                            baseDamageBaseline += expectedP[patch, 0]

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

                            getAllConfigsSorted(configurations, configsE,
                                                baseConfigsPossible,
                                                expectedE[fire], configNos)
                            config = baseConfigsPossible[0]

                            # If the patch is close to the base
                            expFires = 0.0

                            for tt in range(time, time + lookahead):
                                expFires += expFiresComp[tt][base][1]

                            n_c = max(1, expFires)
                            weight_c = 1/n_c

                            expectedDamageBase += (
                                    expectedP[patch, 0] * (1 - weight_c) -
                                    expectedP[patch, config] * weight_c)

                            baseDamageBaseline += expectedP[patch, 0]

                        elif heliCovers[patch, base] <= maxFire:
                            configNos[0] = 0
                            configNos[1] = 0
                            configNos[2] = 0
                            configNos[3] = 1

                            getAllConfigsSorted(configurations, configsE,
                                                baseConfigsPossible,
                                                expectedE[fire], configNos)
                            config = baseConfigsPossible[0]

                            # If the patch is close to the base
                            expFires = 0.0

                            for tt in range(time, time + lookahead):
                                expFires += (expFiresComp[tt][base][1]
                                             + expFiresComp[tt][base][3])

                            n_c = max(1, expFires)
                            weight_c = 1/n_c

                            expectedDamageBase += (
                                    expectedP[patch, 0] * (1 - weight_c) -
                                    expectedP[patch, config] * weight_c)

                            baseDamageBaseline += expectedP[patch, 0]

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
            if lastUpdateDType == 0:
                # Fire
                # Current benefit
                currentImprove = fireCumulativeSavings[fire]

                # 1 more early tanker
                if fireMaxTankersE[thisUpdateIdx] > fireTankersE[thisUpdateIdx]:
                    configNos[0] = fireTankersE[thisUpdateIdx] + 1
                    configNos[1] = fireHelisE[thisUpdateIdx]
                    configNos[2] = fireTankersL[thisUpdateIdx]
                    configNos[3] = fireHelisL[thisUpdateIdx]

                    getAllConfigsSorted(configurations, configsE,
                                        baseConfigsPossible,
                                        expectedE[fire], configNos)
                    config = baseConfigsPossible[0]

                    fireImproveTankerE[thisUpdateIdx] = (
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
                                        expectedE[fire], configNos)
                    config = baseConfigsPossible[0]

                    fireImproveHeliE[thisUpdateIdx] = (
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
                                        expectedE[fire], configNos)
                    config = baseConfigsPossible[0]

                    fireImproveTankerL[thisUpdateIdx] = (
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
                                        expectedE[fire], configNos)
                    config = baseConfigsPossible[0]

                    fireImproveHeliL[thisUpdateIdx] = (
                        expectedE[thisUpdateIdx][0] - currentImprove
                        - expectedE[thisUpdateIdx][config])
                else:
                    fireImproveHeliL[thisUpdateIdx] = 0

            elif lastUpdateDType == 1:
                for patch in range(noPatches):
                    updatePatches[patch] = 0

                for base in range(noBases):
                    updateBases[base] = 0

                # Current benefit
                currentImprove = baseCumulativeSavings[thisUpdateIdx]

                # First recalculate the base that was just updated
                # An extra tanker ################
                if baseMaxTankersE[thisUpdateIdx] > baseTankersE[thisUpdateIdx]:
                    expectedDamageBase = 0.0
                    baseDamageBaseline = 0.0

                    # Calculate the improvement of one more tanker at this
                    # base for each patch
                    for patch in range(noPatches):
                        if tankerCovers[patch, thisUpdateIdx] <= min(
                            maxFire, 1/3):

                            updatePatches[patch] = True

                            weight_total = 0

                            for c in range(4):
                                configNos[c] = 0

                                configNos[0] = 1

                            # Get best possible config
                            for base in range(noBases):
                                if tankerCovers[patch, base] <= min(
                                    maxFire, 1/3):

                                    updateBases[base] = True
                                    configNos[0] += baseTankersE[base]

                                elif tankerCovers[patch, base] <= maxFire:

                                    updateBases[base] = True
                                    configNos[1] += baseTankersE[base]

                                if heliCovers[patch, base] <= min(
                                    maxFire, 1/3):

                                    updateBases[base] = True
                                    configNos[2] += baseHelisE[base]

                                elif heliCovers[patch, base] <= maxFire:

                                    updateBases[base] = True
                                    configNos[3] += baseHelisE[base]

                            # Get all possible configs and sort them by
                            # benefit to this patch
                            getAllConfigsSorted(configurations, configsP,
                                                baseConfigsPossible,
                                                expectedP[patch], configNos)

                            configIdx = 0

                            while weight_total < 1:
                                config = baseConfigsPossible[configIdx]

                                # Expected fires per component

                                # Early tankers
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][0]

                                n_c = max(1, expFires)
                                weight_c = (configNos[0] / n_c /
                                            configurations[config][0])

                                # Early helis
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][1]

                                n_c = max(1, expFires)
                                weight_c = max(weight_c,
                                               (configNos[1] / n_c /
                                                configurations[config][1]))

                                # Late tankers
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][2]

                                n_c = max(1, expFires)
                                weight_c = max(weight_c,
                                               (configNos[2] / n_c /
                                                configurations[config][2]))

                                # Late helis
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][3]

                                n_c = max(1, expFires)
                                weight_c = max(weight_c,
                                               (configNos[3] / n_c /
                                                configurations[config][3]))

                                weight_c = min(weight_c, 1 - weight_total)

                                expectedDamageBase += (
                                    expectedP[patch, config] *
                                    weight_c)

                                weight_total += weight_c
                                configIdx += 1

                            baseDamageBaseline += expectedP[patch, 0]

                        elif tankerCovers[patch, thisUpdateIdx] <= maxFire:
                            updatePatches[patch] = True

                            weight_total = 0
                            for c in range(4):
                                configNos[c] = 0

                                configNos[2] = 1

                            # Get best possible config
                            for base in range(noBases):
                                if tankerCovers[patch, base] <= min(
                                    maxFire, 1/3):

                                    updateBases[base] = True
                                    configNos[0] += baseTankersE[base]

                                elif tankerCovers[patch, base] <= maxFire:

                                    updateBases[base] = True
                                    configNos[1] += baseTankersE[base]

                                if heliCovers[patch, base] <= min(
                                    maxFire, 1/3):

                                    updateBases[base] = True
                                    configNos[2] += baseHelisE[base]

                                elif heliCovers[patch, base] <= maxFire:

                                    updateBases[base] = True
                                    configNos[3] += baseHelisE[base]

                            # Get all possible configs and sort them by
                            # benefit to this patch
                            getAllConfigsSorted(configurations, configsP,
                                                baseConfigsPossible,
                                                expectedP[patch], configNos)

                            configIdx = 0

                            while weight_total < 1:
                                config = baseConfigsPossible[configIdx]

                                # Expected fires per component

                                # Early tankers
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][0]

                                n_c = max(1, expFires)
                                weight_c = (configNos[0] / n_c /
                                            configurations[config][0])

                                # Early helis
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][1]

                                n_c = max(1, expFires)
                                weight_c = max(weight_c,
                                               (configNos[1] / n_c /
                                                configurations[config][1]))

                                # Late tankers
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][2]

                                n_c = max(1, expFires)
                                weight_c = max(weight_c,
                                               (configNos[2] / n_c /
                                                configurations[config][2]))

                                # Late helis
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][3]

                                n_c = max(1, expFires)
                                weight_c = max(weight_c,
                                               (configNos[3] / n_c /
                                                configurations[config][3]))

                                weight_c = min(weight_c, 1 - weight_total)

                                expectedDamageBase += (
                                    expectedP[patch, config] *
                                    weight_c)

                                weight_total += weight_c
                                configIdx += 1

                            baseDamageBaseline += expectedP[patch, 0]

                    baseImproveTankerE[thisUpdateIdx] = (baseDamageBaseline
                        - currentImprove
                        - expectedDamageBase)
                else:
                    baseImproveTankerE[thisUpdateIdx] = 0.0

                # An extra helicopter ############
                if baseMaxTankersE[thisUpdateIdx] > baseTankersE[thisUpdateIdx]:
                    expectedDamageBase = 0.0
                    baseDamageBaseline = 0.0

                    # Calculate the improvement of one more tanker at this
                    # base for each patch
                    for patch in range(noPatches):
                        if heliCovers[patch, thisUpdateIdx] <= min(
                            maxFire, 1/3):

                            updatePatches[patch] = True

                            weight_total = 0
                            for c in range(4):
                                configNos[c] = 0

                                configNos[1] = 1

                            # Get best possible config
                            for base in range(noBases):
                                if tankerCovers[patch, base] <= min(
                                    maxFire, 1/3):

                                    updateBases[base] = True
                                    configNos[0] += baseTankersE[base]

                                elif tankerCovers[patch, base] <= maxFire:

                                    updateBases[base] = True
                                    configNos[1] += baseTankersE[base]

                                if heliCovers[patch, base] <= min(
                                    maxFire, 1/3):

                                    configNos[2] += baseHelisE[base]

                                elif heliCovers[patch, base] <= maxFire:

                                    configNos[3] += baseHelisE[base]

                            # Get all possible configs and sort them by
                            # benefit to this patch
                            getAllConfigsSorted(configurations, configsP,
                                                baseConfigsPossible,
                                                expectedP[patch], configNos)

                            configIdx = 0

                            while weight_total < 1:
                                config = baseConfigsPossible[configIdx]

                                # Expected fires per component

                                # Early tankers
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][0]

                                n_c = max(1, expFires)
                                weight_c = (configNos[0] / n_c /
                                            configurations[config][0])

                                # Early helis
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][1]

                                n_c = max(1, expFires)
                                weight_c = max(weight_c,
                                               (configNos[1] / n_c /
                                                configurations[config][1]))

                                # Late tankers
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][2]

                                n_c = max(1, expFires)
                                weight_c = max(weight_c,
                                               (configNos[2] / n_c /
                                                configurations[config][2]))

                                # Late helis
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][3]

                                n_c = max(1, expFires)
                                weight_c = max(weight_c,
                                               (configNos[3] / n_c /
                                                configurations[config][3]))

                                weight_c = min(weight_c, 1 - weight_total)

                                expectedDamageBase += (
                                    expectedP[patch, config] *
                                    weight_c)

                                weight_total += weight_c
                                configIdx += 1

                            baseDamageBaseline += expectedP[patch, 0]

                        elif heliCovers[patch, thisUpdateIdx] <= maxFire:
                            updatePatches[patch] = True

                            weight_total = 0
                            for c in range(4):
                                configNos[c] = 0

                                configNos[3] = 1

                            # Get best possible config
                            for base in range(noBases):
                                if tankerCovers[patch, base] <= min(
                                    maxFire, 1/3):

                                    updateBases[base] = True
                                    configNos[0] += baseTankersE[base]

                                elif tankerCovers[patch, base] <= maxFire:

                                    updateBases[base] = True
                                    configNos[1] += baseTankersE[base]

                                if heliCovers[patch, base] <= min(
                                    maxFire, 1/3):

                                    configNos[2] += baseHelisE[base]

                                elif heliCovers[patch, base] <= maxFire:

                                    configNos[3] += baseHelisE[base]

                            # Get all possible configs and sort them by
                            # benefit to this patch
                            getAllConfigsSorted(configurations, configsP,
                                                baseConfigsPossible,
                                                expectedP[patch], configNos)

                            configIdx = 0

                            while weight_total < 1:
                                config = baseConfigsPossible[configIdx]

                                # Expected fires per component

                                # Early tankers
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][0]

                                n_c = max(1, expFires)
                                weight_c = (configNos[0] / n_c /
                                            configurations[config][0])

                                # Early helis
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][1]

                                n_c = max(1, expFires)
                                weight_c = max(weight_c,
                                               (configNos[1] / n_c /
                                                configurations[config][1]))

                                # Late tankers
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][2]

                                n_c = max(1, expFires)
                                weight_c = max(weight_c,
                                               (configNos[2] / n_c /
                                                configurations[config][2]))

                                # Late helis
                                expFires = 0.0

                                for tt in range(time, time + lookahead):
                                    expFires += expFiresComp[time][base][3]

                                n_c = max(1, expFires)
                                weight_c = max(weight_c,
                                               (configNos[3] / n_c /
                                                configurations[config][3]))

                                weight_c = min(weight_c, 1 - weight_total)

                                expectedDamageBase += (
                                    expectedP[patch, config] *
                                    weight_c)

                                weight_total += weight_c
                                configIdx += 1

                            baseDamageBaseline += expectedP[patch, 0]

                    baseImproveHeliE[thisUpdateIdx] = (baseDamageBaseline
                        - currentImprove
                        - expectedDamageBase)
                else:
                    baseImproveHeliE[thisUpdateIdx] = 0.0

                # Bases that are affected by adjusted patches
                # Maybe we can get away with avoiding this step for now
                for baseOther in range(noBases):
                    if updateBases[baseOther]:

                        # First recalculate the base that was just updated
                        # An extra tanker ################
                        if baseMaxTankersE[baseOther] > baseTankersE[baseOther]:
                            expectedDamageBase = 0.0
                            baseDamageBaseline = 0.0

                            # Calculate the improvement of one more tanker at this
                            # base for each patch
                            for patch in range(noPatches):
                                if tankerCovers[patch, baseOther] <= min(
                                    maxFire, 1/3):

                                    weight_total = 0
                                    for c in range(4):
                                        configNos[c] = 0

                                        configNos[0] = 1

                                    # Get best possible config
                                    for base in range(noBases):
                                        if tankerCovers[patch, base] <= min(
                                            maxFire, 1/3):

                                            configNos[0] += baseTankersE[base]

                                        elif tankerCovers[patch, base] <= maxFire:

                                            configNos[1] += baseTankersE[base]

                                        if heliCovers[patch, base] <= min(
                                            maxFire, 1/3):

                                            configNos[2] += baseHelisE[base]

                                        elif heliCovers[patch, base] <= maxFire:

                                            configNos[3] += baseHelisE[base]

                                    # Get all possible configs and sort them by
                                    # benefit to this patch
                                    getAllConfigsSorted(configurations, configsP,
                                                        baseConfigsPossible,
                                                        expectedP[patch], configNos)

                                    configIdx = 0

                                    while weight_total < 1:
                                        config = baseConfigsPossible[configIdx]

                                        # Expected fires per component

                                        # Early tankers
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][0]

                                        n_c = max(1, expFires)
                                        weight_c = (configNos[0] / n_c /
                                                    configurations[config][0])

                                        # Early helis
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][1]

                                        n_c = max(1, expFires)
                                        weight_c = max(weight_c,
                                                       (configNos[1] / n_c /
                                                        configurations[config][1]))

                                        # Late tankers
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][2]

                                        n_c = max(1, expFires)
                                        weight_c = max(weight_c,
                                                       (configNos[2] / n_c /
                                                        configurations[config][2]))

                                        # Late helis
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][3]

                                        n_c = max(1, expFires)
                                        weight_c = max(weight_c,
                                                       (configNos[2] / n_c /
                                                        configurations[config][3]))

                                        weight_c = min(weight_c, 1 - weight_total)

                                        expectedDamageBase += (
                                            expectedP[patch, config] *
                                            weight_c)

                                        weight_total += weight_c
                                        configIdx += 1

                                    baseDamageBaseline += expectedP[patch, 0]

                                elif tankerCovers[patch, baseOther] <= maxFire:

                                    weight_total = 0
                                    for c in range(4):
                                        configNos[c] = 0

                                        configNos[2] = 1

                                    # Get best possible config
                                    for base in range(noBases):
                                        if tankerCovers[patch, base] <= min(
                                            maxFire, 1/3):

                                            configNos[0] += baseTankersE[base]

                                        elif tankerCovers[patch, base] <= maxFire:

                                            configNos[1] += baseTankersE[base]

                                        if heliCovers[patch, base] <= min(
                                            maxFire, 1/3):

                                            configNos[2] += baseHelisE[base]

                                        elif heliCovers[patch, base] <= maxFire:

                                            configNos[3] += baseHelisE[base]

                                    # Get all possible configs and sort them by
                                    # benefit to this patch
                                    getAllConfigsSorted(configurations, configsP,
                                                        baseConfigsPossible,
                                                        expectedP[patch], configNos)

                                    configIdx = 0

                                    while weight_total < 1:
                                        config = baseConfigsPossible[configIdx]

                                        # Expected fires per component

                                        # Early tankers
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][0]

                                        n_c = max(1, expFires)
                                        weight_c = (configNos[0] / n_c /
                                                    configurations[config][0])

                                        # Early helis
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][1]

                                        n_c = max(1, expFires)
                                        weight_c = max(weight_c,
                                                       (configNos[1] / n_c /
                                                        configurations[config][1]))

                                        # Late tankers
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][2]

                                        n_c = max(1, expFires)
                                        weight_c = max(weight_c,
                                                       (configNos[2] / n_c /
                                                        configurations[config][2]))

                                        # Late helis
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][3]

                                        n_c = max(1, expFires)
                                        weight_c = max(weight_c,
                                                       (configNos[3] / n_c /
                                                        configurations[config][3]))

                                        weight_c = min(weight_c, 1 - weight_total)

                                        expectedDamageBase += (
                                            expectedP[patch, config] *
                                            weight_c)

                                        weight_total += weight_c
                                        configIdx += 1

                                    baseDamageBaseline += expectedP[patch, 0]

                            baseImproveTankerE[baseOther] = (baseDamageBaseline
                                - currentImprove
                                - expectedDamageBase)
                        else:
                            baseImproveTankerE[baseOther] = 0.0

                        # An extra helicopter ############
                        if baseMaxTankersE[baseOther] > baseTankersE[baseOther]:
                            expectedDamageBase = 0.0
                            baseDamageBaseline = 0.0

                            # Calculate the improvement of one more tanker at this
                            # base for each patch
                            for patch in range(noPatches):
                                if heliCovers[patch, baseOther] <= min(
                                    maxFire, 1/3):

                                    updatePatches[patch] = True

                                    weight_total = 0
                                    for c in range(4):
                                        configNos[c] = 0

                                        configNos[1] = 1

                                    # Get best possible config
                                    for base in range(noBases):
                                        if tankerCovers[patch, base] <= min(
                                            maxFire, 1/3):

                                            configNos[0] += baseTankersE[base]

                                        elif tankerCovers[patch, base] <= maxFire:

                                            configNos[1] += baseTankersE[base]

                                        if heliCovers[patch, base] <= min(
                                            maxFire, 1/3):

                                            configNos[2] += baseHelisE[base]

                                        elif heliCovers[patch, base] <= maxFire:

                                            configNos[3] += baseHelisE[base]

                                    # Get all possible configs and sort them by
                                    # benefit to this patch
                                    getAllConfigsSorted(configurations, configsP,
                                                        baseConfigsPossible,
                                                        expectedP[patch], configNos)

                                    configIdx = 0

                                    while weight_total < 1:
                                        config = baseConfigsPossible[configIdx]

                                        # Expected fires per component

                                        # Early tankers
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][0]

                                        n_c = max(1, expFires)
                                        weight_c = (configNos[0] / n_c /
                                                    configurations[config][0])

                                        # Early helis
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][1]

                                        n_c = max(1, expFires)
                                        weight_c = max(weight_c,
                                                       (configNos[1] / n_c /
                                                        configurations[config][1]))

                                        # Late tankers
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][2]

                                        n_c = max(1, expFires)
                                        weight_c = max(weight_c,
                                                       (configNos[2] / n_c /
                                                        configurations[config][2]))

                                        # Late helis
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][3]

                                        n_c = max(1, expFires)
                                        weight_c = max(weight_c,
                                                       (configNos[3] / n_c /
                                                        configurations[config][3]))

                                        weight_c = min(weight_c, 1 - weight_total)

                                        expectedDamageBase += (
                                            expectedP[patch, config] *
                                            weight_c)

                                        weight_total += weight_c
                                        configIdx += 1

                                    baseDamageBaseline += expectedP[patch, 0]

                                elif heliCovers[patch, baseOther] <= maxFire:

                                    weight_total = 0
                                    for c in range(4):
                                        configNos[c] = 0

                                        configNos[3] = 1

                                    # Get best possible config
                                    for base in range(noBases):
                                        if tankerCovers[patch, base] <= min(
                                            maxFire, 1/3):

                                            configNos[0] += baseTankersE[base]

                                        elif tankerCovers[patch, base] <= maxFire:

                                            configNos[1] += baseTankersE[base]

                                        if heliCovers[patch, base] <= min(
                                            maxFire, 1/3):

                                            configNos[2] += baseHelisE[base]

                                        elif heliCovers[patch, base] <= maxFire:

                                            configNos[3] += baseHelisE[base]

                                    # Get all possible configs and sort them by
                                    # benefit to this patch
                                    getAllConfigsSorted(configurations, configsP,
                                                        baseConfigsPossible,
                                                        expectedP[patch], configNos)

                                    configIdx = 0

                                    while weight_total < 1:
                                        config = baseConfigsPossible[configIdx]

                                        # Expected fires per component

                                        # Early tankers
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][0]

                                        n_c = max(1, expFires)
                                        weight_c = (configNos[0] / n_c /
                                                    configurations[config][0])

                                        # Early helis
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][1]

                                        n_c = max(1, expFires)
                                        weight_c = max(weight_c,
                                                       (configNos[1] / n_c /
                                                        configurations[config][1]))

                                        # Late tankers
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][2]

                                        n_c = max(1, expFires)
                                        weight_c = max(weight_c,
                                                       (configNos[2] / n_c /
                                                        configurations[config][2]))

                                        # Late helis
                                        expFires = 0.0

                                        for tt in range(time, time + lookahead):
                                            expFires += expFiresComp[time][base][3]

                                        n_c = max(1, expFires)
                                        weight_c = max(weight_c,
                                                       (configNos[3] / n_c /
                                                        configurations[config][3]))

                                        weight_c = min(weight_c, 1 - weight_total)

                                        expectedDamageBase += (
                                            expectedP[patch, config] *
                                            weight_c)

                                        weight_total += weight_c
                                        configIdx += 1

                                    baseDamageBaseline += expectedP[patch, 0]

                            baseImproveHeliE[baseOther] = (baseDamageBaseline
                                - currentImprove
                                - expectedDamageBase)
                        else:
                            baseImproveHeliE[baseOther] = 0.0

            # Pick the best assignment ########################################
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
        if thisUpdateDType == 0:
            if thisUpdateACType == 1:
                fireTankersE[thisUpdateIdx] += 1
                fireCumulativeSavings[thisUpdateIdx] += (
                    fireImproveTankerE[thisUpdateIdx])

            elif thisUpdateACType == 2:
                fireHelisE[thisUpdateIdx] += 1
                fireCumulativeSavings[thisUpdateIdx] += (
                    fireImproveHeliE[thisUpdateIdx])

            elif thisUpdateACType == 3:
                fireTankersL[thisUpdateIdx] += 1
                fireCumulativeSavings[thisUpdateIdx] += (
                    fireImproveTankerL[thisUpdateIdx])

            elif thisUpdateACType == 4:
                fireHelisL[thisUpdateIdx] += 1
                fireCumulativeSavings[thisUpdateIdx] += (
                    fireImproveHeliL[thisUpdateIdx])

        elif thisUpdateDType == 1:
            if thisUpdateACType == 1:
                baseTankersE[thisUpdateIdx] += 1
                baseCumulativeSavings[thisUpdateIdx] += (
                    baseImproveTankerE[thisUpdateIdx])

            elif thisUpdateACType == 2:
                baseHelisE[thisUpdateIdx] += 1
                baseCumulativeSavings[thisUpdateIdx] += (
                    baseImproveHeliE[thisUpdateIdx])

        # For given assignment, pick best A/C #################################
        minMax = math.inf
        nextAC = 0
        remaining -= 1

        # Next assignment is to a fire
        if thisUpdateDType == 0:
            if thisUpdateACType == 1:
                for resource in range(noAircraft):
                    nextBest = 0

                    if (resourceTypes[resource] == 0
                        and aircraftAssignments[thread_id][time][resource][0] == 0):

                        if canFire[resource, thisUpdateIdx]:
                            dist = math.sqrt(((
                                aircraftLocations[thread_id][time][resource][0]
                                - fireLocations[thread_id][time][thisUpdateIdx][
                                0])*40000*math.cos(
                                    (aircraftLocations[thread_id][time][resource][1]
                                     + fireLocations[thread_id][time][
                                     thisUpdateIdx][1]) * math.pi/360)/360)
                                     ** 2
                                + ((fireLocations[thread_id][time][
                                    thisUpdateIdx][1] - aircraftLocations[
                                    thread_id][time][resource][1])*
                                   40000/360)**2)
                            travTime = dist / resourceSpeeds[resource]

                            if travTime <= 1/3:
                                # This resource is a candidate

                                # Check potential benefit to other fires
                                for fire in range(noFires[thread_id][time]):
                                    if (fire != thisUpdateIdx
                                        and canFire[resource, fire]):

                                        dist = math.sqrt(((
                                            aircraftLocations[thread_id][time][resource][0]
                                            - fireLocations[thread_id][time][
                                            fire][0])*40000*math.cos(
                                                (aircraftLocations[thread_id][
                                                time][resource][1] + fireLocations[
                                                thread_id][time][fire][
                                                1]) * math.pi/360)/360) ** 2
                                            + ((fireLocations[thread_id][time][
                                                fire][1] - aircraftLocations[thread_id][
                                                resource][time][1])*40000/360)
                                                **2)
                                        travTime = (dist /
                                            resourceSpeeds[resource])

                                        if travTime < 1/3:
                                            if (fireImproveTankerE[fire] >
                                                nextBest):

                                                nextBest = (
                                                    fireImproveTankerE[fire])
                                        else:
                                            if (fireImproveTankerL[fire] >
                                                nextBest):

                                                nextBest = (
                                                    fireImproveTankerL[fire])

                                # Check potential benefit to bases
                                for base in range(noBases):
                                    if canBase[resource, base]:

                                        if (baseImproveTankerE[base] >
                                            nextBest):

                                            nextBest = (
                                                baseImproveTankerE[base])

                if nextBest < minMax:
                    minMax = nextBest
                    nextAC = resource

            elif thisUpdateACType == 2:
                for resource in range(noAircraft):
                    nextBest = 0

                    if (resourceTypes[resource] == 1
                        and aircraftAssignments[thread_id][time][resource][0] == 0):

                        if canFire[resource, thisUpdateIdx]:
                            dist = math.sqrt(((
                                aircraftLocations[thread_id][time][resource][0]
                                - fireLocations[thread_id][time][thisUpdateIdx][
                                0])*40000*math.cos(
                                    (aircraftLocations[thread_id][time][resource][1]
                                     + fireLocations[thread_id][time][
                                     thisUpdateIdx][1]) * math.pi/360)/360)
                                     ** 2
                                + ((fireLocations[thread_id][time][
                                    thisUpdateIdx][1] - aircraftLocations[
                                    thread_id][time][resource][1])*
                                   40000/360)**2)
                            travTime = dist / resourceSpeeds[resource]

                            if travTime <= 1/3:
                                # This resource is a candidate

                                # Check potential benefit to other fires
                                for fire in range(noFires[thread_id][time]):
                                    if (fire != thisUpdateIdx
                                        and canFire[resource, fire]):

                                        dist = math.sqrt(((
                                            aircraftLocations[thread_id][time][resource][0]
                                            - fireLocations[thread_id][time][
                                            fire][0])*40000*math.cos(
                                                (aircraftLocations[thread_id][
                                                time][resource][1] + fireLocations[
                                                thread_id][time][fire][
                                                1]) * math.pi/360)/360) ** 2
                                            + ((fireLocations[thread_id][time][
                                                fire][1] - aircraftLocations[
                                                thread_id][time][resource][1])*40000/360)
                                                **2)
                                        travTime = (dist /
                                            resourceSpeeds[resource])

                                        if travTime < 1/3:
                                            if (fireImproveHeliE[fire] >
                                                nextBest):

                                                nextBest = (
                                                    fireImproveHeliE[fire])
                                        else:
                                            if (fireImproveHeliL[fire] >
                                                nextBest):

                                                nextBest = (
                                                    fireImproveHeliL[fire])

                                # Check potential benefit to bases
                                for base in range(noBases):
                                    if canBase[resource, base]:

                                        if (baseImproveHeliE[base] >
                                            nextBest):

                                            nextBest = (
                                                baseImproveHeliE[base])

                if nextBest < minMax:
                    minMax = nextBest
                    nextAC = resource

            if thisUpdateACType == 3:
                for resource in range(noAircraft):
                    nextBest = 0

                    if (resourceTypes[resource] == 0
                        and aircraftAssignments[thread_id][time][resource][0] == 0):

                        if canFire[resource, thisUpdateIdx]:
                            dist = math.sqrt(((
                                aircraftLocations[thread_id][time][resource][0]
                                - fireLocations[thread_id][time][thisUpdateIdx][
                                0])*40000*math.cos(
                                    (aircraftLocations[thread_id][time][resource][1]
                                     + fireLocations[thread_id][time][
                                     thisUpdateIdx][1]) * math.pi/360)/360)
                                     ** 2
                                + ((fireLocations[thread_id][time][
                                    thisUpdateIdx][1] - aircraftLocations[
                                    thread_id][time][resource][1])*
                                   40000/360)**2)
                            travTime = dist / resourceSpeeds[resource]

                            if travTime > 1/3:
                                # This resource is a candidate

                                # Check potential benefit to other fires
                                for fire in range(noFires[thread_id][time]):
                                    if (fire != thisUpdateIdx
                                        and canFire[resource, fire]):

                                        dist = math.sqrt(((
                                            aircraftLocations[thread_id][time][resource][0]
                                            - fireLocations[thread_id][time][
                                            fire][0])*40000*math.cos(
                                                (aircraftLocations[thread_id][
                                                time][resource][1] + fireLocations[
                                                thread_id][time][fire][
                                                1]) * math.pi/360)/360) ** 2
                                            + ((fireLocations[thread_id][time][
                                                fire][1] - aircraftLocations[
                                                thread_id][time][resource][1])*40000/360)
                                                **2)
                                        travTime = (dist /
                                            resourceSpeeds[resource])

                                        if travTime < 1/3:
                                            if (fireImproveTankerE[fire] >
                                                nextBest):

                                                nextBest = (
                                                    fireImproveTankerE[fire])
                                        else:
                                            if (fireImproveTankerL[fire] >
                                                nextBest):

                                                nextBest = (
                                                    fireImproveTankerL[fire])

                                # Check potential benefit to bases
                                for base in range(noBases):
                                    if canBase[resource, base]:

                                        if (baseImproveTankerE[base] >
                                            nextBest):

                                            nextBest = (
                                                baseImproveTankerE[base])

                if nextBest < minMax:
                    minMax = nextBest
                    nextAC = resource

            if thisUpdateACType == 4:
                for resource in range(noAircraft):
                    nextBest = 0

                    if (resourceTypes[resource] == 1
                        and aircraftAssignments[thread_id][time][resource][0] == 0):

                        if canFire[resource, thisUpdateIdx]:
                            dist = math.sqrt(((
                                aircraftLocations[thread_id][time][resource][0]
                                - fireLocations[thread_id][time][thisUpdateIdx][
                                0])*40000*math.cos(
                                    (aircraftLocations[thread_id][time][resource][1]
                                     + fireLocations[thread_id][time][
                                     thisUpdateIdx][1]) * math.pi/360)/360)
                                     ** 2
                                + ((fireLocations[thread_id][time][
                                    thisUpdateIdx][1] - aircraftLocations[
                                    thread_id][time][resource][1])*
                                   40000/360)**2)
                            travTime = dist / resourceSpeeds[resource]

                            if travTime > 1/3:
                                # This resource is a candidate

                                # Check potential benefit to other fires
                                for fire in range(noFires[thread_id][time]):
                                    if (fire != thisUpdateIdx
                                        and canFire[resource, fire]):

                                        dist = math.sqrt(((
                                            aircraftLocations[thread_id][time][resource][0]
                                            - fireLocations[thread_id][time][
                                            fire][0])*40000*math.cos(
                                                (aircraftLocations[thread_id][
                                                time][resource][1] + fireLocations[
                                                thread_id][time][fire][
                                                1]) * math.pi/360)/360) ** 2
                                            + ((fireLocations[thread_id][time][
                                                fire][1] - aircraftLocations[
                                                thread_id][time][resource][1])*40000/360)
                                                **2)
                                        travTime = (dist /
                                            resourceSpeeds[resource])

                                        if travTime < 1/3:
                                            if (fireImproveHeliE[fire] >
                                                nextBest):

                                                nextBest = (
                                                    fireImproveHeliE[fire])
                                        else:
                                            if (fireImproveHeliL[fire] >
                                                nextBest):

                                                nextBest = (
                                                    fireImproveHeliL[fire])

                                # Check potential benefit to bases
                                for base in range(noBases):
                                    if canBase[resource, base]:

                                        if (baseImproveHeliE[base] >
                                            nextBest):

                                            nextBest = (
                                                baseImproveHeliE[base])

                if nextBest < minMax:
                    minMax = nextBest
                    nextAC = resource

        # Next assignment is a base
        elif thisUpdateDType == 1:
            if (thisUpdateACType == 1 or thisUpdateACType == 3):
                for resource in range(noAircraft):
                    nextBest = 0

                    if (resourceTypes[resource] == 0
                        and aircraftAssignments[thread_id][time][resource][0] == 0):

                        if canBase[resource, thisUpdateIdx]:
                            # This resource is a candidate

                            # Check potential benefit to other fires
                            for fire in range(noFires[thread_id][time]):
                                if canFire[resource, fire]:

                                    dist = math.sqrt(((
                                        aircraftLocations[thread_id][time][resource][0]
                                        - fireLocations[thread_id][time][
                                        fire][0])*40000*math.cos(
                                            (aircraftLocations[thread_id][
                                            time][resource][1] + fireLocations[
                                            thread_id][time][fire][
                                            1]) * math.pi/360)/360) ** 2
                                        + ((fireLocations[thread_id][time][
                                            fire][1] - aircraftLocations[
                                            thread_id][time][resource][1])*40000/360)
                                            **2)
                                    travTime = (dist /
                                        resourceSpeeds[resource])

                                    if travTime < 1/3:
                                        if (fireImproveTankerE[fire] >
                                            nextBest):

                                            nextBest = (
                                                fireImproveTankerE[fire])
                                    else:
                                        if (fireImproveTankerL[fire] >
                                            nextBest):

                                            nextBest = (
                                                fireImproveTankerL[fire])

                            # Check potential benefit to bases
                            for base in range(noBases):
                                if (base != thisUpdateIdx
                                    and canBase[resource, fire]):

                                    if (baseImproveTankerE[base] >
                                        nextBest):

                                        nextBest = (
                                            baseImproveTankerE[base])

            if nextBest < minMax:
                minMax = nextBest
                nextAC = resource

            if (thisUpdateACType == 2 or thisUpdateACType == 4):
                for resource in range(noAircraft):
                    nextBest = 0

                    if (resourceTypes[resource] == 1
                        and aircraftAssignments[thread_id][time][resource][0] == 0):
                            # This resource is a candidate

                            # Check potential benefit to other fires
                            for fire in range(noFires[thread_id][time]):
                                if canFire[resource, fire]:

                                    dist = math.sqrt(((
                                        aircraftLocations[thread_id][time][resource][0]
                                        - fireLocations[thread_id][time][
                                        fire][0])*40000*math.cos(
                                            (aircraftLocations[thread_id][
                                            time][resource][1] + fireLocations[
                                            thread_id][time][fire][
                                            1]) * math.pi/360)/360) ** 2
                                        + ((fireLocations[thread_id][time][
                                            fire][1] - aircraftLocations[
                                            thread_id][time][resource][1])*40000/360)
                                            **2)
                                    travTime = (dist /
                                        resourceSpeeds[resource])

                                    if travTime < 1/3:
                                        if (fireImproveHeliE[fire] >
                                            nextBest):

                                            nextBest = (
                                                fireImproveHeliE[fire])
                                    else:
                                        if (fireImproveHeliL[fire] >
                                            nextBest):

                                            nextBest = (
                                                fireImproveHeliL[fire])

                            # Check potential benefit to bases
                            for base in range(noBases):
                                if (base != thisUpdateIdx
                                    and canBase[resource, fire]):

                                    if (baseImproveHeliE[base] >
                                        nextBest):

                                        nextBest = (
                                            baseImproveHeliE[base])

                if nextBest < minMax:
                    minMax = nextBest
                    nextAC = resource

        # Update the assignments matrix #######################################
        if thisUpdateDType == 0:
            aircraftAssignments[thread_id][time][nextAC][1] = thisUpdateIdx + 1
            # The following does not matter as it will never be referenced for
            # a calculation. It is only used to determine if the aircraft is
            # available, therefore we just assign a non-zero value to do this
            aircraftAssignments[thread_id][time][nextAC][0] = 1

            # Pick nearest base to fire that is within the maximum relocation
            # distance from the aircraft's current base
            bestBase = 0
            bestTrav = math.inf
            for base in range(noBases):
                if canBase[nextAC, base]:
                    dist = math.sqrt(((aircraftLocations[thread_id][time][nextAC][0]
                             - baseLocations[base][0])*
                             40000*math.cos(
                                 (aircraftLocations[thread_id][nextAC][time][1]
                                  + baseLocations[base][1])
                                 * math.pi/360)/360) ** 2
                             + ((baseLocations[base][1]
                                 - aircraftLocations[thread_id][time][nextAC][1])*
                                 40000/360)**2)
                travTime = dist / resourceSpeeds[nextAC]

                if travTime < bestTrav:
                    bestBase = base
                    bestTrav = travTime

            aircraftAssignments[thread_id][time][nextAC][0] = bestBase + 1

        elif thisUpdateDType == 1:
            aircraftAssignments[thread_id][time][nextAC][1] = 0
            aircraftAssignments[thread_id][time][nextAC][0] = thisUpdateIdx + 1

        #######################################################################
        # Reduce max component capacities (and possibly incremental
        # improvement) based on assignment just made
        for fire in range(noFires[thread_id][time]):
            if canFire[nextAC, fire]:
                dist = math.sqrt(((aircraftLocations[thread_id][time][nextAC][0]
                             - fireLocations[thread_id][time][fire][0])*
                             40000*math.cos(
                                 (aircraftLocations[thread_id][time][nextAC][1]
                                  + fireLocations[thread_id][time][fire][1])
                                 * math.pi/360)/360) ** 2
                             + ((fireLocations[thread_id][time][fire][1]
                                 - aircraftLocations[thread_id][time][nextAC][1])*
                                 40000/360)**2)
                travTime = dist / resourceSpeeds[nextAC]

                if thisUpdateACType == 1 or thisUpdateACType == 3:
                    if (travTime < 1/3):
                        fireMaxTankersE[fire] -= 1

                        if fireMaxTankersE[fire] == fireTankersE[fire]:
                            fireImproveTankerE[fire] = 0

                    else:
                        fireMaxTankersL[fire] -= 1

                        if fireMaxTankersL[fire] == fireTankersL[fire]:
                            fireImproveTankerL[fire] = 0

                elif thisUpdateACType == 2 or thisUpdateACType == 4:
                    if (travTime < 1/3):
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

    # Compute resulting fire configurations and patch configuration weights
    # given these assignments
    for patch in range(noPatches):
        weight_total = 0
        for c in range(4):
            configNos[c] = 0

        # Get best possible config
        for base in range(noBases):
            if tankerCovers[patch, base] <= min(maxFire, 1/3):
                configNos[0] += baseTankersE[base]

            elif tankerCovers[patch, base] <= maxFire:
                configNos[1] += baseTankersE[base]

            if heliCovers[patch, base] <= min(maxFire, 1/3):
                configNos[2] += baseHelisE[base]

            elif heliCovers[patch, base] <= maxFire:
                configNos[3] += baseHelisE[base]

        # Get all possible configs and sort them by
        # benefit to this patch
        getAllConfigsSorted(configurations, configsP, baseConfigsPossible,
                            expectedP[patch], configNos)

        configIdx = 0

        while weight_total < 1:
            config = baseConfigsPossible[configIdx] - 1

            # Expected fires per component

            # Early tankers
            expFires = 0.0

            for tt in range(time, time + lookahead):
                expFires += expFiresComp[time][base][0]

            n_c = max(1, expFires)
            weight_c = (configNos[0] / n_c / configurations[config][0])

            # Early helis
            expFires = 0.0

            for tt in range(time, time + lookahead):
                expFires += expFiresComp[time][base][1]

            n_c = max(1, expFires)
            weight_c = max(weight_c, configNos[1] / n_c / configurations[config][1])

            # Late tankers
            expFires = 0.0

            for tt in range(time, time + lookahead):
                expFires += expFiresComp[time][base][2]

            n_c = max(1, expFires)
            weight_c = max(weight_c, configNos[2] / n_c / configurations[config][2])

            # Late helis
            expFires = 0.0

            for tt in range(time, time + lookahead):
                expFires += expFiresComp[time][base][3]

            n_c = max(1, expFires)
            weight_c = max(weight_c, configNos[3] / n_c / configurations[config][3])

            weight_c = min(weight_c, 1 - weight_total)
            weight_total += weight_c
            configIdx += 1

            weightsP[patch, config] = weight_c

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

""" WRAPPER """
#@jit(parallel=True, fastmath=True)
def simulateMC(paths, sampleFFDIs, patchVegetations, patchAreas,
               patchLocations, baseLocations, resourceTypes, resourceSpeeds,
               maxHours, configurations, configsE, configsP, thresholds,
               ffdiRanges, rocA2PHMeans, rocA2PHSDs, occurrence, initSizeM,
               initSizeSD, initSuccess, totalSteps, lookahead, stepSize,
               accumulatedDamages, accumulatedHours, fires, fireSizes,
               fireLocations, firePatches, aircraftLocations,
               aircraftAssignments, controls, regressionX, regressionY,
               costs2Go, static=False):

    """ Set global values """
    global noBases
    global noPatches
    global noAircraft
    global noFiresMax
    global noConfigs
    global noConfE
    global noConfP

    noBases = len(baseLocations)
    noPatches = len(patchLocations)
    noAircraft = len(resourceTypes)
    noFiresMax = 200
    noConfigs = len(configurations)
    noConfE = len(configsE)
    noConfP = len(configsP)

    tankerDists = numpy.zeros([noBases, noPatches], dtype=numpy.float32)
    heliDists = numpy.zeros([noBases, noPatches], dtype=numpy.float32)

    tankerDists = numpy.array(
        [[math.sqrt(
               ((patchLocations[patch][0] - baseLocations[base][0])*40000*
                   math.cos((patchLocations[patch][1]
                             + baseLocations[base][1]) * math.pi/360)/360) ** 2
               + ((baseLocations[base][1] - patchLocations[patch][1])*40000
               /360)**2) / resourceSpeeds[0]
          for base in range(noBases)]
         for patch in range(noPatches)])

    heliDists = numpy.array(
        [[math.sqrt(
               ((patchLocations[patch][0] - baseLocations[base][0])*40000*
                   math.cos((patchLocations[patch][1]
                             + baseLocations[base][1]) * math.pi/360)/360) ** 2
               + ((baseLocations[base][1] - patchLocations[patch][1])*40000
               /360)**2) / resourceSpeeds[1]
          for base in range(noBases)]
         for patch in range(noPatches)])

    tankerCovers = numpy.array(
        [[True if tankerDists[patch, base] <= thresholds[0] else False
          for base in range(noBases)]
         for patch in range(noPatches)])

    heliCovers = numpy.array(
        [[True if heliDists[patch, base] <= thresholds[0] else False
          for base in range(noBases)]
         for patch in range(noPatches)])

    batches = math.ceil(paths / 1000)
    batchAmounts = [1000 for batch in range(batches - 1)]
    batchAmounts.append(paths - sum(batchAmounts))

    # Absoloute maximum number of aircraft of each config component allowed
    baseConfigsMax = numpy.array([0, 0, 0, 0], dtype=numpy.int32)
    fireConfigsMax = numpy.array([0, 0, 0, 0], dtype=numpy.int32)

    # Maximum number in each component based on allowed configs
    for c in range(len(configurations)):
        if c+1 in configsP:
            if configsP[0] > baseConfigsMax[0]:
                baseConfigsMax[0] += 1
            if configsP[1] > baseConfigsMax[1]:
                baseConfigsMax[1] += 1
            if configsP[2] > baseConfigsMax[2]:
                baseConfigsMax[2] += 1
            if configsP[3] > baseConfigsMax[3]:
                baseConfigsMax[3] += 1

        if c+1 in configsE:
            if configsE[0] > fireConfigsMax[0]:
                fireConfigsMax[0] += 1
            if configsE[1] > fireConfigsMax[0]:
                fireConfigsMax[1] += 1
            if configsE[2] > fireConfigsMax[0]:
                fireConfigsMax[2] += 1
            if configsE[3] > fireConfigsMax[0]:
                fireConfigsMax[3] += 1

    # Expected fires for each patch at each time step
    expectedFires = numpy.array([[
            numpy.interp(sampleFFDIs[patch][t],
                         ffdiRanges[int(patchVegetations[patch])],
                         occurrence[int(patchVegetations[patch])][t]) *
                         patchAreas[patch]
            for patch in range(noPatches)]
           for t in range(totalSteps + lookahead)])

    # Expected visible fires for each component for each patch at each time
    # step. As FFDI is assumed deterministic, this array is used for all paths
    expFiresComp = numpy.array([[
            numpy.matmul(expectedFires[t], tankerCovers),
            numpy.matmul(expectedFires[t], numpy.logical_not(tankerCovers)),
            numpy.matmul(expectedFires[t], heliCovers),
            numpy.matmul(expectedFires[t], numpy.logical_not(heliCovers))]
        for t in range(totalSteps + lookahead)])

    # Copy universal values to device
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
    d_resourceTypes = cuda.to_device(resourceTypes)
    d_resourceSpeeds = cuda.to_device(resourceSpeeds)
    d_configsE = cuda.to_device(configsE)
    d_configsP = cuda.to_device(configsP)
    d_thresholds = cuda.to_device(thresholds)

    # Run this in chunks (i.e. multiple paths at a time)
    for b, batchSize in enumerate(batchAmounts):
        # CUDA requirements
        batchStart = sum(batchAmounts[:b])
        batchEnd = batchStart + batchAmounts[b]
        threadsperblock = 32
        blockspergrid = (batchSize + (threadsperblock - 1)) // threadsperblock

        # Copy batch-relevant memory to the device
        d_accumulatedDamages = cuda.to_device(accumulatedDamages[
                batchStart:batchEnd])
        d_accumulatedHours = cuda.to_device(accumulatedHours[
                batchStart:batchEnd])
        d_fires = cuda.to_device(fires[batchStart:batchEnd])
        d_fireSizes = cuda.to_device(fireSizes[batchStart:batchEnd])
        d_fireLocations = cuda.to_device(fireLocations[batchStart:batchEnd])
        d_firePatches = cuda.to_device(firePatches[batchStart:batchEnd])
        d_aircraftLocations = cuda.to_device(aircraftLocations[
                batchStart:batchEnd])
        d_aircraftAssignments = cuda.to_device(aircraftAssignments[
                batchStart:batchEnd])
        d_expFiresComp = cuda.to_device(expFiresComp)
        d_tankerDists = cuda.to_device(tankerDists)
        d_heliDists = cuda.to_device(heliDists)
        d_maxHours = cuda.to_device(maxHours)
        d_configurations = cuda.to_device(configurations)
        d_baseConfigsMax = cuda.to_device(baseConfigsMax)
        d_fireConfigsMax = cuda.to_device(fireConfigsMax)
        d_controls = cuda.to_device(controls)
        d_regressionX = cuda.to_device(regressionX)
        d_regressionY = cuda.to_device(regressionY)

        # Initialise all random numbers state to use for each thread
        rng_states = create_xoroshiro128p_states(batchSize, seed=1)

        # Compute the paths in batches to preserve memory (see if we can
        # exploit both GPUs to share the computational load)
        simulateSinglePath[blockspergrid, threadsperblock](
                batchSize, totalSteps, lookahead, d_sampleFFDIs,
                d_expFiresComp, d_patchVegetations, d_patchAreas,
                d_patchLocations, d_baseLocations, d_tankerDists, d_heliDists,
                d_ffdiRanges, d_rocA2PHMeans, d_rocA2PHSDs, d_occurrence,
                d_initSizeM, d_initSizeSD, d_initSuccess, d_resourceTypes,
                d_resourceSpeeds, d_maxHours, d_configurations, d_configsE,
                d_configsP, d_baseConfigsMax, d_fireConfigsMax, d_thresholds,
                d_accumulatedDamages, d_accumulatedHours, d_fires, d_fireSizes,
                d_fireLocations, d_firePatches, d_aircraftLocations,
                d_aircraftAssignments, rng_states, d_controls, d_regressionX,
                d_regressionY, 0, stepSize, False)

#            cuda.synchronize()

#        simulateSinglePath[blockspergrid, threadsperblock](
#                batchSize, totalSteps, lookahead, sampleFFDIs, expFiresComp,
#                patchVegetations, patchAreas, patchLocations, baseLocations,
#                tankerDists, heliDists, ffdiRanges, rocA2PHMeans, rocA2PHSDs,
#                occurrence, initSizeM, initSizeSD, initSuccess, resourceTypes,
#                resourceSpeeds, maxHours, configurations, configsE, configsP,
#                baseConfigsMax, fireConfigsMax, thresholds, accumulatedDamages[
#                batchStart:batchEnd], accumulatedHours[ batchStart:batchEnd],
#                fires[batchStart:batchEnd], fireSizes[batchStart:batchEnd],
#                fireLocations[batchStart:batchEnd], firePatches[
#                batchStart:batchEnd], aircraftLocations[batchStart:batchEnd],
#                aircraftAssignments[batchStart:batchEnd], rng_states[
#                batchStart:batchEnd], controls[batchStart:batchEnd],
#                regressionX, regressionY, 0, stepSize, False)

        # Return memory to the host

def analyseMCPaths():
    pass

@cuda.jit(device=True)
def computeStatesAndControl():
    pass