# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 14:35:12 2018

@author: nicholas
"""

import sys
import numpy
import math
import cmath
from numba import cuda, float32, float64, int32
from numba.types import b1
#from pyculib import rand
from numba.cuda.random import create_xoroshiro128p_states
from numba.cuda.random import xoroshiro128p_normal_float32
from numba.cuda.random import xoroshiro128p_uniform_float32
#import pyqt_fit.nonparam_regression as smooth
#from pyqt_fit import npr_methods
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
#from sklearn import linear_model
import statsmodels.api as sm
#from scipy import stats


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
def simulateSinglePath(paths, totalSteps, lookahead, mr, mrsd, sampleFFDIs,
                       expFiresComp, lambdas, patchVegetations, patchAreas,
                       patchLocations, baseLocations, tankerDists, heliDists,
                       ffdiRanges, rocA2PHMeans, rocA2PHSDs, occurrence,
                       initSizeM, initSizeSD, initSuccess, extSuccess,
                       resourceTypes, resourceSpeeds, maxHours, configurations,
                       configsE, configsP, baseConfigsMax, fireConfigsMax,
                       thresholds, accumulatedDamages, randomFFDIpaths,
                       accumulatedHours, fires, initialExtinguished,
                       fireStarts, fireSizes, fireLocations, firePatches,
                       aircraftLocations, aircraftAssignments, rng_states,
                       states, controls, regressionX, regressionY, costs2Go,
                       start, stepSize, method, optimal, static, discount,
                       expectedTemp):

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
        costs2Go[path][start] = 0

        # Modify to take variables as assignments. Use tuples to define shape
        # e.g. cuda.local.array((16, 1000), dtype=float32)
        expectedE = cuda.local.array((noFiresMax, noConfE), dtype=float32)
        expectedP = cuda.local.array((noPatches, noConfP), dtype=float32)
        weightsP = cuda.local.array((noPatches, noConfP), dtype=float32)
        selectedE = cuda.local.array(noFiresMax, dtype=int32)
        expectedFFDI = cuda.local.array((noPatches, lookahead), dtype=float32)

        for ii in range(noFiresMax):
            for jj in range(noConfE):
                expectedE[ii, jj] = 0.0

        for ii in range(noPatches):
            for jj in range(noConfP):
                expectedP[ii, jj] = 0.0

        for tt in range(start, totalSteps):
            """ For all time steps before the end of the day, we must
            simulate. For the end of day, we don't. We just need the
            termination value. We need the EXPECTED FFDI path to the end. """
            expectedFFDIPath(randomFFDIpaths[path, :, tt],
                sampleFFDIs[:, tt:(totalSteps + lookahead + 1)],
                mr, mrsd, totalSteps + lookahead - tt, expectedFFDI)
#            expectedFFDI = sampleFFDIs[:, tt:(totalSteps + lookahead + 1)]

            expectedDamageExisting(
                    expectedFFDI, firePatches[path][tt - start], fires[path][tt
                    - start], fireSizes[path][tt - start], patchVegetations,
                    ffdiRanges, rocA2PHMeans, rocA2PHSDs, extSuccess, configsE,
                    lookahead, expectedE, rng_states, path, False, totalSteps -
                    tt, discount)

            expectedDamagePotential(
                    expectedFFDI, patchVegetations, patchAreas, ffdiRanges,
                    rocA2PHMeans, rocA2PHSDs, occurrence[:, start:(totalSteps +
                    lookahead + 1), :], initSizeM, initSizeSD, initSuccess,
                    extSuccess, configsP, tt - start, lookahead, expectedP,
                    rng_states, path, False, totalSteps - tt, discount)

#            if path == 2 and tt == 11:
#                for fire in range(fires[path][tt]):
#                    for config in range(noConfE):
#                        expectedTemp[0, fire, config] = expectedE[fire, config]
#
#                    expectedTemp[0, fire, 0] = firePatches[path][tt - start][fire]
#                    expectedTemp[0, fire, 1] = fire + 1
#                    expectedTemp[0, fire, 2] = fireSizes[path][tt - start][fire]
#                    expectedTemp[0, fire, 3] = expectedFFDI[firePatches[path][tt - start][fire], tt]
##                for patch in range(noPatches):
##                    expectedTemp[0, patch, 0] = expectedFFDI[patch, 0]
##                    expectedTemp[0, patch, 1] = sampleFFDIs[patch, tt]
#
#            if path == 2 and tt == 12:
#                for fire in range(fires[path][tt]):
#                    for config in range(noConfE):
#                        expectedTemp[1, fire, config] = expectedE[fire, config]
#
#                    expectedTemp[1, fire, 0] = firePatches[path][tt - start][fire]
#                    expectedTemp[1, fire, 1] = fire + 1
#                    expectedTemp[1, fire, 2] = fireSizes[path][tt - start][fire]
#                    expectedTemp[1, fire, 3] = expectedFFDI[firePatches[path][tt - start][fire], tt]
##                for patch in range(noPatches):
##                    expectedTemp[1, patch, 0] = expectedFFDI[patch, 0]
##                    expectedTemp[1, patch, 1] = sampleFFDIs[patch, tt]

            if static < 0:
                saveState(aircraftAssignments, resourceTypes, resourceSpeeds,
                          maxHours, tankerDists, heliDists, aircraftLocations,
                          accumulatedHours, patchLocations, baseLocations,
                          ffdiRanges, fires, fireSizes, fireLocations,
                          expectedE, expectedP, expFiresComp, configurations,
                          configsE, configsP, baseConfigsMax, fireConfigsMax,
                          selectedE, weightsP, states, lambdas, thresholds,
                          method, stepSize, tt - start, lookahead,
                          expectedTemp, path)

            if (optimal and (static < 0)):
                # ROV Optimal Control
                # Compute the optimal control using regressions.
                bestControl = 0
                bestC2G = math.inf

                """ For all steps in the backward induction we need to
                determine the best control to pick """
                if tt > 0:
                    for control in range(noControls):

                        if tt == totalSteps - 1:
                            """ Determine termination cost at the start of the
                            last period (no future flexibility, just use point
                            estimate for C2G by computing MIP) """
                            assignAircraft(
                                aircraftAssignments, resourceSpeeds,
                                resourceTypes, maxHours, aircraftLocations,
                                accumulatedHours, baseLocations, tankerDists,
                                heliDists, fires, fireSizes, fireLocations,
                                ffdiRanges, configurations, configsE, configsP,
                                baseConfigsMax, fireConfigsMax, thresholds,
                                expFiresComp, expectedE, expectedP, selectedE,
                                weightsP, tt - start, stepSize, lookahead,
                                path, control, lambdas, method,
                                expectedTemp, 0)

                            currC2G = 0.0

                            for fire in range(fires[path][tt - start]):
                                currC2G += expectedE[fire, selectedE[fire]]

                            for patch in range(noPatches):
                                for config in range(noConfP):
                                    currC2G += (weightsP[patch, config]
                                                * expectedP[patch, config])

                        else:
                            """ Need to determine the expected cost to go for
                            each control in order to determine the best one to
                            pick. This requires computing the state for each
                            control using the assignment heuristic and then the
                            regressions. """
                            currC2G = 0.0

                            assignAircraft(
                                aircraftAssignments, resourceSpeeds,
                                resourceTypes, maxHours, aircraftLocations,
                                accumulatedHours, baseLocations, tankerDists,
                                heliDists, fires, fireSizes, fireLocations,
                                ffdiRanges, configurations, configsE, configsP,
                                baseConfigsMax, fireConfigsMax, thresholds,
                                expFiresComp, expectedE, expectedP, selectedE,
                                weightsP, tt - start, stepSize, lookahead,
                                path, control, lambdas, method,
                                expectedTemp, 0)

                            # First, we need to determine the single-period
                            # damage (expected) by selecting this control. This
                            # step also determines the next period state, which
                            # is used to compute the C2G after this time step.
                            # This means that the simulation to the next step
                            # must use expectations
#                            simulateNextStep(
#                                aircraftAssignments, resourceTypes,
#                                resourceSpeeds, aircraftLocations,
#                                accumulatedHours, baseLocations, fires,
#                                initialExtinguished, fireStarts,
#                                patchVegetations, patchLocations, patchAreas,
#                                ffdiRanges, fireLocations, firePatches,
#                                sampleFFDIs, randomFFDIpaths, mr, mrsd,
#                                rocA2PHMeans, rocA2PHSDs, fireSizes, configsE,
#                                configsP, selectedE, weightsP, initSizeM,
#                                initSizeSD, initSuccess, extSuccess,
#                                occurrence[:, start:(totalSteps + lookahead +
#                                1), :], accumulatedDamages, tt - start,
#                                stepSize, rng_states, False, path)
#
#                            """ Single-period expected accumulated damage """
#                            currC2G = 0
#
#                            for patch in range(noPatches):
#                                currC2G += (accumulatedDamages[path, tt + 1,
#                                                               patch]
#                                            - accumulatedDamages[path, tt,
#                                                                 patch])

                            """ Get the expected cost 2 go for this control at
                            this time for the updated state at the next time
                            period. The predictors are the expected damages
                            incurred by::
                            1. Fighting existing fires THIS period,
                            2. Fighting potential fires THIS period,
                            3. Not fighting existing fires AT ALL this period,
                            4. Not fighting potential fires AT ALL this period.
                            It is assumed that in subsequent periods, the fires
                            produced this period (and fought this period) are
                            left unchecked. This way, we measure the OVERALL
                            expected damage reduction of fighting existing and
                            potential fires according to the control's found
                            configurations FOR THIS PERIOD ONLY. By measuring
                            the four dimensions, we can also determine the
                            relative proximity of aircraft to potential and
                            existing fires. To do this, the expected damage
                            computations must look to the end of the horizon.
                            """
                            if len(regressionX.shape) == 4:
                                currC2G += interpolateCost2Go(
                                        states, regressionX, regressionY,
                                        tt - start + 1, path, control)

                            else:
                                currC2G += calculateCost2Go(
                                        states, regressionX, tt - start + 1,
                                        path, control)

                        if currC2G < bestC2G:
                            bestC2G = currC2G
                            bestControl = control

                controls[path][tt] = bestControl

            else:
                if static < 0:
                    # ROV Random Control
                    bestControl = int(noControls*xoroshiro128p_uniform_float32(
                        rng_states, path))
                    controls[path][tt] = bestControl

                else:
                    if optimal:
                        # Dynamic MPC, fixed control
                        bestControl = static
                        controls[path][tt] = static
                    else:
                        # Dynamic MPC, specified control at start, random after
                        if tt == start:
                            bestControl = static
                            controls[path][tt] = static
                        else:
                            bestControl = int(
                                noControls*xoroshiro128p_uniform_float32(
                                    rng_states, path))
                            controls[path][tt] = bestControl

#            bestControl = 0
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
                           method, expectedTemp, 0)

#            if path == 2 and tt == 10:
#                for fire in range(16):
#                    for config in range(36):
##                    expectedTemp[fire, tt] = fireSizes[path][tt - start][fire]
#                        expectedTemp[fire, config] = weightsP[fire, config]

            # SimulateNextStep
            simulateNextStep(aircraftAssignments, resourceTypes,
                             resourceSpeeds, aircraftLocations,
                             accumulatedHours, baseLocations, fires,
                             initialExtinguished, fireStarts, patchVegetations,
                             patchLocations, patchAreas, ffdiRanges,
                             fireLocations, firePatches, sampleFFDIs,
                             randomFFDIpaths, mr, mrsd, rocA2PHMeans,
                             rocA2PHSDs, fireSizes, configsE, configsP,
                             selectedE, weightsP, initSizeM, initSizeSD,
                             initSuccess, extSuccess, occurrence[:, start:(
                             totalSteps + lookahead + 1), :],
                             accumulatedDamages, tt - start, stepSize,
                             rng_states, path)

        """ We need to use the forward path (re-)computation performed here
        to determine the cost-to-go at the PREVIOUS time step. For this, we
        just need to compute the accumulated damage from the PREVIOUS time step
        through to the end. If we are at the first period, we do not need to do
        this """
        if start > 0:
            costs2Go[path][start] = 0

            # We discount the future damages by the discount factor. This allows
            # the algorithm to place more weight on more immediate damage.
            for tt in range(start, totalSteps + lookahead - 1):
                for patch in range(noPatches):
                    costs2Go[path][start] += ((
                        accumulatedDamages[path, tt + 1, patch]
                        - accumulatedDamages[path, tt, patch]) /
                        (1 + discount) ** (tt - start))
#        for tt in range(start, start + 1):
#            """ The cost to go for the prior period will include the expected
#            accumulated damage for that period's control for that period
#            up to now PLUS the C2G from this period onwards, given the updated
#            state expected from running the best control for one period. This
#            is used for computing the regressions in the previous period. As we
#            will have (re-)computed the forward paths up to now, we can just
#            use the remaining accumulated damage to the end period minus the
#            accumulated damage already incurred.
#
#            We also go one extra period here to record the termination value
#            at the end of the final period. """
#
#            costs2Go[path][tt] = 0
#
#            if tt == totalSteps - 1:
#                pass
#                # We can just sum over the future expectation for this period
#                # with the selected assignment. and the following five periods
#                # (nighttime) with no assignment, given that this period's
#                # assignments.
#
#            else:
#                # We need to determine the single period expected damages for
#                # the chosen assignments, which we add to the C2G
#                expectedDamageExisting(
#                    expectedFFDI, firePatches[path][tt - start], fires[path][tt
#                    - start], fireSizes[path][tt - start], patchVegetations,
#                    ffdiRanges, rocA2PHMeans, rocA2PHSDs, extSuccess, configsE,
#                    1, expectedE, rng_states, path, False, 1)
#
#                expectedDamagePotential(
#                    expectedFFDI, patchVegetations, patchAreas, ffdiRanges,
#                    rocA2PHMeans, rocA2PHSDs, occurrence[:, start:(totalSteps
#                    + lookahead + 1), :], initSizeM, initSizeSD, initSuccess,
#                    extSuccess, configsP, tt - start, 1, expectedP,
#                    rng_states, path, False, 1)
#
#            currC2G = 0
#
#            for fire in range(fires[path][tt - start]):
#                currC2G += expectedE[fire, selectedE[fire]]
#
#            for patch in range(noPatches):
#                for config in range(noConfP):
#                    currC2G += (weightsP[patch, config]
#                                * expectedP[patch, config])
#
#            costs2Go[path][tt] += currC2G


@cuda.jit(device=True)
def expectedFFDIPath(randomFFDIpath, meanFFDIs, mr, sd, lookahead,
                     expectedFFDI):
    expectedFFDI[:, 0] = randomFFDIpath;

    # We provide all patches the same uncertainty. The model is a generalised
    # Ornstein-Uhlenbeck process (mean reversion with geometric Brownian
    # motion where the Brownian component is dependent on the square root of
    # the FFDI at the previous time period)
    for tt in range(lookahead):
        expectedFFDI[:, tt+1] = max(((meanFFDIs[:, tt] - expectedFFDI[:, tt])
            * mr + expectedFFDI[:, tt]), 0)


#@jit(nopython=True, fastmath=True)
@cuda.jit(device=True)
def expectedDamageExisting(ffdi_path, fire_patches, no_fires, fire_sizes,
                           patch_vegetations, ffdi_ranges, roc_a2_ph_means,
                           roc_a2_ph_sds, ext_success, configs, lookahead,
                           expected, rng_states, thread_id, random, endTime,
                           discount):
    """ This computes the expected damage FOR THE REMAINING TIME. It applies
    the control for this period and then no control for the remainder. This
    allows us to see the effect of doing something now on the remaining time
    """

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

            damage = 0

            for tt in range(lookahead):
                if tt >= 0:
                    config = 1

                sizeOld = size
                ffdi = ffdi_path[patch, tt]
                success = interpolate1D(ffdi, ffdi_range, ext_succ[config - 1])
                growth = growFire(ffdi, config - 1, ffdi_range, roc_a2_ph_mean,
                                  roc_a2_ph_sd, sizeTemp, rng_states,
                                  thread_id, random) - sizeTemp

                sizeTemp += growth
                size += growth * (1 - success) ** (tt)
                damage += (size - sizeOld) / ((1 + discount) ** (tt))

            expected[fire, c] = damage
#            expected[fire, c] = size - fire_sizes[fire]


@cuda.jit(device=True)
def expectedDamagePotential(ffdi_path, patch_vegetations, patch_areas,
                            ffdi_ranges, roc_a2_ph_means, roc_a2_ph_sds,
                            occurrence, init_size_m, init_size_sd,
                            init_success, ext_success, configs, time,
                            lookahead, expected, rng_states, thread_id,
                            random, endTime, discount):
    """ This looks at the benefit of fighting new fires THIS PERIOD. The
    computed damage is for the ENTIRE horizon. Future time steps are
    treated as nothing being done after this period. This allows us to
    see the benefit of what we do THIS period on the entire horizon. """


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
            expected[patch, c] = 0.0

            for tt in range(lookahead):
                if tt >= 0:
                    config = 1
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
                sizeTemp = math.exp(sizeM + sizeSD ** 2 / 2)

                # Add the damage caused by fires formed this period but
                # extinguished immediately
                damage += occ * sizeTemp / ((1 + discount) ** (tt))

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

                    damage += (occ * growth * (1 - success2) ** (t2 - tt) *
                               (1 - success)) / ((1 + discount) ** (tt))

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
        rad_new = rad_curr + math.exp(gr_mean + gr_sd ** 2 / 2)

    if not cmath.isfinite(rad_new):
        rad_new = rad_curr + math.exp(gr_mean + gr_sd ** 2 / 2)

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
def calculateCost2Go(states, regressionX, time, path, control):

    returnVal = (regressionX[time][control][0][0]
                 + regressionX[time][control][1][0] * states[path][time][0]
                 + regressionX[time][control][2][0] * states[path][time][1]
                 + regressionX[time][control][3][0] * states[path][time][2]
                 + regressionX[time][control][4][0] * states[path][time][0]
                     ** 2
                 + regressionX[time][control][5][0] * states[path][time][0]
                     * states[path][time][1]
                 + regressionX[time][control][6][0] * states[path][time][0]
                     * states[path][time][2]
                 + regressionX[time][control][7][0] * states[path][time][1]
                     ** 2
                 + regressionX[time][control][8][0] * states[path][time][1]
                     * states[path][time][2]
                 + regressionX[time][control][9][0] * states[path][time][2]
                     ** 2)

    return returnVal


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
        (1 - xd)*regressionY[time][control][lowerInd[0]][lowerInd[1]][
            lowerInd[2]]
        + xd*regressionY[time][control][lowerInd[0]+1][lowerInd[1]][
            lowerInd[2]])
    coeffs[1] = (
        (1 - xd)*regressionY[time][control][lowerInd[0]][lowerInd[1]][
            lowerInd[2]+1]
        + xd*regressionY[time][control][lowerInd[0]+1][lowerInd[1]][
            lowerInd[2]+1])
    coeffs[2] = (
        (1 - xd)*regressionY[time][control][lowerInd[0]][lowerInd[1]+1][
            lowerInd[2]]
        + xd*regressionY[time][control][lowerInd[0]+1][lowerInd[1]+1][
            lowerInd[2]])
    coeffs[3] = (
        (1 - xd)*regressionY[time][control][lowerInd[0]][lowerInd[1]][
            lowerInd[2]]
        + xd*regressionY[time][control][lowerInd[0]+1][lowerInd[1]+1][
            lowerInd[2]+1])

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

    global noConfigs
    tempSortList = cuda.local.array(noConfigs, dtype=int32)

    # Zero the output list
    for config in range(len(baseConfigsPossible)):
        baseConfigsPossible[config] = 0
        tempSortList[config] = 0

    # Collect
    found = 0
    for config in range(len(configs)):
        viable = True

        for c in range(4):
            if configurations[configs[config]-1][c] > numbers[c]:
                viable = False
                break

        if viable:
            # Config indexing refers to index of USEFUL PATCH CONFIGS, NOT
            # overall config indices
            tempSortList[found] = config
            found += 1

    # Simple selection sort for now as the list will likely be very small
    for i in range(found):
        iMin = i

        for j in range(i+1, found):
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
                     noFires, initialExtinguished, fireStarts,
                     patchVegetations, patchLocations, patchAreas, ffdiRanges,
                     fireLocations, firePatches, sampleFFDIs, randomFFDIpaths,
                     mr, mrsd, roc_a2_ph_means, roc_a2_ph_sds, fireSizes,
                     fireConfigs, patchConfigs, selectedE, configWeights,
                     initM, initSD, init_succ, ext_succ, occurrence,
                     accumulatedDamage, time, stepSize, rng_states, random,
                     thread_id):

    global noPatches

    """ Update patch FFDIs """
    if random:
        for patch in range(noPatches):
            rand = xoroshiro128p_normal_float32(rng_states, thread_id)
            randomFFDIpaths[thread_id, patch, time + 1] = max(mr * (
                sampleFFDIs[patch, time + 1] - randomFFDIpaths[thread_id,
                patch, time]) + rand*math.sqrt(randomFFDIpaths[thread_id,
                patch, time])*mrsd, 0)
    else:
        for patch in range(noPatches):
            randomFFDIpaths[thread_id, patch, time + 1] = max(mr * (
                sampleFFDIs[patch, time + 1] - randomFFDIpaths[thread_id,
                patch, time]), 0)

    """ Update aircraft locations """
    for resource in range(len(aircraftSpeeds)):
        baseAssignment = int(aircraftAssignments[thread_id][time+1][resource][
                             0]) - 1
        fireAssignment = int(aircraftAssignments[thread_id][time+1][resource][
                             1]) - 1
        speed = aircraftSpeeds[resource]
        acX = aircraftLocations[thread_id][time][resource][0]
        acY = aircraftLocations[thread_id][time][resource][1]

        aircraftLocations[thread_id][time+1][resource][0] = acX
        aircraftLocations[thread_id][time+1][resource][1] = acY

        if fireAssignment >= 0:
            [fireX, fireY] = fireLocations[thread_id][time][fireAssignment]

            distance = geoDist(aircraftLocations[thread_id][time][resource],
                               fireLocations[thread_id][time][fireAssignment])

            if distance / speed > stepSize:
                frac = stepSize / (distance / speed)
#                This distance is not direct as we are not accounting for the
#                curvature of the earth. This will be fixed later.
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
                frac = stepSize / (distance / speed)
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
    for patch in range(noPatches):
        accumulatedDamage[thread_id][time+1][patch] = (
                accumulatedDamage[thread_id][time][patch])

    """ Fight existing fires, maintaining ones that are not extinguished """
    count = 0
    count1 = 0
    count2 = 0

    for fire in range(int(noFires[thread_id][time])):
        patch = int(firePatches[thread_id][time][fire])
        vegetation = int(patchVegetations[patch])
        ffdi_range = ffdiRanges[vegetation]
        roc_a2_ph_mean = roc_a2_ph_means[vegetation]
        roc_a2_ph_sd = roc_a2_ph_sds[vegetation]
        exist_success = ext_succ[vegetation]

        size = fireSizes[thread_id][time][fire]
        sizeTemp = size

        ffdi = randomFFDIpaths[thread_id, patch, time]
        config = fireConfigs[selectedE[fire]]
        success = interpolate1D(ffdi, ffdi_range, exist_success[config - 1])

        if random:
            grow = True
        else:
            grow = False

        size = growFire(ffdi, config - 1, ffdi_range, roc_a2_ph_mean,
                        roc_a2_ph_sd, fireSizes[thread_id][time][fire],
                        rng_states, thread_id, grow)

        accumulatedDamage[thread_id][time+1][patch] += size - sizeTemp

        rand_no = xoroshiro128p_uniform_float32(rng_states, thread_id)

        if rand_no > success:
            fireSizes[thread_id][time+1][count] = size
            fireLocations[thread_id][time + 1][count][0] = (
                    fireLocations[thread_id][time][fire][0])
            fireLocations[thread_id][time + 1][count][1] = (
                    fireLocations[thread_id][time][fire][1])
            firePatches[thread_id][time + 1][count] = firePatches[
                    thread_id][time][fire]
            count += 1

    """ New fires in each patch """
    for patch in range(noPatches):
        vegetation = int(patchVegetations[patch])
        ffdi_range = ffdiRanges[vegetation]
        ffdi = randomFFDIpaths[thread_id, patch, time]
        initial_size_M = initM[vegetation]
        initial_size_SD = initSD[vegetation]
        initial_success = init_succ[vegetation]

        if random:
            rand = xoroshiro128p_uniform_float32(rng_states, thread_id)
            scale = (interpolate1D(ffdi, ffdi_range, occurrence[vegetation][time])
                     * patchAreas[patch])
            # Bottom up summation for Poisson distribution
            cumPr = math.exp(-scale) * scale ** (0)
            factor = 1
            factorCounter = 1
            newFires = 0

            while cumPr <= rand:
                newFires += 1
                cumPr += math.exp(-scale) * scale ** (newFires) / (factor)
                factorCounter += 1
                factor = factor * factorCounter

            if newFires > 0:
                sizeMean = 0.0
                sizeSD = 0.0
                initS = 0.0

                for config in range(len(patchConfigs)):
                    weight = configWeights[patch, config]

                    if weight > 0.0:
                        sizeMean += weight * interpolate1D(
                                ffdi, ffdi_range,
                                initial_size_M[patchConfigs[config] - 1])
                        sizeSD += weight * interpolate1D(
                                ffdi, ffdi_range, initial_size_SD[
                                        patchConfigs[config] -1])
                        initS += weight * interpolate1D(
                                ffdi, ffdi_range, initial_success[
                                        patchConfigs[config] - 1])

                for fire in range(newFires):
                    success = True if initS > xoroshiro128p_uniform_float32(
                            rng_states, thread_id) else False
                    randVal = xoroshiro128p_normal_float32(rng_states, thread_id)
                    size = math.exp(sizeMean + randVal * sizeSD)

                    if not cmath.isfinite(size):
                        size = math.exp(sizeMean + sizeSD ** 2 / 2)

                    accumulatedDamage[thread_id][time+1][patch] += size

                    if not success:
                        fireSizes[thread_id][time+1][count] = size
                        fireLocations[thread_id][time + 1][count][0] = (
                                patchLocations[patch][0])
                        fireLocations[thread_id][time + 1][count][1] = (
                                patchLocations[patch][1])
                        firePatches[thread_id][time + 1][count] = patch
                        count += 1
                        count2 += 1
                    else:
                        count1 += 1
                        count2 += 1
        else:
            """ We are only doing a 1 step expectation in order to compute the
            single-period cost (which will be added to the C2G) so we will not
            actually save the results other than accumulated damage. """
            noFires = (interpolate1D(ffdi, ffdi_range, occurrence[vegetation][
                       time]) * patchAreas[patch])

            sizeMean = 0.0
            sizeSD = 0.0
            initS = 0.0

            for config in range(len(patchConfigs)):
                weight = configWeights[patch, config]

                if weight > 0.0:
                    sizeMean += weight * interpolate1D(
                            ffdi, ffdi_range,
                            initial_size_M[patchConfigs[config] - 1])
                    sizeSD += weight * interpolate1D(
                            ffdi, ffdi_range, initial_size_SD[
                                    patchConfigs[config] -1])
                    initS += weight * interpolate1D(
                            ffdi, ffdi_range, initial_success[
                                    patchConfigs[config] - 1])

            # Fires created this period
            accumulatedDamage[thread_id][time+1][patch] += (noFires * math.exp(
                sizeMean + sizeSD ** 2 / 2))

    noFires[thread_id][time + 1] = count
    initialExtinguished[thread_id][time] = count1
    fireStarts[thread_id][time] = count2


@cuda.jit(device=True)
def assignAircraft(aircraftAssignments, resourceSpeeds, resourceTypes,
                   maxHours, aircraftLocations, accumulatedHours,
                   baseLocations, tankerCovers, heliCovers, noFires,
                   fireSizes, fireLocations, ffdiRanges, configurations,
                   configsE, configsP, baseConfigsMax, fireConfigsMax,
                   thresholds, expFiresComp, expectedE, expectedP, selectedE,
                   weightsP, time, stepSize, lookahead, thread_id, control,
                   lambdas, method, expectedTemp, dummy):

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
    baseImproveTankerL = cuda.local.array(noBases, dtype=float32)
    baseImproveHeliL = cuda.local.array(noBases, dtype=float32)
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
    baseConfigsPossible = cuda.local.array(noConfigs, dtype=int32)
    configNos = cuda.local.array(4, dtype=int32)
    fireCumulativeSavings = cuda.local.array(noFiresMax, dtype=float32)
    patchExpectedDamage = cuda.local.array(noPatches, dtype=float32)
    nearest = cuda.local.array(noAircraft, dtype=float32)

    """ First make sure that the next period assignments are zeroed """
    for resource in range(noAircraft):
        aircraftAssignments[thread_id][time+1][resource][0] = 0
        aircraftAssignments[thread_id][time+1][resource][1] = 0

    """ Threshold distances """
    if dummy == 0:
        if method == 1:
            # Custom thresholds for relocation distances
            fw = 1
            pw = 1
            maxFire = min(thresholds[1], lambdas[control][0])
            maxBase = lambdas[control][1]
        elif method == 2:
            # Weighting between Fires and Potential and relocation
            fw = lambdas[control][0]
            pw = 1 - lambdas[control][0]
            maxFire = min(thresholds[1], lambdas[control][1])
            maxBase = lambdas[control][1]

    elif dummy == 1:
        fw = 0.95
#        # Assign only to cover existing fires
#        if method == 1:
#            for c in range(len(lambdas[:])):
#                fw = max(fw, lambdas[c][0])

        pw = 1 - fw
        maxFire = thresholds[1]
        maxBase = math.inf

    elif dummy == 2:
        # Assign only to cover potential fires
        fw= 0
        pw = 1 - fw
        maxFire = thresholds[1]
        maxBase = math.inf

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
        # Initialise expected patch damages to the unprotected damage
        patchExpectedDamage[patch] = pw * expectedP[patch, 0]

    for fire in range(noFiresMax):
        fireTankersE[fire] = 0
        fireTankersL[fire] = 0
        fireHelisE[fire] = 0
        fireHelisL[fire] = 0
        fireMaxTankersE[fire] = 0
        fireMaxTankersL[fire] = 0
        fireMaxHelisE[fire] = 0
        fireMaxHelisL[fire] = 0
        fireCumulativeSavings[fire] = 0

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
                    if dist <= thresholds[0]:
                        baseMaxTankersE[base] += 1
                    elif dist <= thresholds[1]:
                        baseMaxTankersL[base] += 1

                elif resourceTypes[resource] == 1:
                    if dist <= thresholds[0]:
                        baseMaxHelisE[base] += 1
                    elif dist <= thresholds[1]:
                        baseMaxHelisL[base] += 1

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
                    if travTime < thresholds[0]:
                        fireMaxTankersE[fire] += 1
                    elif travTime <= thresholds[1]:
                        fireMaxTankersL[fire] += 1
                elif resourceTypes[resource] == 1:
                    if travTime < thresholds[0]:
                        fireMaxHelisE[fire] += 1
                    elif travTime <= thresholds[1]:
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

                # Tanker Early
                if baseMaxTankersE[base] > 0:
                    expectedDamageBase = 0.0
                    baseDamageBaseline = 0.0

                    for patch in range(noPatches):
                        if tankerCovers[patch, base] <= min(maxFire,
                                thresholds[0]):
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

                # Tanker Late - Get to the base late
                if baseMaxTankersL[base] > 0:
                    expectedDamageBase = 0.0
                    baseDamageBaseline = 0.0

                    for patch in range(noPatches):
                        if tankerCovers[patch, base] <= min(maxFire,
                                thresholds[0]):
                            # If the patch is close to the base
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

                    baseImproveTankerL[base] = (baseDamageBaseline
                        - expectedDamageBase)

                    if baseImproveTankerE[base] > thisUpdateImprovement:
                            thisUpdateImprovement = baseImproveTankerE[base]
                            thisUpdateDType = 1
                            thisUpdateACType = 1
                            thisUpdateIdx = base
                else:
                    baseImproveTankerE[base] = 0.0

                # Heli Early
                if baseMaxHelisE[base] > 0:
                    expectedDamageBase = 0.0
                    baseDamageBaseline = 0.0

                    for patch in range(noPatches):
                        if heliCovers[patch, base] <= min(maxFire,
                                thresholds[0]):
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

                    if baseImproveHeliE[base] > thisUpdateImprovement:
                            thisUpdateImprovement = baseImproveHeliE[base]
                            thisUpdateDType = 1
                            thisUpdateACType = 2
                            thisUpdateIdx = base

                else:
                    baseImproveHeliE[base] = 0.0

                # Heli Late
                if baseMaxHelisL[base] > 0:
                    expectedDamageBase = 0.0
                    baseDamageBaseline = 0.0

                    for patch in range(noPatches):
                        if heliCovers[patch, base] <= min(maxFire,
                                thresholds[0]):
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

                    baseImproveHeliL[base] = (baseDamageBaseline
                        - expectedDamageBase)

                    if baseImproveHeliL[base] > thisUpdateImprovement:
                            thisUpdateImprovement = baseImproveHeliL[base]
                            thisUpdateDType = 1
                            thisUpdateACType = 2
                            thisUpdateIdx = base

                else:
                    baseImproveHeliL[base] = 0.0

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
                # An extra early tanker ################
                if baseMaxTankersE[thisUpdateIdx] > baseTankersE[thisUpdateIdx]:
                    expectedDamageBase = 0.0
                    baseTankersE[thisUpdateIdx] += 1

                    # Calculate the improvement of one more tanker at this
                    # base for each patch
                    for patch in range(noPatches):
                        componentNos(configNos, tankerCovers, heliCovers,
                                     baseTankersE, baseHelisE, baseTankersL,
                                     baseHelisL, updateBases, patch, maxFire,
                                     thresholds, True)

                        if tankerCovers[patch, thisUpdateIdx] <= maxFire:

                            updatePatches[patch] = True

                            if (tankerCovers[patch, thisUpdateIdx] <=
                                    thresholds[0]):
                                configNos[0] += 1

                            elif (tankerCovers[patch, thisUpdateIdx] <=
                                    thresholds[1]):
                                configNos[2] += 1

                            # Get all possible configs and sort them by
                            # benefit to this patch
                            getAllConfigsSorted(
                                    configurations, configsP,
                                    baseConfigsPossible, expectedP[patch],
                                    configNos)

                            expectedDamageBase += expectedDamage(
                                baseConfigsPossible, configurations, configsP,
                                tankerCovers, heliCovers, baseTankersE,
                                baseHelisE, baseTankersL, baseHelisL,
                                expectedP, expFiresComp, weightsP, patch,
                                time, lookahead, pw, maxFire, thresholds)

                        else:
                            expectedDamageBase += patchExpectedDamage[patch]

                    baseImproveTankerE[thisUpdateIdx] = (
                            currentImprove - expectedDamageBase)
                    baseTankersE[thisUpdateIdx] -= 1

                else:
                    baseImproveTankerE[thisUpdateIdx] = 0.0

                # An extra late tanker ################
                if baseMaxTankersL[thisUpdateIdx] > baseTankersL[thisUpdateIdx]:
                    expectedDamageBase = 0.0
                    baseTankersL[thisUpdateIdx] += 1

                    # Calculate the improvement of one more tanker at this
                    # base for each patch
                    for patch in range(noPatches):
                        componentNos(configNos, tankerCovers, heliCovers,
                                     baseTankersE, baseHelisE, baseTankersL,
                                     baseHelisL, updateBases, patch, maxFire,
                                     thresholds, True)

                        if tankerCovers[patch, thisUpdateIdx] <= maxFire:

                            updatePatches[patch] = True

                            if (tankerCovers[patch, thisUpdateIdx] <=
                                    thresholds[1]):
                                configNos[2] += 1

                            # Get all possible configs and sort them by
                            # benefit to this patch
                            getAllConfigsSorted(
                                    configurations, configsP,
                                    baseConfigsPossible, expectedP[patch],
                                    configNos)

                            expectedDamageBase += expectedDamage(
                                baseConfigsPossible, configurations, configsP,
                                tankerCovers, heliCovers, baseTankersE,
                                baseHelisE, baseTankersL, baseHelisL,
                                expectedP, expFiresComp, weightsP, patch,
                                time, lookahead, pw, maxFire, thresholds)

                        else:
                            expectedDamageBase += patchExpectedDamage[patch]

                    baseImproveTankerL[thisUpdateIdx] = (
                            currentImprove - expectedDamageBase)
                    baseTankersL[thisUpdateIdx] -= 1

                else:
                    baseImproveTankerL[thisUpdateIdx] = 0.0

                # An extra early helicopter ############
                if baseMaxHelisE[thisUpdateIdx] > baseHelisE[thisUpdateIdx]:
                    expectedDamageBase = 0.0
                    baseHelisE[thisUpdateIdx] += 1

                    # Calculate the improvement of one more tanker at this
                    # base for each patch
                    for patch in range(noPatches):
                        componentNos(configNos, tankerCovers, heliCovers,
                                     baseTankersE, baseHelisE, baseTankersL,
                                     baseHelisL, updateBases, patch,
                                     maxFire, thresholds, True)

                        if heliCovers[patch, thisUpdateIdx] <= maxFire:

                            updatePatches[patch] = True

                            if (heliCovers[patch, thisUpdateIdx] <=
                                    thresholds[0]):
                                configNos[1] += 1

                            elif (heliCovers[patch, thisUpdateIdx] <=
                                    thresholds[1]):
                                configNos[3] += 1

                            # Get all possible configs and sort them by
                            # benefit to this patch
                            getAllConfigsSorted(
                                    configurations, configsP,
                                    baseConfigsPossible, expectedP[patch],
                                    configNos)

                            expectedDamageBase += expectedDamage(
                                baseConfigsPossible, configurations, configsP,
                                tankerCovers, heliCovers, baseTankersE,
                                baseHelisE, baseTankersL, baseHelisL,
                                expectedP, expFiresComp, weightsP, patch, time,
                                lookahead, pw, maxFire, thresholds)

                        else:
                            expectedDamageBase += patchExpectedDamage[patch]

                    baseImproveHeliE[thisUpdateIdx] = (
                        currentImprove - expectedDamageBase)
                    baseHelisE[thisUpdateIdx] -= 1

                else:
                    baseImproveHeliE[thisUpdateIdx] = 0.0

                # An extra late helicopter ############
                if baseMaxHelisL[thisUpdateIdx] > baseHelisL[thisUpdateIdx]:
                    expectedDamageBase = 0.0
                    baseHelisL[thisUpdateIdx] += 1

                    # Calculate the improvement of one more tanker at this
                    # base for each patch
                    for patch in range(noPatches):
                        componentNos(configNos, tankerCovers, heliCovers,
                                     baseTankersE, baseHelisE, baseTankersL,
                                     baseHelisL, updateBases, patch,
                                     maxFire, thresholds, True)

                        if heliCovers[patch, thisUpdateIdx] <= maxFire:

                            updatePatches[patch] = True

                            if (heliCovers[patch, thisUpdateIdx] <=
                                    thresholds[1]):
                                configNos[3] += 1

                            # Get all possible configs and sort them by
                            # benefit to this patch
                            getAllConfigsSorted(
                                    configurations, configsP,
                                    baseConfigsPossible, expectedP[patch],
                                    configNos)

                            expectedDamageBase += expectedDamage(
                                baseConfigsPossible, configurations, configsP,
                                tankerCovers, heliCovers, baseTankersE,
                                baseHelisE, baseTankersL, baseHelisL,
                                expectedP, expFiresComp, weightsP, patch, time,
                                lookahead, pw, maxFire, thresholds)

                        else:
                            expectedDamageBase += patchExpectedDamage[patch]

                    baseImproveHeliL[thisUpdateIdx] = (
                        currentImprove - expectedDamageBase)
                    baseHelisL[thisUpdateIdx] -= 1

                else:
                    baseImproveHeliL[thisUpdateIdx] = 0.0

                ###############################################################
                # Bases that are affected by adjusted patches
                for baseOther in range(noBases):
                    if updateBases[baseOther]:

                        # First recalculate the base that was just updated
                        # An extra early tanker ################
                        if baseMaxTankersE[baseOther] > baseTankersE[baseOther]:
                            expectedDamageBase = 0.0
                            baseTankersE[baseOther] += 1

                            # Calculate the improvement of one more tanker at
                            # this base for each patch
                            for patch in range(noPatches):
                                componentNos(
                                        configNos, tankerCovers, heliCovers,
                                        baseTankersE, baseHelisE, baseTankersL,
                                        baseHelisL, updateBases, patch,
                                        maxFire, thresholds, False)

                                if tankerCovers[patch, baseOther] <= maxFire:

                                    if (tankerCovers[patch, baseOther] <=
                                            thresholds[0]):
                                        configNos[0] += 1
                                    elif (tankerCovers[patch, baseOther] <=
                                            thresholds[1]):
                                        configNos[2] += 1

                                    # Get all possible configs and sort them by
                                    # benefit to this patch
                                    getAllConfigsSorted(
                                            configurations, configsP,
                                            baseConfigsPossible,
                                            expectedP[patch], configNos)

                                    expectedDamageBase += expectedDamage(
                                            baseConfigsPossible,
                                            configurations, configsP,
                                            tankerCovers, heliCovers,
                                            baseTankersE, baseHelisE,
                                            baseTankersL, baseHelisL,
                                            expectedP, expFiresComp, weightsP,
                                            patch, time, lookahead, pw,
                                            maxFire, thresholds)

                                else:
                                    expectedDamageBase += (
                                            patchExpectedDamage[patch])

                            baseImproveTankerE[baseOther] = (
                                currentImprove - expectedDamageBase)
                            baseTankersE[baseOther] -= 1

                        else:
                            baseImproveTankerE[baseOther] = 0.0

                        # An extra late tanker ###########
                        if baseMaxTankersL[baseOther] > baseTankersL[baseOther]:
                            expectedDamageBase = 0.0
                            baseTankersL[baseOther] += 1

                            # Calculate the improvement of one more tanker at
                            # this base for each patch
                            for patch in range(noPatches):
                                componentNos(
                                        configNos, tankerCovers, heliCovers,
                                        baseTankersE, baseHelisE, baseTankersL,
                                        baseHelisL, updateBases, patch,
                                        maxFire, thresholds, False)

                                if tankerCovers[patch, baseOther] <= maxFire:

                                    if (tankerCovers[patch, baseOther] <=
                                            thresholds[1]):
                                        configNos[2] += 1

                                    # Get all possible configs and sort them by
                                    # benefit to this patch
                                    getAllConfigsSorted(
                                            configurations, configsP,
                                            baseConfigsPossible,
                                            expectedP[patch], configNos)

                                    expectedDamageBase += expectedDamage(
                                            baseConfigsPossible,
                                            configurations, configsP,
                                            tankerCovers, heliCovers,
                                            baseTankersE, baseHelisE,
                                            baseTankersL, baseHelisL,
                                            expectedP, expFiresComp, weightsP,
                                            patch, time, lookahead, pw,
                                            maxFire, thresholds)

                                else:
                                    expectedDamageBase += (
                                            patchExpectedDamage[patch])

                            baseImproveTankerL[baseOther] = (
                                currentImprove - expectedDamageBase)
                            baseTankersL[baseOther] -= 1

                        else:
                            baseImproveTankerE[baseOther] = 0.0

                        # An extra early helicopter ############
                        if baseMaxHelisE[baseOther] > baseHelisE[baseOther]:
                            expectedDamageBase = 0.0
                            baseHelisE[baseOther] += 1

                            # Calculate the improvement of one more tanker at
                            # this base for each patch
                            for patch in range(noPatches):
                                componentNos(
                                        configNos, tankerCovers, heliCovers,
                                        baseTankersE, baseHelisE, baseTankersL,
                                        baseHelisL, updateBases, patch,
                                        maxFire, thresholds, False)

                                if heliCovers[patch, baseOther] <= maxFire:

                                    if (heliCovers[patch, baseOther] <=
                                            thresholds[0]):
                                        configNos[1] += 1
                                    elif (heliCovers[patch, baseOther] <=
                                            thresholds[1]):
                                        configNos[3] += 1

                                    # Get all possible configs and sort them by
                                    # benefit to this patch
                                    getAllConfigsSorted(
                                            configurations, configsP,
                                            baseConfigsPossible,
                                            expectedP[patch], configNos)

                                    expectedDamageBase += expectedDamage(
                                            baseConfigsPossible,
                                            configurations, configsP,
                                            tankerCovers, heliCovers,
                                            baseTankersE, baseHelisE,
                                            baseTankersL, baseHelisL,
                                            expectedP, expFiresComp, weightsP,
                                            patch, time, lookahead, pw,
                                            maxFire, thresholds)

                                else:
                                    expectedDamageBase += (
                                            patchExpectedDamage[patch])

                            baseImproveHeliE[baseOther] = (
                                currentImprove - expectedDamageBase)
                            baseHelisE[baseOther] -= 1

                        else:
                            baseImproveHeliE[baseOther] = 0.0

                        # An extra late helicopter ############
                        if baseMaxHelisL[baseOther] > baseHelisL[baseOther]:
                            expectedDamageBase = 0.0
                            baseHelisL[baseOther] += 1

                            # Calculate the improvement of one more tanker at
                            # this base for each patch
                            for patch in range(noPatches):
                                componentNos(
                                        configNos, tankerCovers, heliCovers,
                                        baseTankersE, baseHelisE, baseTankersL,
                                        baseHelisL, updateBases, patch,
                                        maxFire, thresholds, False)

                                if heliCovers[patch, baseOther] <= maxFire:

                                    if (heliCovers[patch, baseOther] <=
                                            thresholds[1]):
                                        configNos[3] += 1

                                    # Get all possible configs and sort them by
                                    # benefit to this patch
                                    getAllConfigsSorted(
                                            configurations, configsP,
                                            baseConfigsPossible,
                                            expectedP[patch], configNos)

                                    expectedDamageBase += expectedDamage(
                                            baseConfigsPossible,
                                            configurations, configsP,
                                            tankerCovers, heliCovers,
                                            baseTankersE, baseHelisE,
                                            baseTankersL, baseHelisL,
                                            expectedP, expFiresComp, weightsP,
                                            patch, time, lookahead, pw,
                                            maxFire, thresholds)

                                else:
                                    expectedDamageBase += (
                                            patchExpectedDamage[patch])

                            baseImproveHeliL[baseOther] = (
                                currentImprove - expectedDamageBase)
                            baseHelisL[baseOther] -= 1

                        else:
                            baseImproveHeliL[baseOther] = 0.0

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

            if baseImproveTankerL[base] > thisUpdateImprovement:
                    thisUpdateImprovement = baseImproveTankerL[base]
                    thisUpdateDType = 1
                    thisUpdateACType = 3
                    thisUpdateIdx = base

            if baseImproveHeliL[base] > thisUpdateImprovement:
                    thisUpdateImprovement = baseImproveHeliL[base]
                    thisUpdateDType = 1
                    thisUpdateACType = 4
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

            elif thisUpdateACType == 3:
                baseTankersL[thisUpdateIdx] += 1

            elif thisUpdateACType == 4:
                baseHelisL[thisUpdateIdx] += 1

            for patch in range(noPatches):
                if updatePatches[patch]:
                    componentNos(
                            configNos, tankerCovers, heliCovers,
                            baseTankersE, baseHelisE, baseTankersL, baseHelisL,
                            updateBases, patch, maxFire, thresholds, False)

                    # Get all possible configs and sort them by benefit to this
                    # patch
                    getAllConfigsSorted(
                            configurations, configsP, baseConfigsPossible,
                            expectedP[patch], configNos)

                    patchExpectedDamage[patch] = expectedDamage(
                            baseConfigsPossible, configurations, configsP,
                            tankerCovers, heliCovers, baseTankersE, baseHelisE,
                            baseTankersL, baseHelisL, expectedP, expFiresComp,
                            weightsP, patch, time, lookahead, pw, maxFire,
                            thresholds)

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

                            if travTime <= thresholds[0]:
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

                                        if travTime <= thresholds[0]:
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

                                        dist = geoDist(
                                            aircraftLocations[thread_id][time][
                                                resource],
                                            baseLocations[base])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime <= thresholds[0]:
                                            if (baseImproveTankerE[base] >
                                                nextBest):

                                                nextBest = baseImproveTankerE[
                                                    base]
                                        else:
                                            if (baseImproveTankerL[base] >
                                                nextBest):

                                                nextBest = baseImproveTankerL[
                                                    base]

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

                            if travTime <= thresholds[0]:
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

                                        if travTime < thresholds[0]:
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

                                        dist = geoDist(
                                        aircraftLocations[thread_id][time][
                                            resource],
                                        baseLocations[base])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime <= thresholds[0]:
                                            if (baseImproveHeliE[base] >
                                                nextBest):

                                                nextBest = baseImproveHeliE[
                                                    base]
                                        else:
                                            if (baseImproveHeliL[base] >
                                                nextBest):

                                                nextBest = baseImproveHeliL[
                                                    base]

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

                            if travTime > thresholds[0]:
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

                                        if travTime < thresholds[0]:
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

                                        dist = geoDist(
                                        aircraftLocations[thread_id][time][
                                            resource],
                                        baseLocations[base])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime <= thresholds[0]:
                                            if (baseImproveTankerE[base] >
                                                nextBest):

                                                nextBest = baseImproveTankerE[
                                                    base]
                                        else:
                                            if (baseImproveTankerL[base] >
                                                nextBest):

                                                nextBest = baseImproveTankerL[
                                                    base]

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

                            if travTime > thresholds[0]:
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

                                        if travTime < thresholds[0]:
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

                                        dist = geoDist(
                                        aircraftLocations[thread_id][time][
                                            resource],
                                        baseLocations[base])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime <= thresholds[0]:
                                            if (baseImproveHeliE[base] >
                                                nextBest):

                                                nextBest = baseImproveHeliE[
                                                    base]
                                        else:
                                            if (baseImproveHeliL[base] >
                                                nextBest):

                                                nextBest = baseImproveHeliL[
                                                    base]

                                if nextBest < minMax:
                                    minMax = nextBest
                                    nextAC = resource

        # Next assignment is a base
        elif thisUpdateDType == 1:
            if (thisUpdateACType == 1):
                for resource in range(noAircraft):
                    nextBest = 0

                    if (resourceTypes[resource] == 0
                        and int(aircraftAssignments[thread_id][time+1][
                                resource][0] == 0)):

                        if canBase[resource, thisUpdateIdx]:
                            dist = geoDist(
                                aircraftLocations[thread_id][time][resource],
                                baseLocations[thisUpdateIdx])
                            travTime = dist / resourceSpeeds[resource]

                            if travTime <= thresholds[0]:
                                # This resource is a candidate

                                # Check potential benefit to other fires
                                for fire in range(noFires[thread_id][time]):
                                    if canFire[resource, fire]:

                                        dist = geoDist(
                                            aircraftLocations[thread_id][time][
                                                resource],
                                            fireLocations[thread_id][time][
                                                fire])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime <= thresholds[0]:
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
                                    if (base != thisUpdateIdx
                                        and canBase[resource, fire]):

                                        dist = geoDist(
                                            aircraftLocations[thread_id][time][
                                                resource],
                                            baseLocations[base])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime <= thresholds[0]:
                                            if (baseImproveTankerE[base] >
                                                nextBest):

                                                nextBest = baseImproveTankerE[
                                                    base]
                                        else:
                                            if (baseImproveTankerL[base] >
                                                nextBest):

                                                nextBest = baseImproveTankerL[
                                                    base]

                                if nextBest < minMax:
                                    minMax = nextBest
                                    nextAC = resource

            elif (thisUpdateACType == 2):
                for resource in range(noAircraft):
                    nextBest = 0

                    if (resourceTypes[resource] == 1
                        and int(aircraftAssignments[thread_id][time][
                                resource][0] == 0)):

                        if canBase[resource, thisUpdateIdx]:
                            dist = geoDist(
                                aircraftLocations[thread_id][time][resource],
                                baseLocations[thisUpdateIdx])
                            travTime = dist / resourceSpeeds[resource]

                            if travTime <= thresholds[0]:
                                # This resource is a candidate

                                # Check potential benefit to other fires
                                for fire in range(noFires[thread_id][time]):
                                    if canFire[resource, fire]:
                                        dist = geoDist(
                                            aircraftLocations[thread_id][time][
                                                resource],
                                            fireLocations[thread_id][time][
                                                fire])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime < thresholds[0]:
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
                                    if (base != thisUpdateIdx
                                        and canBase[resource, fire]):

                                        dist = geoDist(
                                            aircraftLocations[thread_id][time][
                                                resource],
                                            baseLocations[base])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime <= thresholds[0]:
                                            if (baseImproveTankerE[base] >
                                                nextBest):

                                                nextBest = baseImproveTankerE[
                                                    base]
                                        else:
                                            if (baseImproveTankerL[base] >
                                                nextBest):

                                                nextBest = baseImproveTankerL[
                                                    base]

                                if nextBest < minMax:
                                    minMax = nextBest
                                    nextAC = resource

            elif (thisUpdateACType == 3):
                for resource in range(noAircraft):
                    nextBest = 0

                    if (resourceTypes[resource] == 0
                        and int(aircraftAssignments[thread_id][time+1][
                                resource][0] == 0)):

                        if canBase[resource, thisUpdateIdx]:
                            dist = geoDist(
                                aircraftLocations[thread_id][time][resource],
                                baseLocations[thisUpdateIdx])
                            travTime = dist / resourceSpeeds[resource]

                            if travTime > thresholds[0]:
                                # This resource is a candidate

                                # Check potential benefit to other fires
                                for fire in range(noFires[thread_id][time]):
                                    if canFire[resource, fire]:

                                        dist = geoDist(
                                            aircraftLocations[thread_id][time][
                                                resource],
                                            fireLocations[thread_id][time][
                                                fire])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime <= thresholds[0]:
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
                                    if (base != thisUpdateIdx
                                        and canBase[resource, fire]):

                                        dist = geoDist(
                                            aircraftLocations[thread_id][time][
                                                resource],
                                            baseLocations[base])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime <= thresholds[0]:
                                            if (baseImproveTankerE[base] >
                                                nextBest):

                                                nextBest = baseImproveTankerE[
                                                    base]
                                        else:
                                            if (baseImproveTankerL[base] >
                                                nextBest):

                                                nextBest = baseImproveTankerL[
                                                    base]

                                if nextBest < minMax:
                                    minMax = nextBest
                                    nextAC = resource

            elif (thisUpdateACType == 4):
                for resource in range(noAircraft):
                    nextBest = 0

                    if (resourceTypes[resource] == 1
                        and int(aircraftAssignments[thread_id][time+1][
                                resource][0] == 0)):

                        if canBase[resource, thisUpdateIdx]:
                            dist = geoDist(
                                aircraftLocations[thread_id][time][resource],
                                baseLocations[thisUpdateIdx])
                            travTime = dist / resourceSpeeds[resource]

                            if travTime > thresholds[0]:
                                # This resource is a candidate

                                # Check potential benefit to other fires
                                for fire in range(noFires[thread_id][time]):
                                    if canFire[resource, fire]:

                                        dist = geoDist(
                                            aircraftLocations[thread_id][time][
                                                resource],
                                            fireLocations[thread_id][time][
                                                fire])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime <= thresholds[0]:
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
                                    if (base != thisUpdateIdx
                                        and canBase[resource, fire]):

                                        dist = geoDist(
                                            aircraftLocations[thread_id][time][
                                                resource],
                                            baseLocations[base])

                                        travTime = (dist /
                                                    resourceSpeeds[resource])

                                        if travTime <= thresholds[0]:
                                            if (baseImproveTankerE[base] >
                                                nextBest):

                                                nextBest = baseImproveTankerE[
                                                    base]
                                        else:
                                            if (baseImproveTankerL[base] >
                                                nextBest):

                                                nextBest = baseImproveTankerL[
                                                    base]

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

#        if thread_id == 8:
#            if time == 1:
#                for fire in range(16):
#                    expectedTemp[0, fire, noAircraft - remaining] = fireImproveTankerE[fire]
#                    expectedTemp[1, fire, noAircraft - remaining] = fireImproveHeliE[fire]
#                    expectedTemp[2, fire, noAircraft - remaining] = fireImproveTankerL[fire]
#                    expectedTemp[3, fire, noAircraft - remaining] = fireImproveHeliL[fire]
#                for base in range(noBases):
#                    expectedTemp[4, base, noAircraft - remaining] = baseImproveTankerE[base]
#                    expectedTemp[5, base, noAircraft - remaining] = baseImproveHeliE[base]


#        if thread_id == 6:
#            for fire in range(7):
#                expectedTemp[0, noAircraft - remaining, fire] = fireMaxTankersE[fire]
#                expectedTemp[1, noAircraft - remaining, fire] = fireMaxHelisE[fire]
#                expectedTemp[2, noAircraft - remaining, fire] = fireMaxTankersL[fire]
#                expectedTemp[3, noAircraft - remaining, fire] = fireMaxHelisL[fire]

        #######################################################################
        # Reduce max component capacities (and possibly incremental
        # improvement) based on assignment just made
        for fire in range(noFires[thread_id][time]):
            if canFire[nextAC, fire]:
                dist = geoDist(aircraftLocations[thread_id][time][nextAC],
                               fireLocations[thread_id][time][fire])

                travTime = dist / resourceSpeeds[nextAC]

                if thisUpdateACType == 1 or thisUpdateACType == 3:
                    if (travTime <= thresholds[0]):
                        fireMaxTankersE[fire] -= 1
                        fireMaxTankersE[fire] = max(fireMaxTankersE[fire], 0)

                        if fireMaxTankersE[fire] == fireTankersE[fire]:
                            fireImproveTankerE[fire] = 0

                    else:
                        fireMaxTankersL[fire] -= 1
                        fireMaxTankersL[fire] = max(fireMaxTankersL[fire], 0)

                        if fireMaxTankersL[fire] == fireTankersL[fire]:
                            fireImproveTankerL[fire] = 0

                elif thisUpdateACType == 2 or thisUpdateACType == 4:
                    if (travTime <= thresholds[0]):
                        fireMaxHelisE[fire] -= 1
                        fireMaxHelisE[fire] = max(fireMaxHelisE[fire], 0)

                        if fireMaxHelisE[fire] == fireHelisE[fire]:
                            fireImproveHeliE[fire] = 0

                    else:
                        fireMaxHelisL[fire] -= 1
                        fireMaxHelisL[fire] = max(fireMaxHelisL[fire], 0)

                        if fireMaxHelisL[fire] == fireHelisL[fire]:
                            fireImproveHeliL[fire] = 0

        for base in range(noBases):
            if canBase[nextAC, base]:
                if thisUpdateACType == 1:
                    baseMaxTankersE[base] -= 1

                    if baseMaxTankersE[base] == baseTankersE[fire]:
                        baseImproveTankerE[base] = 0

                elif thisUpdateACType == 2:
                    baseMaxHelisE[base] -= 1

                    if baseMaxHelisE[base] == baseHelisE[fire]:
                        baseImproveHeliE[base] = 0

                elif thisUpdateACType == 3:
                    baseMaxTankersL[base] -= 1

                    if baseMaxTankersL[base] == baseTankersL[fire]:
                        baseImproveTankerL[base] = 0

                elif thisUpdateACType == 4:
                    baseMaxHelisE[base] -= 1

                    if baseMaxHelisL[base] == baseHelisL[fire]:
                        baseImproveHeliL[base] = 0

        # Repeat until assignments complete
        lastUpdateDType = thisUpdateDType
        remaining -= 1

    # If we have aircraft that have not been assigned to anything (not possible
    # if the program is run in its entirety), we need to make sure that they
    # are assigned at least to the nearest base. This covers scenarios where
    # an aircraft provides no benefit from the data analysis. In this case, the
    # conditional probabilities should be calculated and provided to the
    # program otherwise it will not know what to do with these aircraft.
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

    # Compute the resulting patch config weights
    for patch in range(noPatches):
        componentNos(configNos, tankerCovers, heliCovers, baseTankersE,
                     baseHelisE, baseTankersL, baseHelisL, updateBases, patch,
                     maxFire, thresholds, False)

        getAllConfigsSorted(configurations, configsP, baseConfigsPossible,
                            expectedP[patch], configNos)

        patchExpectedDamage[patch] = expectedDamage(
                baseConfigsPossible, configurations, configsP, tankerCovers,
                heliCovers, baseTankersE, baseHelisE, baseTankersL, baseHelisL,
                expectedP, expFiresComp, weightsP, patch, time, lookahead, pw,
                maxFire, thresholds)

    # Compute resulting fire configurations and patch configuration weights
    # given these assignments
    for fire in range(noFires[thread_id][time]):
        configNos[0] = fireTankersE[fire]
        configNos[1] = fireHelisE[fire]
        configNos[2] = fireTankersL[fire]
        configNos[3] = fireHelisL[fire]

        getAllConfigsSorted(configurations, configsE, baseConfigsPossible,
                            expectedE[fire], configNos)
        selectedE[fire] = baseConfigsPossible[0]

#    for patch in range(noPatches):
#        if thread_id == 0 and time == 0:
#            for config in range(len(configsP)):
#                expectedTemp[patch, config] = weightsP[patch, config]


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
def expectedDamage(baseConfigsPossible, configurations, configsP, tankerCovers,
                   heliCovers, baseTankersE, baseHelisE, baseTankersL,
                   baseHelisL, expectedP, expFiresComp, weightsP, patch, time,
                   lookahead, pw, maxFire, thresholds):

    global noConfP

    for config in range(noConfP):
        weightsP[patch, config] = 0.0

    expectedDamageBase = 0.0
    configIdx = 0
    weight_total = 0.0
    weight_c = 0

    while weight_total < 1 and configIdx < noConfP:
        config = configsP[baseConfigsPossible[configIdx]]

        weight_c = potentialWeight(
                configurations, tankerCovers, heliCovers, baseTankersE,
                baseHelisE, baseTankersL, baseHelisL, expFiresComp, patch,
                config, time, lookahead, maxFire, thresholds)

        weight_c = min(weight_c, 1 - weight_total)
        weightsP[patch, baseConfigsPossible[configIdx]] = weight_c

        expectedDamageBase += pw * (expectedP[patch,
                                              baseConfigsPossible[configIdx]]
                                    * weight_c)

        weight_total += weight_c
        configIdx += 1

    return expectedDamageBase


@cuda.jit(device=True)
def componentNos(configNos, tankerCovers, heliCovers, baseTankersE, baseHelisE,
                 baseTankersL, baseHelisL, updateBases, patch, maxFire,
                 thresholds, initial):

    for c in range(4):
        configNos[c] = 0

    # Get best possible config
    for base in range(noBases):
        if tankerCovers[patch, base] <= min(
            maxFire, thresholds[0]):

            if initial:
                updateBases[base] = True

            configNos[0] += baseTankersE[base]
            configNos[2] += baseTankersL[base]

        elif tankerCovers[patch, base] <= maxFire:

            if initial:
                updateBases[base] = True

            configNos[2] += (baseTankersE[base] + baseTankersL[base])

        if heliCovers[patch, base] <= min(
            maxFire, thresholds[0]):

            if initial:
                updateBases[base] = True

            configNos[1] += baseHelisE[base]
            configNos[3] += baseHelisL[base]

        elif heliCovers[patch, base] <= maxFire:

            if initial:
                updateBases[base] = True

            configNos[3] += (baseHelisE[base] + baseHelisL[base])


@cuda.jit(device=True)
def potentialWeight(configurations, tankerCovers, heliCovers, baseTankersE,
                    baseHelisE, baseTankersL, baseHelisL, expFiresComp, patch,
                    config, time, lookahead, maxFire, thresholds):

    weight_c = 0.0

    # Early tankers
    a_cb = 0.0

    for base in range(noBases):
        expFires = 0.0

        for tt in range(time, time + lookahead):
            expFires += expFiresComp[tt, 0, base]

        n_c = max(1, expFires)

        a_cb += ((baseTankersE[base]
                  if tankerCovers[patch, base] <= min(thresholds[0], maxFire)
                  else 0) / n_c)

    if configurations[config - 1, 0] > 0:
        q_c = max(1, configurations[config - 1, 0])
        weight_c = a_cb / q_c
    else:
        weight_c = 1

    # Early helis
    a_cb = 0.0

    for base in range(noBases):
        expFires = 0.0

        for tt in range(time, time + lookahead):
            expFires += expFiresComp[tt, 1, base]

        n_c = max(1, expFires)

        a_cb += ((baseHelisE[base]
                  if heliCovers[patch, base] <= min(thresholds[0], maxFire)
                  else 0) / n_c)

    if configurations[config - 1, 1] > 0:
        q_c = max(1, configurations[config - 1, 1])
        weight_c = min(weight_c, (a_cb / q_c))

    # Late tankers
    a_cb = 0.0

    for base in range(noBases):
        expFires = 0.0

        for tt in range(time, time + lookahead):
            expFires += expFiresComp[tt, 2, base]

        n_c = max(1, expFires)

        a_cb += ((baseTankersE[base]
                  if (tankerCovers[patch, base] <= maxFire and
                      tankerCovers[patch, base] > thresholds[0])
                  else 0) / n_c)

        a_cb += ((baseTankersL[base]
                  if (tankerCovers[patch, base] <= maxFire)
                  else 0) / n_c)

    if configurations[config - 1, 2] > 0:
        q_c = max(1, configurations[config - 1, 2])
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
                      heliCovers[patch, base] > thresholds[0])
                  else 0) / n_c)

        a_cb += ((baseHelisL[base]
                  if (heliCovers[patch, base] <= maxFire)
                  else 0) / n_c)

    if configurations[config - 1, 3] > 0:
        q_c = max(1, configurations[config - 1, 3])
        weight_c = min(weight_c, (a_cb / q_c))

    return weight_c


@cuda.jit(device=True)
def saveState(resourceAssignments, resourceTypes, resourceSpeeds, maxHours,
              tankerCovers, heliCovers, aircraftLocations, accumulatedHours,
              patchLocations, baseLocations, ffdiRanges, fires, fireSizes,
              fireLocations, expectedE, expectedP, expFiresComp,
              configurations, configsE, configsP, baseConfigsMax,
              fireConfigsMax, selectedE, weightsP, states, lambdas, thresholds,
              method, stepSize, time, lookahead, expectedTemp, thread_id):

    global noAircraft
    global noPatches
    global noConfigs
    statesTemp = cuda.local.array(4, dtype=float32)
    updateBases = cuda.local.array(noBases, dtype=b1)
    baseConfigsPossible = cuda.local.array(noConfigs, dtype=int32)
    configNos = cuda.local.array(4, dtype=int32)

    """ STATES - USED """
    """ 1. Expected potential damage (no relocation, no existing) """
    """ Or weighted cover without relocation """
    states[thread_id, time, 0] = potentialNoRelocation(
            configNos, tankerCovers, heliCovers, resourceTypes,
            aircraftLocations, baseLocations, updateBases, configurations,
            configsP, baseConfigsPossible, expFiresComp, expectedP,
            resourceAssignments, math.inf, time, lookahead, thread_id)

    """ 2. Expected existing damage (no suppression) """
    states[thread_id, time, 1] = existingNoSuppression(expectedE, fires[
        thread_id, time])

    statesRelocate(
        resourceAssignments, resourceSpeeds, resourceTypes, maxHours,
        aircraftLocations, accumulatedHours, baseLocations, tankerCovers,
        heliCovers, fires, fireSizes, fireLocations, ffdiRanges, configNos,
        configurations, configsE, configsP, baseConfigsMax, fireConfigsMax,
        thresholds, expFiresComp, expectedE, expectedP, updateBases,
        baseConfigsPossible, selectedE, weightsP, time, stepSize, lookahead,
        thread_id, lambdas, method, expectedTemp, 1, math.inf, statesTemp)

    """ 3. Expected potential damage (relocate to existing, no existing) """
    """ Relocate as much to existing as needed. Remainder cover potential. """
    """ Or weighted cover with relocation to existing """
    """ Or weighted cover of potential with relocation to existing """
    states[thread_id, time, 2] = statesTemp[0]

    """ STATES - UNUSED/INFORMATION """
    """ 4. Expected existing damage (relocate to existing) (NOT USED YET) """
    states[thread_id, time, 3] = statesTemp[1]

#    """ 5. Expected potential damage (relocate to potential) (NOT USED YET) """
#    """ Relocate as much to potential as needed. Remainder cover existing. """
#    statesRelocate(
#        resourceAssignments, resourceSpeeds, resourceTypes, maxHours,
#        aircraftLocations, accumulatedHours, baseLocations, tankerCovers,
#        heliCovers, fires, fireSizes, fireLocations, ffdiRanges, configNos,
#        configurations, configsE, configsP, baseConfigsMax, fireConfigsMax,
#        thresholds, expFiresComp, expectedE, expectedP, updateBases,
#        baseConfigsPossible, selectedE, weightsP, time, stepSize, lookahead,
#        thread_id, lambdas, method, 2, math.inf, statesTemp)
#
#    states[thread_id, time, 4] = statesTemp[0]
#
#    """ 6. Remaining hours: Simple """
#    stateVal = 0
#    for resource in range(noAircraft):
#        stateVal += maxHours[resource] - accumulatedHours[thread_id, time,
#                                                          resource]
#
#    states[thread_id, time, 5] = stateVal


@cuda.jit(device=True)
def potentialNoRelocation(configNos, tankerCovers, heliCovers, resourceTypes,
                          aircraftLocations, baseLocations, updateBases,
                          configurations, configsP, baseConfigsPossible,
                          expFiresComp, expectedP, aircraftAssignments,
                          maxFire, time, lookahead, thread_id):

    global noBases
    global noPatches
    global noAircraft

    stateVal = 0

    baseTankersE = cuda.local.array(noBases, dtype=int32)
    baseHelisE = cuda.local.array(noBases, dtype=int32)
    baseTankersL = cuda.local.array(noBases, dtype=int32)
    baseHelisL = cuda.local.array(noBases, dtype=int32)
    weightsP = cuda.local.array((noPatches, noConfP), dtype=float32)
    thresholds = cuda.local.array(2, dtype=float32)
    thresholds[0] = math.inf
    thresholds[1] = math.inf

    for base in range(noBases):
        baseTankersE[base] = 0
        baseHelisE[base] = 0
        baseTankersL[base] = 0
        baseHelisL[base] = 0

    for resource in range(noAircraft):
        for base in range(noBases):
            if aircraftAssignments[thread_id][time][resource][0] == base + 1:
                dist = geoDist(aircraftLocations[thread_id][time][resource],
                           baseLocations[base])

                if int(resourceTypes[resource]) == 0:
                    if dist <= thresholds[0]:
                        baseTankersE[base] += 1
                    else:
                        baseTankersL[base] += 1
                elif int(resourceTypes[resource]) == 1:
                    if dist <= thresholds[0]:
                        baseHelisE[base] += 1
                    else:
                        baseHelisL[base] += 1

    # We assume that aircraft can attend to any fire. If this doesn't work, we
    # will consider a 20 minute max distance to see how well staying works vs
    # relocating to patches.
    for patch in range(noPatches):
        componentNos(configNos, tankerCovers, heliCovers, baseTankersE,
                     baseHelisE, baseTankersL, baseHelisL, updateBases, patch,
                     math.inf, thresholds, False)

        getAllConfigsSorted(configurations, configsP, baseConfigsPossible,
                            expectedP[patch], configNos)

        stateVal += expectedDamage(
                baseConfigsPossible, configurations, configsP, tankerCovers,
                heliCovers, baseTankersE, baseHelisE, baseTankersL, baseHelisL,
                expectedP, expFiresComp, weightsP, patch, time, lookahead, 1.0,
                maxFire, thresholds)

    return stateVal


@cuda.jit(device=True)
def existingNoSuppression(expectedE, noFires):
    stateVal = 0

    for fire in range(noFires):
        stateVal += expectedE[fire, 0]

    return stateVal


@cuda.jit(device=True)
def statesRelocate(aircraftAssignments, resourceSpeeds, resourceTypes,
                   maxHours, aircraftLocations, accumulatedHours,
                   baseLocations, tankerCovers, heliCovers, fires, fireSizes,
                   fireLocations, ffdiRanges, configNos, configurations,
                   configsE, configsP, baseConfigsMax, fireConfigsMax,
                   thresholds, expFiresComp, expectedE, expectedP, updateBases,
                   baseConfigsPossible, selectedE, weightsP, time, stepSize,
                   lookahead, thread_id, lambdas, method, expectedTemp, dummy,
                   maxFire, returnedStates):

    # Perform a dummy assignment for the next period, ignoring potential fires
    assignAircraft(aircraftAssignments, resourceSpeeds, resourceTypes,
                   maxHours, aircraftLocations, accumulatedHours,
                   baseLocations, tankerCovers, heliCovers, fires, fireSizes,
                   fireLocations, ffdiRanges, configurations, configsE,
                   configsP, baseConfigsMax, fireConfigsMax, thresholds,
                   expFiresComp, expectedE, expectedP, selectedE, weightsP,
                   time, stepSize, lookahead, thread_id, 0, lambdas,
                   method, expectedTemp, dummy)

    # Find out what potential cover will be with these locations
    global noBases
    global noPatches
    global noAircraft

    stateVal = 0

    baseTankersE = cuda.local.array(noBases, dtype=int32)
    baseHelisE = cuda.local.array(noBases, dtype=int32)
    baseTankersL = cuda.local.array(noBases, dtype=int32)
    baseHelisL = cuda.local.array(noBases, dtype=int32)
    weightsP = cuda.local.array((noPatches, noConfP), dtype=float32)
    thresholds = cuda.local.array(2, dtype=float32)
    thresholds[0] = math.inf
    thresholds[1] = math.inf

    for base in range(noBases):
        baseTankersE[base] = 0
        baseHelisE[base] = 0
        baseTankersL[base] = 0
        baseHelisL[base] = 0

    for resource in range(noAircraft):
        for base in range(noBases):
            if aircraftAssignments[thread_id][time+1][resource][0] == base + 1:
                dist = geoDist(aircraftLocations[thread_id][time][resource],
                           baseLocations[base])

                if int(resourceTypes[resource]) == 0:
                    if dist <= thresholds[0]:
                        baseTankersE[base] += 1
                    else:
                        baseTankersL[base] += 1
                elif int(resourceTypes[resource]) == 1:
                    if dist <= thresholds[0]:
                        baseHelisE[base] += 1
                    else:
                        baseHelisL[base] += 1

    # POTENTIAL FIRE COMPONENT
    # We assume that aircraft can attend to any fire. If this doesn't work, we
    # will consider a 20 minute max distance to see how well staying works vs
    # relocating to patches.
    for patch in range(noPatches):
        componentNos(configNos, tankerCovers, heliCovers, baseTankersE,
                     baseHelisE, baseTankersL, baseHelisL, updateBases, patch,
                     math.inf, thresholds, False)

        getAllConfigsSorted(configurations, configsP, baseConfigsPossible,
                            expectedP[patch], configNos)

        stateVal += expectedDamage(
                baseConfigsPossible, configurations, configsP, tankerCovers,
                heliCovers, baseTankersE, baseHelisE, baseTankersL, baseHelisL,
                expectedP, expFiresComp, weightsP, patch, time, lookahead, 1.0,
                maxFire, thresholds)

    returnedStates[0] = stateVal

    # EXISTING FIRE COMPONENT
    stateVal = 0

    for fire in range(int(fires[thread_id][time])):
        config = int(selectedE[fire])
        stateVal += expectedE[fire][0] - expectedE[fire][config]

    returnedStates[1] = stateVal


def unusedStates():
#    """ Now save the expected damages for this assignment """
#    stateVal = 0
#    for fire in range(fires[thread_id][time]):
#        stateVal += expectedE[fire][selectedE[fire]]
#
#    states[thread_id, time, 1] = stateVal

#    shortTermP = 0
#    for config in range(noConfP):
#        for patch in range(noPatches):
#            shortTermP += weightsP[patch, config] * expectedP[patch, config]
#
#    states[thread_id, time, 2] = shortTermP

    """ Other state values to try """

#    """ Phase 2: Sum of Weighted Distance to Fires and Patches AFTER
#    assignment """
#    stateVal = 0
#    # Fires
#    for resource in range(noAircraft):
#        for fire in range(fires[thread_id][time]):
#            dist = geoDist(aircraftLocations[thread_id, time+1, resource],
#                           fireLocations[thread_id, time+1, fire])
#
#            stateVal += dist * expectedE[fire, 0]
#
#    states[thread_id, time, 3] = stateVal
#
#    stateVal = 0
#    # Patches
#    for resource in range(noAircraft):
#        for patch in range(noPatches):
#            dist = geoDist(aircraftLocations[thread_id, time+1, resource],
#                           patchLocations[patch])
#
#            stateVal += dist * expectedP[patch, 0]
#
#    states[thread_id, time, 4] = stateVal
#
#    """" Remaining hours: Weighted """
#    stateVal = 0
#
#    for resource in range(noAircraft):
#        weight = 0
#        for fire in range(fires[thread_id, time]):
#            weight += expectedE[fire, 0]
#
#        for patch in range(noPatches):
#            weight += expectedP[patch, 0]
#
#        stateVal += weight * (maxHours[resource]
#            - accumulatedHours[thread_id, time, resource])
#
#    states[thread_id, time, 5] = stateVal
#
#    """ OPTION 1 """
#    """ Phase 1: Sum of Weighted Distances to Fires and Patches BEFORE
#    assignments"""
#    stateVal = 0
#    # Fires
#    for resource in range(noAircraft):
#        for fire in range(fires[thread_id][time]):
#            dist = geoDist(aircraftLocations[thread_id, time, resource],
#                           fireLocations[thread_id, time, fire])
#
#            stateVal += dist * expectedE[fire, 0]
#
#    states[thread_id, time, 6] = stateVal
#
#    stateVal = 0
#    # Patches
#    for resource in range(noAircraft):
#        for patch in range(noPatches):
#            dist = geoDist(aircraftLocations[thread_id, time, resource],
#                           patchLocations[patch])
#
#            stateVal += dist * expectedP[patch, 0]
#
#    states[thread_id, time, 7] = stateVal

    """ OPTION 2 """
    """ Phase 3: Short-Term Expected Damage from Fires and Patches BEFORE
    assignments"""
#    if time > 0:
#        # Fires
#        states[thread_id, time, 8] = shortTermE
#
#        # Patches
#        states[thread_id, time, 9] = shortTermP
#    else:
#        # Compute the no-relocation option to determine the baseline expected
#        # damage for time 0
#        states[thread_id, time, 8] = 0
#        states[thread_id, time, 9] = 0


"""/////////////////////////////// WRAPPERS ////////////////////////////////"""
""" Main Wrapper """
def simulateROV(paths, sampleFFDIs, patchVegetations, patchAreas,
                patchLocations, baseLocations, resourceTypes, resourceSpeeds,
                maxHours, configurations, configsE, configsP, thresholds,
                ffdiRanges, rocA2PHMeans, rocA2PHSDs, occurrence, initSizeM,
                initSizeSD, initSuccess, extSuccess, tankerDists, heliDists,
                fireConfigsMax, baseConfigsMax, expFiresComp, totalSteps,
                lookahead, mr, mrsd, stepSize, accumulatedDamages,
                randomFFDIpaths, accumulatedHours, fires, initialExtinguished,
                fireStarts, fireSizes, fireLocations, firePatches,
                aircraftLocations, aircraftAssignments, controls, regressionX,
                regressionY, regModels, rSquared, tStatistic, pValue, states,
                costs2Go, lambdas, method, noCont, mapStates, mapC2G,
                discount):

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

    kernelReg = True if regressionY.shape[3] > 1 else False

    """ Initial Monte Carlo Paths """
    print("Now run initial")
    simulateMC(
            paths, d_sampleFFDIs, d_patchVegetations, d_patchAreas,
            d_patchLocations, d_baseLocations, d_resourceTypes,
            d_resourceSpeeds, d_maxHours, d_configurations, d_configsE,
            d_configsP, d_thresholds, d_ffdiRanges, d_rocA2PHMeans,
            d_rocA2PHSDs, d_occurrence, d_initSizeM, d_initSizeSD,
            d_initSuccess, d_extSuccess, d_tankerDists, d_heliDists,
            d_fireConfigsMax, d_baseConfigsMax, d_expFiresComp, d_lambdas,
            totalSteps, lookahead, mr, mrsd, stepSize, accumulatedDamages,
            randomFFDIpaths, accumulatedHours, fires, initialExtinguished,
            fireStarts, fireLocations, firePatches, aircraftLocations,
            aircraftAssignments, controls, d_regressionX, d_regressionY,
            states, costs2Go, method, noControls, discount)

    """ BACKWARD INDUCTION """
    """ Regressions and Forward Path Re-Computations"""
    for tt in range(totalSteps-1, -1, -1):
        """ Compute the regression using the relevant states and
        costs2Go for all but the last and first periods. """
        if tt == totalSteps - 1:
            print('terminal value')
            for control in range(noControls):

                # We are just storing them here. They are not used as we use
                # The MIP runs to determine the best control at the end time.
                xs = numpy.array([states[idx, tt, 0:4]
                                  for idx in range(len(controls[:, tt]))
                                  if controls[idx, tt] == control])

                ys = numpy.array([costs2Go[idx, tt+1]
                                  for idx in range(len(controls[:, tt]))
                                  if controls[idx, tt] == control])

                mapStates[tt, control] = xs
                mapC2G[tt, control] = ys

#                poly_reg = PolynomialFeatures(degree=2)
#                X_ = poly_reg.fit_transform(xs)
#                est = sm.OLS(ys, X_)
#                clf = est.fit()
#                rSquared[tt, control] = clf.rsquared_adj
#                regModels[tt, control] = clf
#
#                coeffs = clf.params
#                regressionX[tt][control, :, 0] =  coeffs
#                regressionY[tt][control] = numpy.zeros(1)
#
#                """ Push the regressions back onto the GPU for reuse in the
#                forward path recomputations """
#                d_regressionX[tt][control] = cuda.to_device(
#                        numpy.ascontiguousarray(regressionX[tt][control]))
#                d_regressionY[tt][control] = cuda.to_device(
#                        numpy.ascontiguousarray(regressionY[tt][control]))

        elif tt < totalSteps - 1 and tt > 0:
            """ We don't need to compute the C2G at the final period as we just
            take the best expected damage in the MIP. This is done in the
            Simulation Kernel """

            print('got here ' + str(tt))

            for control in range(noControls):

                if kernelReg:
                    xs = numpy.array([states[idx, tt, 0:4]
                                      for idx in range(len(controls[:, tt]))
                                      if controls[idx, tt] == control])

                    ys = numpy.array([costs2Go[idx, tt]
                                      for idx in range(len(controls[:, tt]))
                                      if controls[idx, tt] == control])

                    mapStates[tt, control] = xs
                    mapC2G[tt, control] = ys

#                    reg = smooth.NonParamRegression(
#                        xs, ys, method=npr_methods.LocalPolynomialKernel(q=2))
#                    reg.fit()
                    regModels[tt, control] = KernelRidge(
                        kernel='rbf', gamma=0.1, alpha=0.1)
                    regModels[tt, control].fit(xs, ys)
                    rSquared[tt, control] = regModels[tt, control].score(
                        xs, ys)
                    regModels[tt, control] = regModels[tt, control]

                    tempGrid = numpy.meshgrid(
                        numpy.linspace(min(xs[:, 0]), max(xs[:, 0], 50)),
                        numpy.linspace(min(xs[:, 1]), max(xs[:, 1], 50)),
                        numpy.linspace(min(xs[:, 2]), max(xs[:, 2], 50)),
                        numpy.linspace(min(xs[:, 3]), max(xs[:, 3], 50)))

                    regressionY[tt][control] = regModels[tt, control].predict(
                        numpy.array([tempGrid[0].flatten(),
                                     tempGrid[1].flatten(),
                                     tempGrid[2].flatten(),
                                     tempGrid[3].flatten()]).transpose())

                    regressionX[tt, control, :, :, :] = numpy.array([
                        numpy.linspace(min(xs[0]), max(xs[0], 50)),
                        numpy.linspace(min(xs[1]), max(xs[1], 50)),
                        numpy.linspace(min(xs[2]), max(xs[2], 50)),
                        numpy.linspace(min(xs[3]), max(xs[3], 50))])

                    """ Push the regressions back onto the GPU for reuse in the
                    forward path recomputations """
                    d_regressionX[tt, control, :, :] = cuda.to_device(
                            numpy.ascontiguousarray(
                                regressionX[tt, control, :, :]))
                    d_regressionY[tt, control, :] = cuda.to_device(
                            numpy.ascontiguousarray(
                                regressionY[tt, control, :, :, :]))

                else:
                    xs = numpy.array([states[idx, tt, 0:4]
                                      for idx in range(len(controls[:, tt]))
                                      if controls[idx, tt] == control])
                    """ Values for regression are the costs to go at future
                    time periods (this is already optimised w.r.t. the next
                    time step's optimal control) and the single period
                    damage from this time step's control """
                    ys = numpy.array([costs2Go[idx, tt+1]
                                      + accumulatedDamages[idx, tt+1].sum()
                                      - accumulatedDamages[idx, tt].sum()
                                       for idx in range(len(controls[:, tt]))
                                       if controls[idx, tt] == control])

                    mapStates[tt, control] = xs
                    mapC2G[tt, control] = ys

                    poly_reg = PolynomialFeatures(degree=2)
                    X_ = poly_reg.fit_transform(xs)
#                    clf = linear_model.LinearRegression()
#                    clf.fit(X_, ys)
#                    rSquared[tt, control] = clf.score(X_, ys)
                    est = sm.OLS(ys, X_)
                    regModels[tt, control] = est.fit()
                    rSquared[tt, control] = regModels[tt, control].rsquared_adj

                    coeffs = regModels[tt, control].params
                    regressionX[tt][control, :, 0] =  coeffs
#                    print(regressionX[tt][control, :, 0])
                    regressionY[tt][control] = numpy.zeros(1)

                    """ Push the regressions back onto the GPU for reuse in the
                    forward path recomputations """
                    d_regressionX[tt][control] = cuda.to_device(
                            numpy.ascontiguousarray(regressionX[tt][control]))
                    d_regressionY[tt][control] = cuda.to_device(
                            numpy.ascontiguousarray(regressionY[tt][control]))
#                    print('blin')

        elif tt == 0:
            bestControl = 0
            bestC2GStart = math.inf

            for control in range(noControls):
                """ The following is just for record-keeping purposes with
                regards to state. Cost-to-go is still important. """
                xs = numpy.array([states[idx, tt, 0:4]
                                  for idx in range(len(controls[:, tt]))
                                  if controls[idx, tt] == control])

                ys = numpy.array([costs2Go[idx, tt+1]
                                  + accumulatedDamages[idx, tt+1].sum()
                                  - accumulatedDamages[idx, tt].sum()
                                   for idx in range(len(controls[:, tt]))
                                   if controls[idx, tt] == control])

                mapStates[tt, control] = xs
                mapC2G[tt, control] = ys

                currC2GStart = ys.sum()

                if currC2GStart < bestC2GStart:
                    bestControl = control

            controls[:, 0] = (
                    numpy.ones(paths, dtype=numpy.int32) * bestControl)

            print('ROV Complete')

        print('Now run ' + str(tt))
        simulateMC(
                paths, d_sampleFFDIs, d_patchVegetations, d_patchAreas,
                d_patchLocations, d_baseLocations, d_resourceTypes,
                d_resourceSpeeds, d_maxHours, d_configurations, d_configsE,
                d_configsP, d_thresholds, d_ffdiRanges, d_rocA2PHMeans,
                d_rocA2PHSDs, d_occurrence, d_initSizeM, d_initSizeSD,
                d_initSuccess, d_extSuccess, d_tankerDists, d_heliDists,
                d_fireConfigsMax, d_baseConfigsMax, d_expFiresComp, d_lambdas,
                totalSteps, lookahead, mr, mrsd, stepSize, accumulatedDamages,
                randomFFDIpaths, accumulatedHours, fires, initialExtinguished,
                fireStarts, fireSizes, fireLocations, firePatches,
                aircraftLocations, aircraftAssignments, controls,
                d_regressionX, d_regressionY, states, costs2Go, method,
                noControls, discount, start=tt, optimal=True)

    """ Pull the final states and costs 2 go from the GPU and save to an output
    file. For analysis purposes, we need to print our paths to output csv files
    or data dumps (use Pandas?/Spark?)
    The extraction is already done but the saving of the data is not. We save
    the data in the calling routine. """


""" Dynamic MPC """
def simulateMPCDyn(paths, sampleFFDIs, patchVegetations, patchAreas,
                   patchLocations, baseLocations, resourceTypes,
                   resourceSpeeds, maxHours, configurations, configsE,
                   configsP, thresholds, ffdiRanges, rocA2PHMeans, rocA2PHSDs,
                   occurrence, initSizeM, initSizeSD, initSuccess, extSuccess,
                   tankerDists, heliDists, fireConfigsMax, baseConfigsMax,
                   expFiresComp, totalSteps, lookahead, mr, mrsd, stepSize,
                   accumulatedDamages, randomFFDIpaths, accumulatedHours,
                   fires, fireSizes, fireLocations, firePatches,
                   aircraftLocations, aircraftAssignments, controls,
                   regressionX, regressionY, regModels, c2gSaved, lambdas,
                   method, noCont, discount=1, randChoice=True, testControl=0):

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
    d_tankerDists = cuda.to_device(tankerDists)
    d_heliDists = cuda.to_device(heliDists)
    d_maxHours = cuda.to_device(maxHours)
    d_configurations = cuda.to_device(configurations)
    d_baseConfigsMax = cuda.to_device(baseConfigsMax)
    d_fireConfigsMax = cuda.to_device(fireConfigsMax)
    d_expFiresComp = cuda.to_device(expFiresComp)
    d_lambdas = cuda.to_device(lambdas)

    """ The following are created here but are not needed after running the
    kernels found here """
    d_regressionX = cuda.device_array([totalSteps, noControls, 50, 3])
    d_regressionY = cuda.device_array([totalSteps, noControls, 50, 50, 50])
    states = numpy.zeros([paths, totalSteps + 1, 10], dtype=numpy.float32)
    costs2Go = numpy.zeros([paths, totalSteps], dtype=numpy.float32)

    """ Initial Monte Carlo Paths """
    simulateMC(
            paths, d_sampleFFDIs, d_patchVegetations, d_patchAreas,
            d_patchLocations, d_baseLocations, d_resourceTypes,
            d_resourceSpeeds, d_maxHours, d_configurations, d_configsE,
            d_configsP, d_thresholds, d_ffdiRanges, d_rocA2PHMeans,
            d_rocA2PHSDs, d_occurrence, d_initSizeM, d_initSizeSD,
            d_initSuccess, d_extSuccess, d_tankerDists, d_heliDists,
            d_fireConfigsMax, d_baseConfigsMax, d_expFiresComp, d_lambdas,
            totalSteps, lookahead, mr, mrsd, stepSize, accumulatedDamages,
            randomFFDIpaths, accumulatedHours, fires, fireSizes, fireLocations,
            firePatches, aircraftLocations, aircraftAssignments, controls,
            d_regressionX, d_regressionY, states, costs2Go, method, noControls,
            discount, optimal=randChoice, static=testControl)

    """ Pull the final costs 2 go from the GPU ready to save to an output
    file. We save the data in the calling routine. """
    c2gSaved[:] = costs2Go[:, 0]


""" Monte Carlo Routine """
#@jit(parallel=True, fastmath=True)
def simulateMC(paths, d_sampleFFDIs, d_patchVegetations, d_patchAreas,
               d_patchLocations, d_baseLocations, d_resourceTypes,
               d_resourceSpeeds, d_maxHours, d_configurations, d_configsE,
               d_configsP, d_thresholds, d_ffdiRanges, d_rocA2PHMeans,
               d_rocA2PHSDs, d_occurrence, d_initSizeM, d_initSizeSD,
               d_initSuccess, d_extSuccess, d_tankerDists, d_heliDists,
               d_fireConfigsMax, d_baseConfigsMax, d_expFiresComp, d_lambdas,
               totalSteps, lookahead, mr, mrsd, stepSize, accumulatedDamages,
               randomFFDIpaths, accumulatedHours, fires, initialExtinguished,
               fireStarts, fireSizes, fireLocations, firePatches,
               aircraftLocations, aircraftAssignments, controls, d_regressionX,
               d_regressionY, states, costs2Go, method, noControls, discount=1,
               start=0, optimal=False, static=-1):

    # Input values prefixed with 'd_' are already on the device and will not
    # be copied across. Values without the prefix need to be copied across.

    batches = math.ceil(paths / 10000)
    batchAmounts = [10000 for batch in range(batches - 1)]
    batchAmounts.append(paths - sum(batchAmounts))

    expectedTemp = numpy.zeros([2, 50, 27])
#    expectedTemp = numpy.zeros([4, 16, 7])
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
        d_accumulatedDamages = cuda.to_device(
                numpy.ascontiguousarray(accumulatedDamages[
                batchStart:batchEnd, start:(totalSteps+1), :]))
        d_randomFFDIpaths = cuda.to_device(
                numpy.ascontiguousarray(randomFFDIpaths[
                batchStart:batchEnd, start:(totalSteps+1), :]))
        d_accumulatedHours = cuda.to_device(
                numpy.ascontiguousarray(accumulatedHours[
                batchStart:batchEnd, start:(totalSteps+1), :]))
        d_fires = cuda.to_device(numpy.ascontiguousarray(
                fires[batchStart:batchEnd, start:(totalSteps+1)]))
        d_initialExtinguished = cuda.to_device(numpy.ascontiguousarray(
                initialExtinguished[batchStart:batchEnd,
                                    start:(totalSteps+1)]))
        d_fireStarts = cuda.to_device(numpy.ascontiguousarray(
                fireStarts[batchStart:batchEnd, start:(totalSteps+1)]))
        d_fireSizes = cuda.to_device(numpy.ascontiguousarray(
                fireSizes[batchStart:batchEnd, start:(totalSteps+1), :]))
        d_fireLocations = cuda.to_device(numpy.ascontiguousarray(
                fireLocations[batchStart:batchEnd, start:(totalSteps+1), :]))
        d_firePatches = cuda.to_device(numpy.ascontiguousarray(
                firePatches[batchStart:batchEnd, start:(totalSteps+1), :]))
        d_aircraftLocations = cuda.to_device(numpy.ascontiguousarray(
                aircraftLocations[batchStart:batchEnd,
                                  start:(totalSteps+1), :]))
        d_aircraftAssignments = cuda.to_device(numpy.ascontiguousarray(
                aircraftAssignments[batchStart:batchEnd,
                                    start:(totalSteps+1), :]))
        d_controls = cuda.to_device(numpy.ascontiguousarray(
                controls[batchStart:batchEnd, start:(totalSteps+1)]))
        d_states = cuda.to_device(numpy.ascontiguousarray(
                states[batchStart:batchEnd, start:(totalSteps+1), :]))
        d_costs2Go = cuda.to_device(numpy.ascontiguousarray(
                costs2Go[batchStart:batchEnd, start:(totalSteps+1)]))

        # Initialise all random numbers state to use for each thread
        rng_states = create_xoroshiro128p_states(batchSize, seed=1)

        # Compute the paths in batches to preserve memory (see if we can
        # exploit both GPUs to share the computational load)
        simulateSinglePath[blockspergrid, threadsperblock](
                batchSize, totalSteps, lookahead, mr, mrsd, d_sampleFFDIs,
                d_expFiresComp, d_lambdas, d_patchVegetations, d_patchAreas,
                d_patchLocations, d_baseLocations, d_tankerDists, d_heliDists,
                d_ffdiRanges, d_rocA2PHMeans, d_rocA2PHSDs, d_occurrence,
                d_initSizeM, d_initSizeSD, d_initSuccess, d_extSuccess,
                d_resourceTypes, d_resourceSpeeds, d_maxHours,
                d_configurations, d_configsE, d_configsP, d_baseConfigsMax,
                d_fireConfigsMax, d_thresholds, d_accumulatedDamages,
                d_randomFFDIpaths, d_accumulatedHours, d_fires,
                d_initialExtinguished, d_fireStarts, d_fireSizes,
                d_fireLocations, d_firePatches, d_aircraftLocations,
                d_aircraftAssignments, rng_states, d_states, d_controls,
                d_regressionX, d_regressionY, d_costs2Go, start, stepSize,
                method, optimal, static, discount, d_expectedTemp)

        cuda.synchronize()

        # Return memory to the host. We unfortunately have to do this all the
        # time due to the batching requirement to prevent excessive memory
        # use on the GPU
        d_accumulatedDamages.copy_to_host(numpy.ascontiguousarray(
                accumulatedDamages[batchStart:batchEnd,
                                   start:(totalSteps+1), :]))
        d_accumulatedHours.copy_to_host(numpy.ascontiguousarray(
                accumulatedHours[batchStart:batchEnd,
                                 start:(totalSteps+1), :]))
        d_fires.copy_to_host(numpy.ascontiguousarray(
                fires[batchStart:batchEnd, start:(totalSteps+1)]))
        d_initialExtinguished.copy_to_host(numpy.ascontiguousarray(
                initialExtinguished[batchStart:batchEnd,
                                    start:(totalSteps+1)]))
        d_fireStarts.copy_to_host(numpy.ascontiguousarray(fireStarts[
                batchStart:batchEnd, start:(totalSteps+1)]))
        d_fireSizes.copy_to_host(numpy.ascontiguousarray(
                fireSizes[batchStart:batchEnd, start:(totalSteps+1), :]))
        d_fireLocations.copy_to_host(numpy.ascontiguousarray(
                fireLocations[batchStart:batchEnd, start:(totalSteps+1), :]))
        d_firePatches.copy_to_host(numpy.ascontiguousarray(
                firePatches[batchStart:batchEnd, start:(totalSteps+1), :]))
        d_aircraftLocations.copy_to_host(numpy.ascontiguousarray(
                aircraftLocations[batchStart:batchEnd,
                                  start:(totalSteps+1), :]))
        d_aircraftAssignments.copy_to_host(numpy.ascontiguousarray(
                aircraftAssignments[batchStart:batchEnd,
                                    start:(totalSteps+1), :]))
        d_controls.copy_to_host(numpy.ascontiguousarray(
                controls[batchStart:batchEnd, start:(totalSteps+1)]))
        d_states.copy_to_host(numpy.ascontiguousarray(
                states[batchStart:batchEnd, start:(totalSteps+1), :]))
        d_costs2Go.copy_to_host(numpy.ascontiguousarray(
                costs2Go[batchStart:batchEnd, start:(totalSteps+1)]))

    d_expectedTemp.copy_to_host(expectedTemp)
    numpy.set_printoptions(threshold=numpy.nan)
#    print(expectedTemp)
#    print(expectedTemp[0, :, :])
#    print(expectedTemp[1, :, :])
#    print(expectedTemp[2, :, :])
#    print(expectedTemp[3, :, :])
#    print(expectedTemp[4, :, :])
#    print(expectedTemp[5, :, :])
#    print(sum(sum(accumulatedDamages[:,-1,:])/accumulatedDamages.shape[0]))
#    print(fireStarts[0:10])
#    print(firePatches[1, :, 0:16])
#    print(fireSizes[1, :, 0:16])
#    print(fires[1])
#    print(aircraftAssignments[6, :, :])

def analyseMCPaths():
    pass
