
    """ NOT USED """
#    def simulateROVNew(self, exogenousPaths, randCont, endogenousPaths):
#        # Computes the policy map for the problem that is used by the simulator
#        # to make decisions. The decisions are made by determining the state of
#        # they system before plugging into the policy map.
#        region = self.model.getRegion()
#        timeSteps = self.model.getTotalSteps()
#        patches = len(region.getPatches())
#        resources = region.getResources()
#        fires = region.getFires()
#        configsE = self.model.getUsefulConfigurationsExisting()
#        configsP = self.model.getUsefulConfigurationsPotential()
#        sampleFFDIs = self.model.getSamplePaths()
#
#        """ Initial assignment of aircraft to bases (Col 1) and fires (Col 2)
#        A value of zero indicates no assignment (only applicable for fires) """
#        assignments = self.model.getRegion().getAssignments()
#
#        regionSize = region.getX().size
#        samplePaths = (
#                len(sampleFFDIs)
#                if len(sampleFFDIs) > 0
#                else self.model.getRuns())
#        samplePaths2 = self.model.getMCPaths()
#        lookahead = self.model.getLookahead()
#        runs = self.model.getRuns()
#
#        self.finalDamageMaps = [None]*samplePaths
#        self.expectedDamages = [None]*samplePaths
#        self.realisedAssignments = [None]*samplePaths
#        self.realisedFires = [None]*samplePaths
#        self.realisedFFDIs = [None]*samplePaths
#        self.aircraftHours = [None]*samplePaths
#
#        wg = region.getWeatherGenerator()
#
#        for ii in range(samplePaths):
#            self.finalDamageMaps[ii] = [None]*runs
#            self.expectedDamages[ii] = [None]*runs
#            self.realisedAssignments[ii] = [None]*runs
#            self.realisedFires[ii] = [None]*runs
#            self.realisedFFDIs[ii] = [None]*runs
#            self.aircraftHours[ii] = [None]*runs
#
#            for run in range(self.model.getRuns()):
#                damage = 0
#                assignmentsPath = [None]*(timeSteps + 1)
#                assignmentsPath[0] = copy.copy(assignments)
#                firesPath = copy.copy(fires)
#                resourcesPath = copy.copy(resources)
#                activeFires = [fire for fire in firesPath]
#                self.realisedFires[ii][run] = [None]*(timeSteps + 1)
#                self.realisedFires[ii][run][0] = copy.copy(activeFires)
#                self.finalDamageMaps[ii][run] = numpy.empty([timeSteps + 1,
#                                                             patches])
#                self.finalDamageMaps[ii][run][0] = numpy.zeros([patches])
#                self.aircraftHours[ii][run] = numpy.zeros([timeSteps + 1,
#                                                           len(resources)])
#
#                rain = numpy.zeros([timeSteps+1+lookahead, regionSize])
#                rain[0] = region.getRain()
#                precipitation = numpy.zeros([timeSteps+1+lookahead,
#                                             regionSize])
#                precipitation[0] = region.getHumidity()
#                temperatureMin = numpy.zeros([timeSteps+1+lookahead,
#                                              regionSize])
#                temperatureMin[0] = region.getTemperatureMin()
#                temperatureMax = numpy.zeros([timeSteps+1+lookahead,
#                                              regionSize])
#                temperatureMax[0] = region.getTemperatureMax()
#                windNS = numpy.zeros([timeSteps+1+lookahead, regionSize])
#                windNS[0] = region.getWindN()
#                windEW = numpy.zeros([timeSteps+1+lookahead, regionSize])
#                windEW[0] = region.getWindE()
#                FFDI = numpy.zeros([timeSteps+1+lookahead, regionSize])
#                FFDI[0] = region.getDangerIndex()
#                windRegimes = numpy.zeros([timeSteps+1+lookahead])
#                windRegimes[0] = region.getWindRegime()
#                accumulatedDamage = numpy.zeros([timeSteps+1, patches])
#                accumulatedHours = numpy.zeros([timeSteps+1, len(resources)])
#
#                for tt in range(timeSteps):
#                    if len(sampleFFDIs) == 0:
#                        pass
#                    else:
#                        expectedFFDI = sampleFFDIs[ii][:, tt:(tt + lookahea
#                                                              + 1)]
#
#                    """ Use the policy maps/regressions to make assignment
#                    decisions """
#                    """ States """
#
#
#                    """ Determine Control Based on ROV Analysis """
#
#
#                    """ New Assignments """
#
#
#                    """ Simulate to Update """
#                    # Simulate the fire growth, firefighting success and the
#                    # new positions of each resources
#                    damage += self.simulateSinglePeriod(
#                            assignmentsPath, resourcesPath, firesPath,
#                            activeFires, accumulatedDamage, accumulatedHours,
#                            patchConfigs, fireConfigs, FFDI[tt], tt)
#
#                    self.aircraftHours[ii][run][tt + 1] = numpy.array([
#                            resourcesPath[r].getFlyingHours()
#                            for r in range(len(
#                                    self.model.getRegion().getResources()))])
#
#                    # Simulate the realised weather for the next time step
#                    if len(sampleFFDIs) == 0:
#                        wg.computeWeather(
#                                rain, precipitation, temperatureMin,
#                                temperatureMax, windRegimes, windNS, windEW,
#                                FFDI, tt)
#                    else:
#                        FFDI[tt + 1] = sampleFFDIs[ii][:, tt + 1]
#
#                # Store the output results
#                self.finalDamageMaps[ii][run] = accumulatedDamage
#                self.expectedDamages[ii][run] = damage
#                self.realisedAssignments[ii][run] = assignmentsPath
#                self.realisedFFDIs[ii][run] = FFDI
#                self.aircraftHours[ii][run] = accumulatedHours
#
#        for ii in range(samplePaths):
#            for run in range(self.model.getRuns()):
#                """Save the results for this sample"""
#                self.writeOutResults(ii, run)
#
#        self.writeOutSummary()
