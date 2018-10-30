import os
import sys

import pandas as pd
import numpy as np

def main():
    experiment = sys.argv[1]

    scenarios = os.listdir(os.path.join(experiment, 'Outputs'))
    
    for scenario in scenarios:
        scenarioPath = os.path.join(experiment, 'Outputs', scenario)
        
        samples = os.listdir(scenarioPath)
        
        results = []
        noRuns = 0
        
        for sample in samples:
            samplePath = os.path.join(scenarioPath, sample)
            
            sampleResults = []
            
            runs = os.listdir(samplePath)
            noRuns = max(noRuns, len(runs))
            
            for run in runs:
                runPath = os.path.join(samplePath, run, 'Accumulated_Patch_Damage.csv')
                
                accumulatedDamage = pd.read_csv(runPath)
                sampleResults.append(accumulatedDamage[accumulatedDamage.columns[-1]].sum())
                
            results.append(sampleResults)
            
        dataFrame = pd.DataFrame(
                results,
                columns=['RUN_' + str(ii+1) for ii in range(noRuns)],
                index=[ii for ii in range(len(samples))])
        
        dataFrame.to_csv(os.path.join(scenarioPath, 'Summary_Results.csv'))

if True:
    main()
    
    
