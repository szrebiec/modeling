######################################################################################################################################
# The purpose of this code is to provide a function that will automagically produce all summary statistics that I can think of that 
#   are 100% generic. code implicitly assumes a data frame of all charter variables
#
# Step 0: Header
# Func 1: Data frame EDA
#      a: find most common values
#      b: numeric values
#      c: Summarize data
# Func 2: produce correlations
#
######################################################################################################################################

##### Step 0: header #####

import pandas as pd
import numpy as np

#####  Func 1: Data frame EDA #####
#  Func 1a: find most common values #

def MostCommonValues(Series, NVals = 5, MaxStringLength = 10): 
    
    #treat each variable as charater, return the fill rate and the most common values. 
    #Code seems to bug out if data is not string to begin with    
    #should be a better way

    xlist = []
    ylist = []
    
    for x, y in zip(Series.value_counts().keys()[0:NVals], Series.value_counts()[0:NVals].tolist() ): 
        
        if len(x) > MaxStringLength: x = str(x[0:MaxStringLength]) + '... '
        xlist.append(x)
        ylist.append(y)

    MCV = ''    
    for i in range(0,min(NVals, len(xlist) )): 
        MCV += str(xlist[i])+ ': ' + str(ylist[i]) + ', ' 

    return MCV

# Func 1b: numeric values #####
#####  Func 1c: Summarize data #####

def CharSummaryValues(FullSample): 
    # treat each variable as charater, get the number of distinct values. 
        
    CharSummary = FullSample.copy()
    for x in CharSummary.columns: 
        CharSummary[x] = CharSummary[x].astype(str)

    CharSummarStats = CharSummary.describe().transpose()
    CharSummarStats.drop(columns = ['count'], inplace = True)
    CharSummarStats.reset_index(inplace = True)
    CharSummarStats.rename(columns = {'index': 'Variable'}, inplace = True)
    return CharSummarStats

##### Step 3: Sumary of numeric data #####

def NumericSummaryValues(FullSample): 
    # attempt to recode each string attribute as numeric, where successful create summary statistics
    i = 0 
    
    NonNumericVariables = ''
    
    for x in FullSample.columns: 
        try: FullSample[x] = FullSample[x].astype(float)
        except:
            NonNumericVariables = NonNumericVariables + x + ', '

    print('Mean and quantiles will not be produced for non-numeric data. Non numeric variables include: ' + NonNumericVariables)        
    NumericSummary = FullSample.describe().transpose()
    NumericSummary.reset_index(inplace = True)
    NumericSummary.rename(columns = {'index': 'Variable'}, inplace = True)
    return NumericSummary


def SummarizeData(FullSample, NVals = 5, MaxStringLength = 10):
    
    rows = []
    for x in FullSample.columns: # TrueCharAttributes + NumericAttributes: 
        FullSample[x] = FullSample[x].astype(str)
        try: rows.append([x, (FullSample.shape[0] -FullSample[x].isnull().sum()) / FullSample.shape[0],  MostCommonValues(FullSample[x], NVals = NVals, MaxStringLength = MaxStringLength)])
        except: print('MCV issue with '+ x)

    MCVAndFillSummary = pd.DataFrame(rows, columns=["Variable", "Fill Rate", 'Most common Values'])
    
    CharSummarStats = CharSummaryValues(FullSample)
    NumericSummary = NumericSummaryValues(FullSample)
    
    return NumericSummary, pd.merge(CharSummarStats, MCVAndFillSummary, how = 'outer', on = 'Variable')

