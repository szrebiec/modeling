

###############################################################################################
# Central limit theorem says, add lots of things get a normal RV. Here, we do that to 
#   produce pearson residuals, which can be useful in diagnosing a misfit. 
#
# step 0: header
# fct  1: pearson_quantiled_residuals
###############################################################################################

# step 0: header

import pandas as pd
import numpy as np
import sys
from scipy.stats import linregress

sys.path.append('/home/jovyan/workspace/rnd-cross_lines/workflow_tools/exhibits/')

import equal_wt_lift as bin
import matplotlib.pyplot as plt

# fct 1: pearson_quantiled_residuals

def pearson_quantiled_residuals(indep_var, 
                                pred, 
                                actual, 
                                weight, 
                                q = 1000,
                                title = 'Give your plot a title you git, by setting the title argument!!!', 
                                var_name = 'x'): 
    """
    Calculate Pearson quantiled residuals and create a scatter plot.

    Args:
        indep_var (pd.Series): Independent variable values. should be numeric
        pred (pd.Series): Predicted values.
        actual (pd.Series): Actual response values.
        weight (pd.Series): Weight for each observation.
        q (int, optional): Number of quantiles for binning. Defaults to 1000.
        title (str, optional): Title for the scatter plot. Defaults to 'Give your plot a title you git, by setting the title argument!!!'.
        var_name (str, optional): Name of the independent variable. Defaults to 'x'.

    Returns:
        None
    """
    df = pd.DataFrame({'weight': weight, 
                       'pred': pred, 
                       'actual': actual, 
                       'x': indep_var})
    
    df = bin.equal_weight_bin_rand_splits_for_ties(df, 'weight', 'x', q)  
    # returns df with 1 more var (bin), bad scott: pass series not FRAMES!!!

    fudge_factor = actual.sum() / pred.sum() 
    
    df['wtd_x'] = df['x'] * df['weight']

    grouper = df[['bin', 'pred', 'actual', 'weight', 'wtd_x']].groupby('bin').sum().reset_index()

    grouper['pearson_residual'] = (grouper['actual'] - grouper['pred'] * fudge_factor)/ grouper['weight']
    grouper['x_avg'] = grouper['wtd_x'] / grouper['weight']

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(grouper['x_avg'] , grouper['pearson_residual'], color='blue', label='Residual')

    # Fit a linear regression line
    slope, intercept, r_value, p_value, std_err = linregress(grouper['x_avg'] , grouper['pearson_residual'])
    regression_line = slope * grouper['x_avg']  + intercept
    plt.plot(grouper['x_avg'], regression_line, color='red', label='Regression Line')

    # Add labels and title
    plt.xlabel(f'average {var_name}')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()