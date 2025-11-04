###################################################################################################################
# simmple fct for computing a weighted gini
#
# step 0: header
# step 1: gini fct
###################################################################################################################

# step 0: header
import pandas as pd 
import numpy as np
from sklearn.metrics import auc

# step 1: gini fct

def gini(score, y, weight = pd.Series()):
    """
    Calculate the Gini coefficient for a binary classification model.

    Parameters:
        score (pd.Series): Predicted scores or probabilities.
        y (pd.Series): dep var >= 0 and non missing.
        weight (pd.Series, optional): Weight for each observation (default is an empty Series).

    Returns:
        float: Gini coefficient value.

    Notes:
        - If `weight` is not provided, equal weights are assigned to all observations.
        - The Gini coefficient measures the inequality in the distribution of predicted scores.
        - A Gini coefficient of 0 indicates perfect equality (all scores are the same).
        - A Gini coefficient of 1 indicates perfect inequality.
    """

    if weight.empty: 
         weight =  pd.Series(1, index=score.index)
    
    gini_frame = pd.DataFrame({'weight': weight, 'loss': y, 'score': score})

    # let's break ties randomly
    np.random.seed(42)
    gini_frame['random'] = pd.Series(np.random.uniform(size=gini_frame.shape[0]))
    gini_frame = gini_frame.sort_values(['score', 'random'] ).reset_index(drop = True)

    gini_frame['cdf_loss'] =gini_frame['loss'].cumsum() / gini_frame['loss'].sum()
    gini_frame['cdf_weight'] =gini_frame['weight'].cumsum() / gini_frame['weight'].sum()

    gini = 1- 2* auc(gini_frame['cdf_weight'], gini_frame['cdf_loss'])
    # auc =area under lorenz curve, so 1/2 - auc = area between, double it to get gini

    return gini

