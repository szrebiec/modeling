#######################################################################################################################################
#
#
# Steps 0: Header
#       1: quantile grpers
#       2: wtd quantile grper
#       3: Graph x vs y (intende to be quantiled)
# Parts 0: Header
#######################################################################################################################################

##### Part 0: Header ####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


##### step 1: simple quantile grper using a rnandom number to break ties

def q_grouper_simple(df, xvar, n_bins):
    # df = all_train.copy()
    # xvar = all_x_vars[0]
    # n_bins = 100
    local = df[[xvar]].copy()
    std = np.std(local[xvar])
    local['x_noisy'] = local[xvar] + np.random.normal(0, std / 1000, local.shape[0])
    local[xvar + '_quantile'] = pd.qcut(local['x_noisy'], n_bins)

    return local[xvar + '_quantile']

##### step 2: simple wtd quantile grper using a rnandom number to break ties, under construction

def q_grouper_weighted(df, xvar, wvar, key_var, n_bins):
    #coming soon
    # df = all_train.copy()
    # xvar = all_x_vars[0]
    # n_bins = 100
    print('fct not finished')
    local = df[[xvar]].copy()
    std = np.std(local[xvar])
    local['x_noisy'] = local[xvar] + np.random.normal(0, std / 1000, local.shape[0])
    local[xvar + '_quantile'] = pd.qcut(local['x_noisy'], n_bins)
    local = pd.merge(local, grouped, how='left', on=xvar + '_quantile')

    return local[xvar + '_quantile']

##### step 3: simple quantile grper using a rnandom number to break ties

def eda_plot_x_vs_y(df, xvar, yvar, nbins=100):
    local = df[[xvar, yvar]].copy()
    local[xvar + '_quantile'] = q_grouper_simple(local, xvar, nbins)

    grped = local.groupby(xvar + '_quantile').mean()
    grped.reset_index(inplace=True, drop=True)

    plt.title('EDA plot ' + xvar + ' vs ' + yvar)
    plt.ylabel(yvar)
    plt.xlabel(xvar + ' quantiled')
    plt.scatter(grped[xvar], grped[yvar])


