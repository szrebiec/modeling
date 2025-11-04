########################################################################################################################
# Pretty straight forward function which runs all categorical embeddings and returns
#
# Step 0: header
#      1: function
########################################################################################################################

#Step 0: header
import pandas as pd
import numpy as np
import sys
import os
sys.path.append("../diagnostics/")
sys.path.append("../feature_engineering/")
from metrics import ks_calc
from categorical import value_count_mapper
from categorical import lookup_value_mapper

#Step 1: function combining all cat embedding functions and running KS stats.
def categorical_embeddings(Train, Test, cat_name_list, dep_var_name, weight=None, run_stats = True):
    #computes categorical embeddings, also produces KS

    for cat_name in cat_name_list:
        Train, Test = value_count_mapper(Train, Test, cat_name)
        Train, Test = lookup_value_mapper(Train, Test, cat_name, dep_var_name)

    KS_list = []
    if run_stats:
        for cat_name in cat_name_list:
            ks_1 = ks_calc(Train[cat_name + '_value_counts'], Train[dep_var_name], weight=weight)
            KS_list.append([cat_name + '_value_counts', ks_1])
            ks_2 = ks_calc(Train['l_'+ cat_name + '_avg'], Train[dep_var_name], weight=weight)
            KS_list.append(['l_'+ cat_name + '_avg', ks_2] )

    KS_stats.sort_values('KS', inplace = True, ascending=False).sort_values('KS', inplace=True, ascending = False)

    return Train, \
           Test, \
           pd.DataFrame(KS_list, columns = ['variable', 'KS'])