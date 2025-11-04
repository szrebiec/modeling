########################################################################################################
# here we will develop a tool to compute the quality of an individual attribute for improving outcomes
# using KS
#
# step 0: header
# fnct 1: ks_select function, which uses KS to identify and rank each variable's predictive strength
# step 2: test
########################################################################################################

# step 0: header
import sys
import numpy as np
import pandas as pd

sys.path.append('/home/jovyan/workspace/rnd-cross_lines/workflow_tools/exhibits/')

import ks 

# fnct 1: ks_select function, which uses KS to identify and rank each variable's predictive strength
def ks_select(df, weight_name, y_name, x_var_list = [], using_rand_hurdle = True): 
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic for a list of variables, returning a dataframe ordered based on 
    the size of the statistic. The data fram also includes material to diagnose the strenth as an increasing variable and as a decreasing variable. By default any cariable that underperforms a random number will be eliminated. 

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        weight_name (str): The name of the weight column.
        y_name (str): The name of the dependent variable column.
        x_var_list (list, optional): List of independent variable names. If empty, all numeric columns will be used.
        using_rand_hurdle (bool, optional): Whether to use a random hurdle for benchmarking. Default is True.

    Returns:
        pd.DataFrame: A DataFrame with KS statistics for each variable. the KS for using the varialbe as an increasing variable and the ks as a dercreasing variables

    Example:

    
        df = pd.read_csv('data.csv')
        result = ks_select(df,  weight_name = 'adjep300k', y_name = 'loss', x_var_list = [])
        print(result)
    """    

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    local = df.select_dtypes(include=numerics)

    if len(x_var_list) == 0: 
        x_var_list = local.columns.to_list()

    rec_list = []
    for x in x_var_list: 
        results = dict()

        results['variable'] = x

        results['ks'], results['ks_dwn'], results['ks_up'] = ks.ks_statistic(local[x], local[y_name], local[weight_name], return_directional = True)
        rec_list.append(results)

    biv_lift = pd.DataFrame(rec_list) #

    if using_rand_hurdle: 
        local['random_hurdle'] = np.random.uniform(0, 1, local.shape[0])
        bnch_mrk = ks.ks_statistic(local['random_hurdle'], local[y_name], local[weight_name])
        biv_lift = biv_lift[biv_lift['ks'] > bnch_mrk]
                                          
    return biv_lift.sort_values('ks', ascending = False)

# step 2: test
if __name__ == '__main__': 

    sys.path.append('/home/jovyan/workspace/rnd-cross_lines/gbm_identify_pa_oppurtunities/data_pipelines')

    import data_loader_gbm_improvements as dat
    train, _, _ = dat.load_coll_data()

    testing = train[['loss', 'adjep300k', 'driver_minus_claims', 'f_lane_dept_wng', 'prior_ins_act_tbl_fct', 
                'youth_driver_cnt_per_vehcnt_calc', 'pts_mcy_cap_per_drvrcnt_calc']]

    X = ks_select(testing, weight_name = 'adjep300k', y_name = 'loss', x_var_list = [])
    print(X)

