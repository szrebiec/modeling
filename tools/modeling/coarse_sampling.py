##########################################################################################################
# The mathod of defining an indivisible atom of risk often leads to individual observations being highly 
# correlated, e.g. same policy/module/vehicle but different driver. To combat this the fct here is 
# designed to put correlated observations in the same train/test split. User must specify the atom
#
# step 0: header
# step 1: train_test split using a specified level of granularity
##########################################################################################################

# step 0: header

import pandas as pd
import numpy as np

# step 1: train_test split using a specified level of granularit

def get_train_test_split(df, granularity = ['st', 'policy_num', 'coverage'], size = 0.8): 
    """
    Splits a DataFrame into training and test sets based on specified granularity.

    Args:
        df (pd.DataFrame): The input DataFrame.
        granularity (list, optional): List of columns to consider for granularity. Defaults to ['st', 'policy_num', 'coverage'].
        size (float, optional): Proportion of data to allocate for training (between 0 and 1). Defaults to 0.8.

    Returns:
        pd.DataFrame, pd.DataFrame: Train and test DataFrames.
    """

    np.random.seed(42)

    key_frame = df[granularity][df[granularity].duplicated() == False]
    key_frame['random'] =  np.random.uniform(low=0, high=1, size=(key_frame.shape[0], 1))

    train = pd.merge(df, 
                    key_frame[granularity][key_frame['random'] < size],
                    how = 'inner', 
                    on = granularity )

    test = pd.merge(df, 
                    key_frame[granularity][key_frame['random'] >= size],
                    how = 'inner', 
                    on = granularity )
    
    return train, test