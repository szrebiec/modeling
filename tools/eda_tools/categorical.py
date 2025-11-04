       
######################################################################################################################
# The purpose of this code is to implement features for categorical data of moderately high to high cardinality
# This inspired by steps taken by Owen Zhang:https://www.youtube.com/watch?v=LgLcfZjNF44 time circa 40 minute mark
#
# Step 0: Header
# 1: Mapping to Value counts
# 2: Leave one out embedding
# 3:
#
########################################################################################################################

# Step 0: Header
import pandas as pd
import numpy as np
import sys
import os
import pandas as pd
import numpy as np

# fct 1: 
def tree_based_cat_embedding(x, y):
    # arguments:
    #   x: a pd.Series, or numpy array for the indep var
    #   y: a pd.Series, or numpy array for the dep var
    # returns: a sklearn decision tree model, for use in variable transformation
    
    from sklearn.tree import DecisionTreeRegressor
    sum_y = y.sum()
    if y.sum(): 
        sum_obs = x.shape[0]
        n_min = int(np.ceil(sum_obs / sum_y *100))

        #currently VERY BASIC
        tree_model = DecisionTreeRegressor(max_depth = 5, random_state = 42, min_samples_split = n_min)

        #fit tree model with x and y
        tree_model.fit(x.values.reshape(len(x),-1), y)

        return tree_model
    else: 
        print('data to thin to use')

sys.path.append("../diagnostics/")

# Step 1: Mapping to Value counts
def value_count_mapper(Train, Test, cat_name):
    # maps a string variable to the count of the number of obs with that value.
    # intended to be used on the full sample: since the dependent variable is not used there is negligible info leakage
    # function not tested with non-string data... probably will break with float.

    Train = Train.assign(one = 1)
    vc = Train[[cat_name, 'one']].groupby(cat_name, as_index=False).count()
    mapper = {a: b for a, b in zip(vc[cat_name], vc['one'])}

    # kwargs pattern is a way is pack up function arguments into a dictionary (** unpacks).
    # It's needed because Pandas method doesn't take string variable name.
    kwargs = {cat_name + '_value_counts' : Train[cat_name].map(mapper)}
    Train = Train.assign(**kwargs)

    test_ave_response = Test[cat_name].map(mapper)
    test_kwargs = {cat_name + '_value_counts': test_ave_response}
    Test = Test.assign(**test_kwargs)

    return Train, Test

# Step 2: Leave one out embedding
    def lookup_value_mapper(Train, Test, cat_name, dep_var_name):

    Train = Train.assign(one = 1)
    aggregations = {'one' : sum, dep_var_name: sum}
    vc = Train[[cat_name, dep_var_name, 'one']].groupby(cat_name, as_index=False).agg(aggregations)

    #pass values to dictionary for fast mapping
    Nobs_mapper = {A: B for A, B in zip(vc[cat_name], vc['one'])}
    dep_var_mapper = {A: B for A, B in zip(vc[cat_name], vc[dep_var_name])}

    #leave one out scoring for Training data and add in noise
    Noise_vector = Train[dep_var_name].std() * np.random.randn(Train.shape[0]) / np.sqrt(Train[cat_name].map(Nobs_mapper))
    leave_one_out_ave_response = (Train[cat_name].map(dep_var_mapper) - Train[dep_var_name]) / (Train[cat_name].map(Nobs_mapper) - 1)

    # kwargs pattern is a way is pack up function arguments into a dictionary (** unpacks). 
    # It's needed because Pandas method doesn't take string variable name. 
    kwargs = {'l_'+ cat_name + '_avg' : leave_one_out_ave_response + Noise_vector}
    Train = Train.assign(**kwargs)

    #look up scoring for Test data
    test_ave_response = Test[cat_name].map(dep_var_mapper) / Test[cat_name].map(Nobs_mapper)
    test_kwargs = {'l_'+ cat_name + '_avg' : test_ave_response}
    Test = Test.assign(**test_kwargs)

    return Train, Test


# Main for testing
if __name__ == '__main__':
    Base = '/home/ubuntu/code/workflow_tools/Python/feature_engineering'
    #EXAMPLE_DATA_FILE = './CatData.csv'
    EXAMPLE_DATA_FILE = Base + '/CatData.csv'
    CatData = pd.read_csv(EXAMPLE_DATA_FILE)
    Train = CatData[CatData.random < 0.6]
    Test = CatData[CatData.random >= 0.6]

    cat_name_list = ['category']
    dep_var_name = 'y'
    for cat_name in cat_name_list:
    Train, Test = value_count_mapper(Train, Test, cat_name)
    Train, Test = lookup_value_mapper(Train, Test, cat_name, dep_var_name)

    print("\nTrain: Category to 'leave-one-out noisy mean response'\n")
    print(Train.sample(10))
    print("\nTest: Category to 'mean response'\n")
    print(Test.sample(10))
        