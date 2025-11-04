#####################################################################################################################
# Simple fct to create a stacked lorenz curve
#
# step 0: header
# step 1: lorenz curve function 
#####################################################################################################################

# step 0: header

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/jovyan/workspace/rnd-cross_lines/workflow_tools/exhibits/')
import gini as robin_williams

# step 1: lorenz curve function 
def lorenz_curve(score_list, 
                weight_list , 
                loss_list, 
                title = 'Give your graph a title you git by setting title = X', 
                label_list = [], 
                x_axis_label = 'Cdf Weight', 
                y_axis_label = 'Cdf Loss'):
    '''
    Plot Lorentz curves based on provided scores, weights, and y-values.
    
    Parameters:
        score_list (pd.Series): list of Series containing scores.
        weight_list (pd.Series): list of Series containing weights
        loss_list (pd.Series): list of Series containing y-values.
        title (str, optional if you are a git): Title for the plot.
        label_list (str, optional if you are a git): list of label for the plot. defacto default to a string seq number for model
        x_axis_label (str, optional) defaults to 'Cdf Weight'
        y_axis_label (str, optional) defaults to 'Cdf Loss'
    example: 

    '''
    
    from sklearn.metrics import auc
    
    # Sort the data based on score_1
    if len(label_list) == 0: label_list = ['model ' + str(x) for x in range(len(score_list))]

    for score, weight, loss, label  in zip(score_list, weight_list, loss_list, label_list): 
        sorted_data = pd.DataFrame({'score': score, 'weight': weight, 'y': loss})
        sorted_data.sort_values('score', inplace=True)
        cumulative_weight = sorted_data['weight'].cumsum() / sorted_data['weight'].sum()
        cumulative_y = sorted_data['y'].cumsum() / sorted_data['y'].sum()
        cumulative_weight = pd.Series([0.0] + cumulative_weight.to_list() + [1.0])
        cumulative_y = pd.Series([0.0] + cumulative_y.to_list() + [1.0])
        gini = np.round(robin_williams.gini(score, loss, weight = weight), 3)
        plt.plot(cumulative_weight, cumulative_y, label=label + f'. Gini = {gini}')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='y=x')
    
    # Plot the Lorentz curves
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    

# step 2: run a test case
if __name__ == '__main__': 

    test_data = pd.read_csv('C:\\Users\\SZrebiec\\code\\workflow_tools\\exhibits\\sample_scores.csv')
    lorenz_curve(test_data.score, test_data.weight, test_data.loss, test_data.score_2)