###############################
# Simple fct to create a stacked lorenz curve
#
# step 0: header
# step 1: binning_function
# step 1: binning_function
# step 3: run a test case
#####################################################################################################################

# step 0: header
import pandas as pd
import numpy as np

# step 1: binning_function

def find_equal_weight_bin(df, weight_var_name, score_var_name, q): 
    """
    Computes equal-weight bins based on scores and cumulative weights.

    Args:
        df (pd.DataFrame): Input DataFrame containing relevant columns.
        weight_var_name (str): Name of the weight column.
        score_var_name (str): Name of the score column.
        q (int): Number of bins (quantiles) to create.

    Returns:
        pd.DataFrame: DataFrame with additional columns for cumulative weight (cdf_wt)
                      and bin assignments (bin).
    """    
    
    df_sorted = df.sort_values([score_var_name])
    total_weight = df_sorted[weight_var_name].sum()
    df_sorted['cdf_wt'] = df_sorted[weight_var_name].cumsum() / total_weight
    
    score_q_list = []
    for i in range(1, q): 
        score_q_list.append((df_sorted[score_var_name][df_sorted['cdf_wt'] >  i / q]).min())
    
    score_q_list = list(set(score_q_list)) # deduping
    
    df_sorted['bin'] = 0.0

    for quant in score_q_list: 
        df_sorted['bin'] += (df_sorted[score_var_name] > quant)
        
    return df_sorted


'''
def equal_weight_bin_rand_splits_for_ties(df, weight_var_name, score_var_name, q): 
    """
    Computes equal-weight bins based on scores and cumulative weights.

    Args:
        df (pd.DataFrame): Input DataFrame containing relevant columns.
        weight_var_name (str): Name of the weight column.
        score_var_name (str): Name of the score column.
        q (int): Number of bins (quantiles) to create.

    Returns:
        pd.DataFrame: DataFrame with additional columns for cumulative weight (cdf_wt)
                      and bin assignments (bin).
    """    
    
    df['random_order'] = np.random.uniform(low=0.0, high=1.0, size=len(df))
    df_sorted = df.sort_values([score_var_name, 'random_order'])
    total_weight = df_sorted[weight_var_name].sum()

    df_sorted['bin'] =  np.clip(np.floor(q * df_sorted[weight_var_name].cumsum() /( total_weight)) , a_min=None, a_max=q-1)
    # create a bin starting at 0, that represents the quantile. max, goes to q-1 not q, (edge effect)

    del df_sorted['random_order']
        
    return df_sorted
'''

# step 2: create a lift chart using equal weight binning: 
def lift_chart(df, 
               score_name, 
               wt_name, 
               loss_name,
               N_bins = 10, 
               bin_name = 'bin', 
               num_var_list = [], 
               x_label = 'Add a a x label by specifying x_label, you git!', 
               y_label = 'Add a a y label by specifying y_label, you git!', 
               title = 'Add a title by specifying title, you git!'): 
    """
    Create a lift chart for a given dataset.

    Args:
        df (pd.DataFrame): Input DataFrame containing relevant data.
        score_name (str): Name of the score column. This should predict loss per weight-modify as needed. 
        wt_name (str): Name of the weight column.
        loss_name (str): Name of the loss column.
        N_bins (int, optional): Number of bins for equal-weight binning. Defaults to 10.
        bin_name (str, optional): Name of the bin column. Defaults to 'bin'.
        num_var_list (list, optional): List of additional numeric variable names. Defaults to [].
        x_label (str, optional if you are a git): X-axis label. Defaults to 'Add a x label by specifying x_label, you git!'.
        y_label (str, optional if you are a git): Y-axis label. Defaults to 'Add a y label by specifying y_label, you git!'.
        title (str, optional if you are a git): Title of the plot. Defaults to 'Add a title by specifying title, you git!'.

    Returns:
        None
    """
    
    df = find_equal_weight_bin(df, wt_name, score_name, N_bins) 
    
    grouper = df[[bin_name] + [score_name, wt_name, loss_name] + num_var_list].groupby(bin_name).sum().reset_index().sort_values('bin')
    
    grouper[f'{loss_name} per {wt_name}'] = grouper[loss_name] / grouper[wt_name]
   
    display(grouper[[bin_name, f'{loss_name} per {wt_name}', loss_name] + num_var_list])
    
    import matplotlib.pyplot as plt
    
    # Create a bar plot
    plt.bar(grouper[bin_name], grouper[f'{loss_name} per {wt_name}'], color='maroon', alpha=0.7)

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()
    
    
# step 2: create a lift chart using equal weight binning, plotting predicted and actual LRR: 
def lift_pred_vs_actual_chart(df, 
                              score_name, 
                              wt_name, 
                              pred_loss_name, 
                              loss_name,
                              N_bins = 10, 
                              bin_name = 'bin', 
                              num_var_list = [], 
                              x_label = 'Add a a x label by specifying x_label, you git!', 
                              y_label = 'Add a a y label by specifying y_label, you git!', 
                              title = 'Add a title by specifying title, you git!',
                              return_chart = False): 
    """
    Create a lift chart comparing predicted loss and actual loss per weight.

    Args:
        df (pd.DataFrame): Input DataFrame containing relevant data.
        score_name (str): Name of the score column. This should predict loss per weight-modify as needed. 
        wt_name (str): Name of the weight column.
        pred_loss_name (str): Name of the predicted loss column.
        loss_name (str): Name of the actual loss column.
        N_bins (int, optional): Number of bins for equal-weight binning. Defaults to 10.
        bin_name (str, optional): Name of the bin column. Defaults to 'bin'.
        num_var_list (list, optional): List of additional numeric variable names. Defaults to [].
        x_label (str, optional if you are a git): X-axis label. Defaults to 'Add a x label by specifying x_label, you git!'.
        y_label (str, optional if you are a git): Y-axis label. Defaults to 'Add a y label by specifying y_label, you git!'.
        title (str, optional if you are a git): Title of the plot. Defaults to 'Add a title by specifying title, you git!'.

    Returns:
        None
    """    
    
    df = find_equal_weight_bin(df, wt_name, score_name, N_bins) 
    
    grouper = df[[bin_name] + [score_name, wt_name, loss_name, pred_loss_name] + num_var_list].groupby(bin_name).sum().reset_index().sort_values('bin')
    
    grouper[f'{loss_name} per {wt_name}'] = grouper[loss_name] / grouper[wt_name]   
    grouper[f'{pred_loss_name} per {wt_name}'] = grouper[pred_loss_name] / grouper[wt_name]
    
    display(grouper[[bin_name, f'{loss_name} per {wt_name}', f'{pred_loss_name} per {wt_name}'] + num_var_list])

    import matplotlib.pyplot as plt

    # Create a bar plot
    plt.bar(grouper[bin_name] - 0.4/2, grouper[f'{loss_name} per {wt_name}'], color='maroon', alpha=0.7, label=f'{loss_name} per {wt_name}', width = 0.4)
    plt.bar(grouper[bin_name] + 0.4/2, grouper[f'{pred_loss_name} per {wt_name}'], color='darkgoldenrod', alpha=0.7, label=f'{pred_loss_name} per {wt_name}', width = 0.4)    

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()
    
    if return_chart: 
        return grouper
    
def scatter_plot_eq_wt(df, 
                       score_name, 
                       wt_name, 
                       loss_name,
                       N_bins = 10, 
                       bin_name = 'bin', 
                       x_label = 'Add a a x label by specifying x_label, you git!', 
                       y_label = 'Add a a y label by specifying y_label, you git!', 
                       title = 'Add a title by specifying title, you git!'
                      ): 

    import matplotlib.pyplot as plt
                       
    # Create a scatter plot
    
    df = find_equal_weight_bin(df, wt_name, score_name, N_bins) 
    
    df['one'] = 1.0
    grouper = df[[bin_name] + [score_name, wt_name, loss_name, 'one']].groupby(bin_name).sum().reset_index().sort_values('bin')
    
    grouper['lrr'] = grouper[loss_name] / grouper[wt_name]
    grouper['score_name'] = grouper[score_name] / grouper['one']
        
    plt.scatter(grouper['score_name'], grouper['lrr'], label = 'empirical')

    # Fit a linear regression line (you can use other regression models if needed)
    coefficients = np.polyfit(grouper['score_name'], grouper['lrr'], 1)
    fit_line = np.poly1d(coefficients)
    plt.plot(grouper['score_name'], fit_line(grouper['score_name']), color='red', label='Fit Line')

    # Add title and axis labels
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Show the legend
    plt.legend()

    # Display the plot
    plt.show()
