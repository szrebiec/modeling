########################################################################################################
# module contains code to produce the standard_lift charts used at Mercury
# Sample syntax: 
#           mc_lift_chart(loss_series, pred_loss_series, bin_series, weight_series)
#
# step 0: header
# step 1: helper function for fixed bins of pred_lrr
# step 2: main lift chart fct
#
# testing in the wild
########################################################################################################

# step 0: header
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np

# step 1: helper function for fixed bins of pred_lrr
def generic_fixed_width_binner(score, bw = 0.10, n_bins = 5):
    """
    Bin scores into fixed-width intervals.

    This function takes a series of scores and bins them into fixed-width intervals cerntered about 1.0.
    The width of each bin is determined by the `bw` parameter, and the number of bins
    is determined by the `n_bins` parameter.

    Parameters:
        score (pd.Series): The series of scores to be binned.
        bw (float, optional): The width of each bin. Default is 0.10.
        n_bins (int, optional): The number of bins. Default is 5.

    Returns:
        pd.Series: A series with the same index as `score`, containing the bin labels.

    The function works as follows:
    - Initializes a series `bin` with the same length as `score`, filled with 'MT'.
    - Sets the bin label for scores less than or equal to the lower edge case.
    - Iterates through the range of bins, setting the bin labels for each interval.
    - Sets the bin label for scores greater than the upper edge case.

    Example:
    >>> score = pd.Series([0.85, 0.95, 1.05, 1.15, 1.25])
    >>> generic_fixed_width_binner(score)
        0    <= -20.0%
        1    -20.0% to -10.0%
        2    -10.0% to 0.0%
        3     0.0% to 10.0%
        4    >20.0%
    """
    bin = pd.Series([np.nan] * len(score))
    
    #initialize edge case
    
    catergories = []
    bin.loc[score <= 1- (n_bins /2 -1)* bw ] = f'<= -{str(np.round((n_bins /2 -1) *bw*100))}%'
    #print( f'<= -{str(np.round((n_bins /2 -1) *bw*100))}%')
    catergories.append(f'<= -{str(np.round((n_bins /2 -1) *bw*100))}%')
    low_edge = 1- (n_bins /2 -1)* bw 
    
    for counter in range(n_bins - 2): 
        bin.loc[(score <= low_edge + bw) & (score > low_edge )] = f'{str(np.round(low_edge*100 -100))}% to {str(np.round((low_edge + bw) *100 -100))}%'
        catergories.append(f'{str(np.round(low_edge*100 -100))}% to {str(np.round((low_edge + bw) *100 -100))}%')
        #print(f'{str(np.round(low_edge*100 -100))}% to {str(np.round((low_edge + bw) *100 -100))}%')
        low_edge += bw
    
    bin.loc[score > low_edge] = f'>{str(np.round((n_bins /2 -1) *bw *100))}%'
    catergories.append(f'>{str(np.round((n_bins /2 -1) *bw *100))}%')
    #print(f'>{str(np.round((n_bins /2 -1) *bw *100))}%')
    ordered_bin = pd.Categorical(bin, categories=catergories, ordered=True)
    
    return pd.Series(ordered_bin)


def manual_binner(score, break_pts = [0.7, 0.85, 1.0, 1.15, 1.3], name_list = []): 
    
    bin = pd.Series([0] * len(score))
    for b in break_pts: 
        bin += (score > b) + 0.0
    
    base_values = [x for x in range(len(break_pts) +1)]
    if len(name_list) >0: 
        values = name_list
    else: 
        values = base_values
    
    mapper = dict()
    for key, value in zip(base_values, values): 
        mapper[key] = value
    
    return bin.map(mapper)

# step 2: main lift chart fct
def lift_chart(loss_series, 
               pred_loss_series, 
               weight_series, 
               bin_series = pd.Series(), 
               bw = 0.1, 
               n_bins = 5):
    
    '''
    This function creates a lift chart for a given set of series. It groups the data by bins, calculates loss ratios and 
    weights, and then plots the actual and updated loss ratio relativities along with the portion of predicted loss.

    This code was made for a residual model, where loss, weight, and predicted loss are all expected to have the same average. 
    Other use cases have not been tested.  

    Parameters:
        loss_series (pd.Series): The series of loss values.
        pred_loss_series (pd.Series): The series of predicted loss values.
        bin_series (pd.Series): The series of bin values.
        weight_series (pd.Series): The series of weight values.
        n_bins (pd.Series): number of bins to use
    Returns:
        pd.DataFrame: A DataFrame with the calculated values and the bin as index.
    '''
    
    import matplotlib.pyplot as plt
    
    if bin_series.empty: 
        bin_series = generic_fixed_width_binner(pred_loss_series / weight_series, bw, n_bins)
    
    local = pd.DataFrame({'loss': loss_series, 'pred_loss': pred_loss_series, 'bin': bin_series, 'weight': weight_series, 'pred_order': pred_loss_series / weight_series})
    
    local = local.sort_values(['pred_order'])
    local.drop(columns = ['pred_order'], inplace = True)
    grouper = local.groupby('bin').sum().reset_index()
    grouper = grouper.sort_values(['bin'])
    
    grouper['predicted_lr'] = grouper['pred_loss'] / grouper['weight']
    grouper['pct_wt'] = grouper['weight'] /grouper['weight'].sum()
    grouper['actual_lr'] = grouper['loss'] / grouper['weight']
    grouper['updated_lr']= grouper['loss'] /grouper['pred_loss']
    
    # Create a figure and axes
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # First broken line plot (blue)
    ax1.plot(grouper['bin'], grouper['actual_lr'], color='blue', linestyle='-', label='Current loss ratio relativity')
    
    # Second broken line plot (orange)
    ax1.plot(grouper['bin'], grouper['updated_lr'], color='orange', linestyle='-', label='Loss ratio relativity using updated score')
    
    # Create a secondary axis for the bar plot
    ax2 = ax1.twinx()
    ax2.bar(grouper['bin'], grouper['pct_wt'], color='gray', alpha=0.5, label='portion of ma_2_0 predicted loss')
    
    # Set axis labels and title
    ax1.set_xlabel('Bin',fontsize=24)
    ax1.set_ylabel('Loss ratio relativity', fontsize=18)
    ax2.set_ylabel('portion of ma_2_0 predicted loss', color='gray')
    plt.title('Collision residual model (hold out data)', fontsize=24)
    
    # Add legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    
    ax1.set_xticklabels(grouper['bin'], rotation=45)
    ax1.set_ylim(0, 2.0)
    ax2.set_ylim(0, 1.0)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    display(grouper)
        
    return grouper