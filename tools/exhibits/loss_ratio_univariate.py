#############################################################################################################################################
# This function produces a bivariate graph and compares various binnings for an independent variable
#
#
#
#
#
#############################################################################################################################################

# step 0: header


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import copy

# step 1: plotting function
def var_chart( df:pd.DataFrame, variable:str , models: dict, incurred_loss: str, earned_premium:str, ee:str ):
    """
    This function will take a pandas dataframe, 
    a variable to group by, a dictionary of models,
    and the columns for incurred loss, earned premium,
    and exposure, and will plot the predicted loss ratios for each model and actual loss ratio,
    as well as the exposure on a bar chart. The x-axis will be the variable, and the y-axis will be the loss ratio. 
    The loss ratios will be on the right axis, and the exposure will be on the left axis. The function will return a plot of the data.
    """
    
    #get the loss ratio predictions
    predictions = {}
    for model in list( models ):
        predictions[ model ] = { 'prediction': models[ model ].predict( df ) }
    
    #make subset copy after getting predictions so we don't remove columns needed for GLM predictions
    df_copy = copy.deepcopy( df[[  variable, incurred_loss, earned_premium, ee ]] )
    
    #adjust the loss ratio prediction to get the predicted earned premium
    for model in list( models ):
        df_copy[ model + '_predicted_ep' ] = df_copy[ incurred_loss ] / predictions[ model ][ 'prediction' ]
    
    #grouping column dictionary
    grouping_columns = { 'incurred_loss':'sum',earned_premium:'sum', 'ee':'sum' }
    for model in list( models ):
        grouping_columns[ model + '_predicted_ep' ] = 'sum'
    
    #group the data
    df_grouped = df_copy.groupby( variable ).agg( grouping_columns ).reset_index()
    
    #calculate the loss ratios for GLMs
    for model in list( models ):
        df_grouped[ model + '_predicted_lr' ] = df_grouped[ incurred_loss ] / df_grouped[ model + '_predicted_ep' ]
        lrr_denom = df_grouped[ incurred_loss ].sum() / df_grouped[ model + '_predicted_ep' ].sum()
        df_grouped[ model + '_predicted_lrr' ] = df_grouped[ model + '_predicted_lr' ] / lrr_denom
    
    #calculate the actual loss ratios relativity
    df_grouped['actual_lr'] = df_grouped['incurred_loss'] / df_grouped[earned_premium]
    lrr_denom = df_grouped['incurred_loss'].sum() / df_grouped[earned_premium].sum()
    df_grouped['actual_lrr'] = df_grouped['actual_lr'] / lrr_denom
    
    #plot the predictions and tagret lines on the first axis, ee as a bar chart on the second axis, with variable as the x-axis
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    df_grouped.plot(x=variable, y=ee, kind='bar', ax=ax1, color = 'green')
    df_grouped.plot(  x=variable,  y=[ 'actual_lrr' ], ax=ax2, secondary_y=False, color = 'red')
    for model in list( models ):
        df_grouped.plot(  x=variable,  y=[ model + '_predicted_lrr' ], ax=ax2, secondary_y=False)
    plt.show()
    
    return df_grouped

if __name__ == '__main__': 
    
    #create dummy pandas data
    df = pd.DataFrame({
        'written_premium': np.random.uniform( 1000, 2000, 1000 ),
        'ee': np.random.uniform( .9, 1, 1000 )
    })
    df['incurred_loss'] = df['written_premium'] * np.random.uniform( .8, 1.02, 1000 )
    df['earned_premium'] = df['written_premium'] * df['ee']
    df['loss_ratio'] = df['incurred_loss'] / df['earned_premium']
    df['covA_factor'] = ( df['written_premium'] / 1000 ) * np.random.uniform( .8, 1.1, 1000 )
    df['covA'] =  df['covA_factor'] * np.log( df['written_premium'] ) * 100
    df['adj_ep'] = df['earned_premium']  / df['covA_factor']
    df['target'] = df['incurred_loss'] / df['adj_ep']
    
    #bin the covA variable
    df['covA_group_10'] = pd.cut( df['covA'], bins=10 )
    
    
    #build a tweedie model
    glm_model1 = smf.glm(formula='target ~ covA', 
                        data=df, 
                        family=sm.families.Tweedie(var_power=1.5), 
                        freq_weights=np.asarray( df[ 'adj_ep' ] )
                        ).fit()
    
    #build a tweedie model
    df['covA2'] = df['covA'] - np.random.uniform( 0, 100, 1000 )
    glm_model2 = smf.glm(formula='target ~ covA2', 
                        data=df, 
                        family=sm.families.Tweedie(var_power=1.5), 
                        freq_weights=np.asarray( df[ 'adj_ep' ] )
                        ).fit()
    #plot one GLMs
    var_chart( df, 'covA_group_10', {'model1': glm_model1}, 'incurred_loss', 'earned_premium', 'ee' )
    
    #plot two GLMs
    var_chart( df, 'covA_group_10', {'model1': glm_model1, 'model2': glm_model2}, 'incurred_loss', 'earned_premium', 'ee' )


