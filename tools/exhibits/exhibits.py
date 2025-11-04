################################################################################################
# A collection of frequently used exhibits
#
# Step 0: Header
# fct 1: LorenzCurve
# fct 2: Lazy KS, ties broken randomly
# fct 3: a ROC curve for a model
# fct 3: a PR curve for a model
# fct 4: equal weight binning fct
# fct 5: Insurance gains_chart (under construction
################################################################################################

##### step 0: Header #####

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# fct 1: Lorenz Curve 

def LorenzCurve(df,
                title = 'Lorenz Curve',
                ascending = True, #arg for backwards Lorenz Curve
                LVarName = 'future_incurred',
                WeightName = 'one',
                ScoreName = 'Pred',
                xlabel = 'CDF Weight',
                ylabel = 'CDF loss'):
    # function arguments:  
    # df: dataframe with columns LVarName, WeightName, ScoreName
    # title: title of plot
    # ascending: boolean, if True, plot will be backwards
    # LVarName: name of loss variable
    # WeightName: name of weight variable
    # ScoreName: name of score variable
    # xlabel: label for x axis
    # ylabel: label for y axis    

    import sklearn.metrics as metrics

    Local = df[[LVarName, WeightName, ScoreName]]
    Local.sort_values(ScoreName, ascending = ascending, inplace = True)
    Local['Baseline'] = Local[WeightName].cumsum() / Local[WeightName].sum()
    Local['Model Performance'] = Local[LVarName].cumsum() / Local[LVarName].sum()
    fig, ax = plt.subplots(figsize=(12, 10))
    ModelGini = 2* abs(metrics.auc(Local['Baseline'], Local['Model Performance']) -
                       metrics.auc(Local['Baseline'], Local['Baseline']) )
    ax.plot(Local['Baseline'],  Local['Model Performance'],
            label='Model performance, gini = {ModelGini}'.format(ModelGini = ModelGini),
            linestyle='-',)
    ax.plot(Local['Baseline'],  Local['Baseline'],
            label='Baseline performance,',
            linestyle='-',)

    ax.legend(loc=4)
    ax.set_xlim(0,1)
    ax.set_ylim(0 , 1)
    ax.set_yticks(np.arange(0, 1.1,.1))
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
# fct 2: Lazy KS, ties broken randomly
def LazyKS(XvarName, df = Train1, WtVarName = 'one', LossVarName = 'y'):
    # fct arguments: 
    # XvarName: name of the variable to be evaluated
    # df: dataframe
    # WtVarName: name of the weight variable (default = 'one')
    # LossVarName: name of the loss variable (optional)
    
    dflocal = df[[XvarName, WtVarName, LossVarName]].sort_values([XvarName]).reset_index()
    dflocal['CDFWt'] = dflocal[WtVarName].cumsum() /dflocal[WtVarName].sum()
    dflocal['CDFy'] = dflocal[LossVarName].cumsum() /dflocal[LossVarName].sum()
    
    return max(np.abs(dflocal['CDFy'] - dflocal['CDFWt']))

# fct 3: a ROC curve for a model
def roc_curve(pred, y, title = 'ROC Curve'): 
    # fct arguments: 
    # pred: predicted values
    # y: actual values
    # title: title of plot (default = 'ROC Curve')
    
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()
    
    return roc_auc

# fct 3: a PR curve for a model
def pr_curve(pred, y, title = 'PR Curve'): 
    # fct arguments: 
    # pred: predicted probabilities
    # y: actual labels
    # title: title of plot (default = 'PR Curve')
    # returns: average precision score

    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, thresholds = precision_recall_curve(y, pred)
    average_precision = average_precision_score(y, pred)

    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = %0.3f)' % average_precision)
    plt.title(title)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.show()

    return average_precision

# fct 2: Lazy KS, ties broken randomly
# fct 3: a ROC curve for a model
# fct 3: a PR curve for a model
# fct 4: equal weight binning fct
def equal_weight_bins(x, weight, q = 10):
    # function to create equal weight bins
    # x is the variable to bin
    # weight is the weight of each observation
    # q is the number of bins (default = 10)
    # returns a pandas series with the bin number for each observation

    df = pd.DataFrame({'x': x, 'weight': weight})

    df = df.sort_values(by = 'x')
    df['cum_weight'] = df.weight.cumsum()
    df['bin'] = pd.qcut(df.cum_percent, q, labels = False)

    # create a dictionary that maps the variable to the bin
    bin_dict = dict(zip(df.x, df.bin))

    # map the variable to the bin
    return x.map(bin_dict)

# fct 2: Lazy KS, ties broken randomly
# fct 3: a ROC curve for a model
# fct 3: a PR curve for a model
# fct 4: equal weight binning fct
# fct 5: Insurance gains_chart (under construction

def bar_chart(df, pred_name, y_name, weight_name = 'one', q=10): 

    local = df[[pred_name, y_name, weight_name]].copy()
    local['pred_bin'] = equal_weight_bins(local[pred_name], local[weight_name], q = q)
    
    grped_data = local.groupby('pred_bin').sum().reset_index()
    
    grped_data['avg_pred'] = grped_data[pred_name] / grped_data[weight_name]
    grped_data['avg_y'] = grped_data[y_name] / grped_data[weight_name]

    grped_data[['avg_y']].plot(kind = 'bar', stacked = True)
    plt.show()
    
    display(grped_data)

    return df    
    
    
    
'''
ll_train['bin'] = pd.qcut(all_train['pred'], 
                           q=10, 
                           labels = [f"quantile_{i}" for i in range(1,10+1)])

all_train['wgt'] = 1.0


### Preprocess data for Gains chart
agg_types = {'pred': ['min','max'], all_y_var: ['sum', 'mean'], 'wgt': ['sum']} 
agg_data = all_train.groupby('bin').agg(agg_types)
new_cols = []
for col in agg_data.columns.values:  
    new_cols = new_cols+ ["_".join(col)]
agg_data.columns = new_cols
agg_data['bin'] = np.arange(agg_data.shape[0])
agg_data['bin_size'] = agg_data['wgt_sum']/np.sum(agg_data['wgt_sum']) 

fig, ax1 = plt.subplots(sharex = 'all', sharey='all')

lcolor = 'lightsteelblue'
ax1.set_xlabel('Bin')
ax1.set_ylabel('pct of policies')
ax1.bar(agg_data['bin'], 100*agg_data['bin_size'], color=lcolor)
ax1.tick_params(axis='y')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

rcolor = 'tab:red'
ax2.set_ylabel('Loss Cost', color=rcolor)  # we already handled the x-label with ax1
ax2.plot(agg_data['bin'], 100 *agg_data[all_y_var+ '_mean'],  linestyle='--', marker='o', color=rcolor)
ax2.tick_params(axis='y', labelcolor=rcolor)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Gains Chart', pad=20)
ax1.grid(True)
ax2.grid(False)
ax1.set_ylim(0, 100)
plt.show()    

'''