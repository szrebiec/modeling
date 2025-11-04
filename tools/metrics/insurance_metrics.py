####################################################################################################################################
# here we return extra args for easy plotting
# Nothing deep here, just want it more g2g for practical use cases
#
# Step 0: header
#      1: pr-auc
#      2: ginni
#     
####################################################################################################################################

# step 0: header

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve


# fnct 1: pr auc

def pr_auc(df, y_name, score_name):  
    precision, recall, _ = precision_recall_curve(df[y_name], df[score_name])
    return auc(recall, precision), precision, recall


# fnct 2: gini

def gini(df, weight_name, y_name, score_name): 
    local = df[[weight_name, y_name, score_name]].copy()
    local = local.sort_values(score_name)
    local['cdf_w'] = local[weight_name].cumsum() / local[weight_name].sum()
    local['cdf_y'] = local[weight_name].cumsum() / local[weight_name].sum()    
    
    return auc(local['cdf_w'], local['cdf_y']), local['cdf_w'], local['cdf_y']

    