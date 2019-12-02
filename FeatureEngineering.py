from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.stats import rankdata
import numpy as np
import pandas as pd

def feature_normalizer(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    # sample data from normal distribution
    dist_data = np.sort(np.random.normal(loc=mean, scale=std, size=len(x)))
    # unlike sort, with rank identical values gets the same rank
    sort_idx = rankdata(x, method='min') - 1
    # match bewteen x and dist data according to the rank
    return dist_data[sort_idx]


def build_binary_features(df, cols, inplace=False):
    assert isinstance(cols,(str, list)), 'cols must be either a string or a list, but got {}'.format(type(cols))
    df = df.copy() if not inplace else df
    cols = [cols] if not isinstance(cols, list) else cols
    for c in cols:
        assert c in df.keys(), 'feature {} does not exist in df'.format(c)
    for c in cols:
        df = __list2multihot__(df,c)
    return df


def best_binning(df, col_name, target, target_type, num_of_bins=4, inplace=False):
    if not inplace:
        df = df.copy()
    assert isinstance(col_name,str), 'col_name must be a string, but got {}'.format(type(col_name))
    assert col_name in list(df.keys()), '{} not in df'.format(col_name)
    assert target_type in ['numeric','categorical'], 'target_type must be  in [numeric, categorical], but got {}'.format(target_type)
    bins = [-np.inf]
    bins = __get_rules__(df, col_name, target, target_type, num_of_bins)
    bins.append(np.inf)
    df['binned_'+col_name] = pd.cut(df[col_name], bins=bins, labels=False)
    return df


def __get_rules__(df, col_name, target, target_type, num_of_bins):
    if target_type == 'categorical':
        DT = DecisionTreeClassifier(max_leaf_nodes=num_of_bins, random_state=0)
    else:
        DT = DecisionTreeRegressor(max_leaf_nodes=num_of_bins, random_state=0)
    # find best split
    x,y = np.array(list(df[col_name])), list(df[target])
    DT.fit(x[:,np.newaxis],y)
    # get splitting rules
    rules = DT.tree_.threshold.tolist()
    return np.sort(list(set(rules))).tolist()


def __list2multihot__(df,c):
    df_work = df.copy()
    if not isinstance(df.loc[0][c], list):
        df_work[c] = df[c].apply(lambda x: [x])
    # get uniques
    unq = []
    for l in df_work[c]:
        unq.extend(l)
    unq = list(set(unq))
    # build dataframe
    columns = [c+'_'+str(val) for val in unq]
    df1hot = pd.DataFrame(np.zeros([len(df), len(unq)]), columns=columns)
    for idx, l in enumerate(df_work[c]):
        for val in l:
            df1hot.loc[idx, c+'_'+str(val)] = 1
    df = pd.concat((df,df1hot), axis=1)
    return df
