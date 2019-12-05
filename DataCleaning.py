# visualize missing values
import missingno as msno
import time
import numpy as np
import pprint as pp
from termcolor import colored
import pandas as pd

def nan_analysis(df, figure_size=(12, 5)):
    # fig, axs = plt.subplots(3,1)
    # nan ratio in each feature
    msno.bar(df, figsize=figure_size)
    time.sleep(0.2)
    # nan ratio in each row
    msno.matrix(df, figsize=figure_size)
    time.sleep(0.2)
    # plot nan correlation between features
    msno.heatmap(df, figsize=figure_size)


def clean_nans(df, row_nan_threshold=0.3, col_nan_threshold=0.3, feature_type=None, categorical_group_size=10,
               print_feature_type=True, method='median', alternative_method='minimum_impact', num_of_bins=5,
               inplace=False):
    # Asserts
    valid_types = ['id', 'categorical', 'numeric', 'str']
    assert method in ['median', 'best_predictor', 'minimum_impact'], 'There is no method named {}'.format(method)
    assert alternative_method in ['median',
                                  'minimum_impact'], 'alternative_method must be in [median,minimum_impact], but got {}'.format(
        alternative_method)
    assert isinstance(row_nan_threshold, float), 'row_nan_threshold must be float, but got {}'.format(
        type(row_nan_threshold))
    assert isinstance(col_nan_threshold, (float, dict)), 'col_nan_threshold must be float or dict, but got {}'.format(
        type(col_nan_threshold))
    assert isinstance(feature_type,
                      dict) or feature_type is None, 'feature_type must be either dict or None, but got {}'.format(
        type(feature_type))
    if isinstance(feature_type, dict):
        assert all(x in valid_types for x in set(
            feature_type.values())), 'The valid types are {}, but feature_type contains {}'.format(
            set(valid_types, feature_type.values()))
        
    if not inplace:
        df = df.copy()
    # Drop nans
    print('Droping rows and columns with too many nans...')
    # drop rows
    original_len = len(df)
    df.dropna(axis=0, thresh=len(df.keys()) - int(len(df.keys()) * row_nan_threshold), inplace=True)
    print("{} rows was dropped".format(original_len - len(df)))
    #  drop columns
    original_len = len(df.keys())
    if isinstance(col_nan_threshold, float):
        count_na = df.isna().sum()
        count_na = count_na / len(df)
        cols_to_drop = list(count_na[count_na>col_nan_threshold].index)
        df.dropna(axis=1, thresh=len(df)-int(len(df)*col_nan_threshold), inplace=True)
    else:
        cols_to_drop = []
        for c in list(df.keys()):
            if not c in col_nan_threshold:
                print('No threshold was specified for feature '+colored(c, 'red')+'. The column will not be droped'.format(c))
                continue
            nan_ratio = df[c].isna().sum()/len(df)
            threshold = col_nan_threshold[c]# if c in col_nan_threshold else 1
            if nan_ratio > threshold:
                cols_to_drop.append(c)
        df.drop(cols_to_drop, axis=1, inplace=True)
    dropping_comment = "{} columns was dropped".format(len(cols_to_drop))
    if len(cols_to_drop)>0:
        dropping_comment += ': '+ ', '.join(map(str, cols_to_drop))
    print(dropping_comment)
    print('Dropped rows and columns successfully!')
    print('---------------------------------------------------------------------')

    # Build feature dictionary
    print('Buiding feature type dict...')
    print('Valid types are: ', ' / '.join(map(str, valid_types)))
    feature_type = __build_feature_dict__(df, feature_type, categorical_group_size, print_feature_type, valid_types)
    print('Built feature type dict successfully!')
    print('---------------------------------------------------------------------')

    # Fill Nans
    print('Filling nans using the {} method...'.format(method))

    def fill_median(c):
        if feature_type[c] == 'categorical':
            df[c].fillna(df[c].mode().iloc[0], inplace=True)
        else:  # numeric
            df[c].fillna(df[c].median(), inplace=True)

    def fill_best_predictor(c):
        predictor = __find_best_feature_predictor__(df=df, c=c, feature_type=feature_type)
        if predictor is None:
            print('No valid predictoer for {}, fill with {}'.format(c, alternative_method))
            fill_methodes_dict[alternative_method](c)
        else:  # numeric
            print('Fill {} using {}'.format(c, predictor))
            fill_nans_over_feature(c=c, predictor=predictor)

    def fill_min_impact(c):
        # maintain the data distribution of feature c
        if feature_type[c] == 'categorical':
            counter = df.groupby(c).size()
            df[c] = df[c].apply(lambda x: np.random.choice(counter.index, p=list(counter / counter.sum())))
        else:  # numeric
            df['binned_' + c] = pd.qcut(df[c], num_of_bins, labels=False)
            df['binned_' + c] = df['binned_' + c].apply(
                lambda x: x if not pd.isna(x) else np.random.choice(num_of_bins))
            fill_nans_over_feature(c=c, predictor='binned_' + c)
            df.drop('binned_' + c, axis=1, inplace=True)

    def fill_nans_over_feature(c, predictor):
        groups = df.groupby(predictor)
        group_names = df[predictor].unique()
        for g in group_names:
            if feature_type[c] == 'categorical':
                df.loc[df[predictor] == g, c] = df.loc[df[predictor] == g, c].transform(
                    lambda x: x.fillna(x.mode().iloc[0]))
            else:  # numeric
                df.loc[df[predictor] == g, c] = df.loc[df[predictor] == g, c].transform(lambda x: x.fillna(x.median()))

    fill_methodes_dict = {'median': fill_median, 'best_predictor': fill_best_predictor,
                          'minimum_impact': fill_min_impact}
    # fill first the columns with the smallest amount of nans (>0)
    features_nans = df.isna().sum()
    features_nans = list(features_nans[features_nans > 0].sort_values().index)
    print('Features to fill: ', features_nans)
    fill_method = fill_methodes_dict[method]
    for c in features_nans:
        if not feature_type[c] in ['categorical', 'numeric']:
            warning = "Feature {} can't be filled since it is from type {}.".format(c, feature_type[c])
            print(colored(warning, 'red'))
            continue
        fill_method(c)
    print('Filled nans successfully!')
    print('---------------------------------------------------------------------')
    print('Final state of df:')
    df.info()
    return df


def __build_feature_dict__(df, feature_type, categorical_group_size, print_feature_type, valid_types):
    feature_type = {} if feature_type is None else feature_type
    # clean unexisted keys
    keys = list(feature_type.keys())
    for k in keys:
        if not k in df.keys():
            del feature_type[k]
    # fill types for unspecified keys
    for f in df.keys():
        if not f in feature_type.keys():
            feature_type[f] = __find_type__(df[f], f, categorical_group_size)
    if print_feature_type:
        print('Feature type dict:')
        pp.pprint(feature_type)
    return feature_type


def __find_type__(x, f, average_group_size=10, num_of_labels_warning=20):
    n_labels = x.nunique()
    if float(len(x) / n_labels) > average_group_size:
        t = 'categorical'
        if n_labels > num_of_labels_warning:  # many classes
            warning = "Warning: feature {} was classified as categorical, but has more than {} different classes. If this classification is wrong, please specify {}'s type in feature_type".format(
                f, num_of_labels_warning, f)
            print(colored(warning, 'red'))
    elif np.issubdtype(x.dtype, np.number):
        if all(x == x.index) or all(x == x.index+1):
            t = 'id'
            warning = "Warning: feature {} was classified as id".format(f)
            print(colored(warning, 'red'))
        else:
            t = 'numeric'
    else:
        t = 'str'
        if n_labels < x.count():  # hes repetition
            warning = "Warning: feature {} was classified as str, but contains repetitions. If this classification is wrong, please specify {}'s type in feature_type".format(
                f, f)
            print(colored(warning, 'red'))
    return t


def __find_best_feature_predictor__(df, c, feature_type):
    # look for the categorical feature which best predicts column 'c' (minimize the spreadness)
    if feature_type[c] == 'categorical':  # spreadness will be measured via entropy
        def calc_spreadness(x):
            x = x.value_counts(normalize=True, sort=False).to_numpy()
            return -np.multiply(x, np.log(x)).sum()
    else:  # c is a numeric feature. spreadness will be measured via std
        calc_spreadness = lambda x: x.std()
    best_feature = None
    best_spreadness = calc_spreadness(df[c])
    # use categorical features with no nans
    cat_features = df[[f for f, t in feature_type.items() if t == 'categorical']].dropna(axis=1)
    for f in cat_features.keys():
        groups = df.groupby(f)
        # iterate over each group
        values = []
        ratios = []
        group_of_nans = False
        for _, group in groups:
            if group[c].count == 0:
                group_of_nans = True
            ratios.append(len(group))
            values.append(calc_spreadness(group[c]))
        ratios = np.array(ratios) / sum(ratios)
        spreadness = np.average(values, weights=ratios)
        if spreadness < best_spreadness and not group_of_nans:
            best_feature = f
            best_spreadness = spreadness
    return best_feature
