from scipy.stats import rankdata
import numpy as np

def feature_normalizer(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    dist_data = np.sort(np.random.normal(loc=mean, scale=std, size=len(x)))
    sort_idx = rankdata(x, method='min') - 1
    return dist_data[sort_idx]
