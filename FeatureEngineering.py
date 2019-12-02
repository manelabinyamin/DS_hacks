from scipy.stats import rankdata
import numpy as np

def feature_normalizer(x):
    dist_data = np.sort(np.random.normal(loc=np.mean(x), scale=np.std(x), size=len(x)))
    sort_idx = rankdata(x, method='min') - 1
    return dist_data[sort_idx]
