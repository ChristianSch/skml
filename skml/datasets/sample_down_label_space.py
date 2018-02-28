import numpy as np
from operator import itemgetter


def sample_down_label_space(y, k, method='most-frequent'):
    """
    Samples down label space, such that the returned label
    space retains order of the original labels, but
    removes labels which do not meet certain criteria
    (see `method`).

    Parameters
    ----------
    y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
        Multi-label targets
    k : number
        Number of returned labels, has to be smaller than the number of
        distinct labels in `y`
    method : string, default = 'most-frequent'
        Method to sample the label space down. Currently supported
        is only by top k most frequent labels.
    """
    if k > y.shape[1]:
        raise ValueError('Cannot sample more labels than given')

    if method == 'most-frequent':
        # create mapping of frequencies per column (label)
        freqs = list(map(lambda x: (x[0], x[1]), enumerate(np.sum(y, axis=0))))
        freqs.sort(key=itemgetter(1), reverse=True)
        # select top k labels, restore original order
        # if we wouldn't restore the original order, the labels would
        # be ordered not by original column, but by "most frequent occuring"
        # and could render classifications probably unusable
        sampled_indices = sorted(list(map(lambda x: x[0], freqs[:k])))

        return np.take(y, sampled_indices, axis=1)

    else:
        raise ValueError('No such sample method {0}'.format(method))
