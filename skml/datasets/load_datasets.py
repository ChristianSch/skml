import numpy as np
from sklearn.datasets import fetch_mldata


def load_dataset(name):
    """
    Loads a multi-label classification dataset.

    Parameters
    ----------
    name : string
        Name of the dataset. Currently only 'yeast' is available.
    """
    if name == 'yeast':
        data = fetch_mldata('yeast')
        X = data.data
        y = data.target.toarray().astype(np.int).T

        return (X, y)
    else:
        raise Exception("No such dataset")
