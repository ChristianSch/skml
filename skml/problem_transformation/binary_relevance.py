import random
import numpy as np
from sklearn.base import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin, ClassifierMixin
from sklearn.utils import validation


class BinaryRelevance(
        BaseEstimator, MetaEstimatorMixin, ClassifierMixin):
    """Implements the binary relevance problem transformation strategy,
    where for each label a distinct binary classifier is trained to
    classify whether label should be assigned to an instance or not.
    """
    def __init__(self, estimator):
        """
        Constructs a classifier for each single label, thus having n_labels
        estimators predicting a single label for an instance.

        Parameters
        ----------
        estimator :
            scikit-learn compatible classifier instance. Will be
            copied (with all hyperparameters) before use, hence
            original will be left untouched.

        Returns
        -------
        Binary relevance problem transformed MLC classifier
        """
        # get's cloned later on
        self.estimator = estimator
        self.estimators_ = []

    def fit(self, X, y):
        """Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-label targets
        """

        validation.check_X_y(X, y, multi_output=True)
        y = validation.check_array(y, accept_sparse=True)

        self.estimators_ = [clone(self.estimator).fit(X, y[:, i])
                            for i in range(y.shape[1])]

    def predict(self, X):
        """
        Predicts the labels for the given instances.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        Returns
        -------
        array-like, shape = [n_samples, n_labels]
            Estimated labels
        """
        validation.check_is_fitted(self, 'estimators_')
        return np.array([e.predict(X) for e in self.estimators_]).T
