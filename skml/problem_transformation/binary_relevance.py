import random
import numpy as np
from sklearn.base import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin, ClassifierMixin
from sklearn.utils import validation


class BinaryRelevance(
        BaseEstimator, MetaEstimatorMixin, ClassifierMixin):
    """Implements the binary relevance problem transformation strategy,
    where for each label a distinct binary classifier is trained to
    classify whether the instance should be part of the class or not.
    """
    def __init__(self, estimator):
        """
        Constructs a classifier for each single label, thus having n_labels
        estimators predicting a single label for an instance.

        Parameters
        ----------
        classifier : scikit-learn compatible classifier instance. Will be
                     copied (with all hyperparameters) before use, hence will
                     be left untouched.
        number_of_chains : Number of chains the ensemble shall train
        threshold : Decision threshold to assign a label or not. Has to be
                    between 0 and 1.
        max_features : Number of features to draw from.

        Returns
        -------
        ensemble classifier chain instance
        """
        # get's cloned later on
        self.estimator = estimator
        self.estimators_ = []

    def fit(self, X, y):
        """Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-label targets.
        """

        validation.check_X_y(X, y, multi_output=True)
        y = validation.check_array(y, accept_sparse=True)

        self.estimators_ = [clone(self.estimator).fit(X, y[:, i])
                            for i in xrange(y.shape[1])]

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

    def predict_proba(self, X):
        """
        Predicts the label probabilites for the given instances. Note that
        these probabilities might not be obtainable, depending on the used
        classifiers. They also might have to be calibrated.

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
        return np.array([e.predict_proba(X) for e in self.estimators_]).T
