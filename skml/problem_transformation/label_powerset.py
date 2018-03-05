import random
import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, ClassifierMixin
from sklearn.utils import validation
from sklearn.base import clone


class LabelPowerset(
        BaseEstimator, MetaEstimatorMixin, ClassifierMixin):
    """Implements the label powerset problem transformation strategy,
    where for each possible subset of the label powerset a classifier is
    trained, which classifies whether or not the label subset should be
    assigned to the given instance or not.
    """
    def __init__(self, estimator):
        """
        Creates label powersets for the given labels. Each powerset is
        transformed into a single label, resulting in a multi-class
        classification problem. For :math:`|\mathcal{L}|` labels, this meta
        estimator generates :math:`|2^\mathcal{L}|` labels.

        Note, that rare label combinations result in a small number of samples
        for a given label after the transformation. This might result in rather
        bad learning effects for the combination given by the label powerset.
        Also, the computational complexity is exponential, hence for many
        labels the label powerset increases notably, and might result in
        computational problems.

        Parameters
        ----------
        estimator : scikit-learn compatible classifier instance
            Will be copied (with all hyperparameters) before use,
            hence original will be left untouched.

        Returns
        -------
        label powerset transformed multi-class meta estimator
        """
        # get's cloned later on
        self.estimator = estimator
        self.powerset_lookup = {}
        self.reverse_lookup = {}

    def fit(self, X, y):
        """Fit underlying estimators.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data
        y : (sparse) array-like, shape = [n_samples, ], [n_samples, n_classes]
            Multi-label targets
        """
        # should the powersets that are not seen in the training data also be
        # used as a class?

        # we can actually use simple integers to denote the classes, as the
        # classifiers capable of multi-class classification will transform them
        # on their own.
        self.estimator = clone(self.estimator)

        validation.check_X_y(X, y, multi_output=True)
        y = validation.check_array(y, accept_sparse=True)

        max_label = 0
        y_transformed = []

        for i in range(y.shape[0]):
            lstr = ",".join([str(x) for x in y[i]])

            if lstr not in self.powerset_lookup.keys():
                self.reverse_lookup[max_label] = y[i]
                self.powerset_lookup[lstr] = max_label
                max_label += 1

            y_transformed.append(self.powerset_lookup[lstr])
        y_transformed = np.array(y_transformed)

        self.estimator.fit(X, y_transformed)

    def predict(self, X):
        """
        Predicts the labels for the given instances.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data

        Returns
        -------
        array-like, shape = [n_samples, n_labels]
            Estimated labels
        """
        validation.check_is_fitted(self, 'estimator')
        # enforce sparce np mat
        y_hat = self.estimator.predict(X)

        return np.array([self.reverse_lookup[row] for row in y_hat])
