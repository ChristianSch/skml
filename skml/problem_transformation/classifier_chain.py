from sklearn.base import clone
from sklearn.base import BaseEstimator, MetaEstimatorMixin, ClassifierMixin
from sklearn.utils import validation
import numpy as np


class ClassifierChain(BaseEstimator, MetaEstimatorMixin, ClassifierMixin):
    """
    This classifier constructs a chain of classifiers for multi-label
    classification (MLC).
    For each label (which ought to be given as a binary label indicator vector,
    where 0 stands for "instance does not have label, 1 otherwise") a
    classifier is trained. The classifier predicts one label, and one only
    (called binary relevance). The first classifier predicts the first label of
    the label vector (which outputs 0 or 1), whereas the second predicts the
    second label, but the first label is appended to the feature vector (X[i]).
    The n-th classifier predicts the n-th label given the feature vector,
    where the (n-1)-th labels are appended to.
    """
    def __init__(self, estimator, threshold=.5):
        """Classifer Chain multi-label strategy

        Builds a new classifier chain using the given classifier, which is
        copied :math:`|\mathcal{L}|` times (L is the set of labels).

        Parameters
        ----------
        classifier : The classifier used to build a chain of classifiers.
        """
        self.estimator = estimator
        self.estimators_ = []
        self.threshold = threshold

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

        for i in range(y.shape[1]):
            c = clone(self.estimator)

            # at this point, all classifiers in the chain from the nodes before
            # this one are fitted to the training data, including any
            # subsequently predicted and appended labels.
            if i == 0:
                c.fit(X, y[:, 0])
                y_pred = (c.predict(X)).reshape(-1, 1)
            else:
                # the classifiers that aren't the first classifiers in the
                # chain use a transformed version of the features, where
                # the previously predicted labels are appended.
                stacked = np.hstack((X, y[:, :i]))
                c.fit(stacked, y[:, i])

            self.estimators_.append(c)

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

        for i, c in enumerate(self.estimators_):
            if i == 0:
                y_pred = (c.predict(X)).reshape(-1, 1)
            else:
                stacked = np.hstack((X, y_pred))
                new_y = c.predict(stacked)
                y_pred = np.hstack((y_pred, new_y.reshape(-1, 1)))

        return y_pred
