import numpy as np
from sklearn.base import clone
from sklearn.utils import validation
from .classifier_chain import ClassifierChain


class ProbabilisticClassifierChain(ClassifierChain):
    """
    Learns a chain of classifiers. If the input data is given as
    :math:`\mathbf{x}\in\mathcal{X}`, and the target data as a set
    :math:`\lambda =\{\lambda_1,\cdots,\lambda_m\}`, then
    :math:`|\mathcal{L}|`
    classifiers are learned as follows:

    .. math::

        f_i: \mathcal{X} \\times \{0,1\}^{i-1} \\rightarrow [0,1]

        (\mathbf{x},y_1,\cdots,y_{i-1})\mapsto
        \mathbf{P}(y_i=1|\mathbf{x},y_1,\cdots,y_{i=1})

    So each classifier :math:`f_i` is trained on an augmented input
    space :math:`\mathcal{X} \\times\{0,1\}^{i-1}`. NB: it is implied
    that the classifier predicts the probability of :math:`y_i=1`,
    so it has to support `predict_proba(X)`.

    The inference of the most probable label set is then determined
    via:

    .. math::

        \mathbf{h}^* = \\arg \max_{y\in\mathcal{Y}}
        \mathbf{P}_\mathbf{x}(\mathbf{y})

    where :math:`\mathbf{P}_\mathbf{x}(\mathbf{y})` is given as:

    .. math::

        P_\mathbf{x}(y) = P_\mathbf{x}(y_1)\cdot
        \prod^m_{i=2}P_\mathbf{x}(y_i|y_1,\cdots,y_{i-1})

        = P(\mathbf{y}|\mathbf{x}) = P(y_1|\mathbf{x})\cdot
        \prod^m_{i=2}P(y_i|\mathbf{x},y_1,\cdots,y_{i-1})

        ⁼ P(\mathbf{y}|\mathbf{x}) = f_1(\mathbf{x})

    For each label combination in :math:`2^{|\mathcal{L}|}`
    a posterior probability estimate has to be calculated,
    so the original paper [3] notes feasibility for settings with
    a label space bounded by :math:`|\mathcal{L}| \leq 15`.
    """
    def __init__(self, estimator):
        """Probabilistic Classifer Chain multi-label strategy

        Builds a new probabilistic classifier chain using the given
        classifier, which is copied :math:`|\mathcal{L}|` times
        (L is the set of labels).

        Parameters
        ----------
        estimator : scikit-learn compatible classifier instance
            The classifier used to build a chain of classifiers.
            Will be copied, hence the original will be left untouched.
        """
        super().__init__(estimator)

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

        Y = []
        N_instances = X.shape[0]

        for n in range(N_instances):
            x = X[n].reshape(1, -1)
            y_out = None
            p_max = 0

            for b in range(2 ** self.L):
                p = np.zeros((1, self.L))
                y = np.array(list(map(int, np.binary_repr(b, width=self.L))))

                for i, c in enumerate(self.estimators_):
                    """
                    NB: "predict_proba" returns two probabilities [p_0, p_1]:
                        * p_0: the probability that y_i = 0
                        * p_1: the probability that y_i = 1

                        So if y_i = 1, we take p_1, if y_i = 0, we take
                        p_0 as the probability estimate. As we're
                        interested in the probability given a label vector,
                        we have to select the probability of the value
                        of the respective y_i for all i = 1..|L|.
                    """
                    if i == 0:
                        p[0, i] = c.predict_proba(x)[0][y[i]]
                    else:
                        stacked = np.hstack((x, y[:i].reshape(1, -1))) \
                            .reshape(1, -1)
                        p[0, i] = c.predict_proba(stacked)[0][y[i]]

                pp = np.prod(p)

                if pp > p_max:
                    y_out = y
                    p_max = pp

            Y.append(y_out)

        return np.array(Y)

    def predict_proba(self, X):
        """
        Predicts the label probabilities for the given instances.

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

        Y = []
        N_instances = X.shape[0]

        for n in range(N_instances):
            x = X[n].reshape(1, -1)
            y_out = None
            p_max = 0

            for b in range(2 ** self.L):
                p = np.zeros((1, self.L))
                y = np.array(list(map(int, np.binary_repr(b, width=self.L))))

                for i, c in enumerate(self.estimators_):
                    if i == 0:
                        p[0, i] = c.predict_proba(x)[0][y[i]]
                    else:
                        stacked = np.hstack((x, y[:i].reshape(1, -1))) \
                            .reshape(1, -1)
                        p[0, i] = c.predict_proba(stacked)[0][y[i]]

                pp = np.prod(p)

                if pp > p_max:
                    y_out = p
                    p_max = pp

            Y.append(y_out)

        return np.array(Y)
