from chai import Chai
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_greater
from sklearn.metrics import hamming_loss
import scipy.sparse as sp

from skml.problem_transformation import ClassifierChain
from skml.datasets import load_dataset

X, y = load_dataset('yeast')


class TestCC(Chai):
    def test_cc_fit_predict(self):
        chain = ClassifierChain(RandomForestClassifier())
        chain.fit(X, y)
        y_pred = chain.predict(X)
        hamming_loss(y, y_pred)

    def test_cc_pipeline(self):
        pl = Pipeline([("cc", ClassifierChain(RandomForestClassifier()))])
        pl.fit(X, y)

    def test_cc_gridsearch(self):
        cc = ClassifierChain(RandomForestClassifier())
        cv = GridSearchCV(cc, {'estimator__n_estimators': [10, 20]})
        cv.fit(X, y)

    def test_cc_always_present(self):
        # Test that cc works with classes that are always present or absent.
        cc = ClassifierChain(RandomForestClassifier())
        X_2 = np.array([[2, 3], [4, 0]])
        y_2 = np.array([[1, 1], [1, 0]])
        cc.fit(X, y)

    def test_cc_predict_multi_instances(self):
        clf = ClassifierChain(RandomForestClassifier())
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert y_pred.shape[0] == y.shape[0]

    def test_cc_fit_predict_sparse(self):
        # test fit/predict of sparse matrices
        # test fit/predict of sparse matrices
        for sparse in [sp.csr_matrix, sp.csc_matrix, sp.coo_matrix,
                       sp.dok_matrix, sp.lil_matrix]:
            clf = ClassifierChain(RandomForestClassifier())
            clf.fit(X, sparse(y))
            y_pred = clf.predict(X)
            assert_true(sp.issparse(y_pred))
