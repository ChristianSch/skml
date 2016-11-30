from chai import Chai
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
import numpy as np
import scipy.sparse as sp

from skml.problem_transformation import BinaryRelevance
from skml.datasets import load_dataset

X, y = load_dataset('yeast')


class TestBR(Chai):
    def test_br_fit_predict(self):
        clf = BinaryRelevance(RandomForestClassifier())
        clf.fit(X, y)
        y_pred = clf.predict(X)
        hamming_loss(y, y_pred)

    def test_br_pipeline(self):
        pl = Pipeline([("br", BinaryRelevance(RandomForestClassifier()))])
        pl.fit(X, y)

    def test_br_fit_predict_proba(self):
        clf = BinaryRelevance(RandomForestClassifier())
        clf.fit(X, y)
        y_pred_proba = clf.predict_proba(X)

    def test_br_gridsearch(self):
        br = BinaryRelevance(RandomForestClassifier())
        cv = GridSearchCV(br, {'estimator__n_estimators': [10, 20]})
        cv.fit(X, y)

    def test_br_always_present(self):
        # Test that br works with classes that are always present or absent.
        clf = BinaryRelevance(RandomForestClassifier())
        X_2 = np.array([[2, 3], [4, 0]])
        y_2 = np.array([[1, 1], [1, 0]])
        clf.fit(X, y)

    def test_br_fit_predict_sparse(self):
        # test fit/predict of sparse matrices
        for sparse in [sp.csr_matrix, sp.csc_matrix, sp.coo_matrix,
                       sp.dok_matrix, sp.lil_matrix]:
            clf = BinaryRelevance(RandomForestClassifier())
            clf.fit(X, sparse(y))
            y_pred = clf.predict(X)
            assert_true(sp.issparse(y_pred))
