from chai import Chai
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
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

    def test_br_gridsearch(self):
        br = BinaryRelevance(RandomForestClassifier())
        cv = GridSearchCV(br,
                          {'estimator__n_estimators': [10, 20]},
                          n_jobs=-1)
        cv.fit(X, y)

    def test_br_always_present(self):
        # Test that br works with classes that are always present or absent.
        clf = BinaryRelevance(RandomForestClassifier())
        X_2 = np.array([[2, 3], [4, 0]])
        y_2 = np.array([[1, 1], [1, 0]])
        clf.fit(X, y)

    def test_br_predict_multi_instances(self):
        clf = BinaryRelevance(RandomForestClassifier())
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert_true(y_pred.shape[0] == y.shape[0])

    def test_br_fit_predict_sparse(self):
        # test fit/predict of sparse matrices
        for sparse in [sp.csr_matrix, sp.csc_matrix, sp.coo_matrix,
                       sp.dok_matrix, sp.lil_matrix]:
            clf = BinaryRelevance(RandomForestClassifier())
            clf.fit(X, sparse(y))
            y_pred = clf.predict(X)
            assert_true(sp.issparse(y_pred))
