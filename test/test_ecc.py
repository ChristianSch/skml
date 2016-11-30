from chai import Chai
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import hamming_loss
import scipy.sparse as sp

from skml.ensemble import EnsembleClassifierChain
from skml.datasets import load_dataset

X, y = load_dataset('yeast')


class TestECC(Chai):
    def test_ecc_fit_predict(self):
        ensemble = EnsembleClassifierChain(RandomForestClassifier(),
                                           threshold=.6,
                                           max_features=1.0)
        ensemble.fit(X, y)
        y_pred = ensemble.predict(X)
        hamming_loss(y, y_pred)

    def test_ecc_fit_predict_proba(self):
        ensemble = EnsembleClassifierChain(RandomForestClassifier(),
                                           threshold=.6,
                                           max_features=1.0)
        ensemble.fit(X, y)

        y_pred_proba = ensemble.predict_proba(X)

    def test_ecc_pipeline(self):
        pl = Pipeline([("cc",
                        EnsembleClassifierChain(RandomForestClassifier()))])
        pl.fit(X, y)

    def test_ecc_gridsearch(self):
        ecc = EnsembleClassifierChain(RandomForestClassifier())
        cv = GridSearchCV(ecc, {'estimator__n_estimators': [10, 20]})
        cv.fit(X, y)

    def test_ecc_always_present(self):
        # Test that ecc works with classes that are always present or absent.
        ecc = EnsembleClassifierChain(RandomForestClassifier())
        X_2 = np.array([[2, 3], [4, 0]])
        y_2 = np.array([[1, 1], [1, 0]])
        ecc.fit(X, y)

    def test_ecc_fit_predict_sparse(self):
        # test fit/predict of sparse matrices
        for sparse in [sp.csr_matrix, sp.csc_matrix, sp.coo_matrix,
                       sp.dok_matrix, sp.lil_matrix]:
            clf = EnsembleClassifierChain(RandomForestClassifier())
            clf.fit(X, sparse(y))
            y_pred = clf.predict(X)
            assert_true(sp.issparse(y_pred))
