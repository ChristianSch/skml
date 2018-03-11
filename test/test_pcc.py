from chai import Chai
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import hamming_loss
import scipy.sparse as sp

from skml.problem_transformation.probabilistic_classifier_chain \
   import ProbabilisticClassifierChain
from skml.datasets import load_dataset

X, y = load_dataset('yeast')
# sample down for quicker testing
y = y[:, :6]


class TestPCC(Chai):
    def test_pcc_fit_predict(self):
        clf = ProbabilisticClassifierChain(LogisticRegression())
        clf.fit(X, y)
        y_pred = clf.predict(X)
        hamming_loss(y, y_pred)

    def test_pcc_pipeline(self):
        pl = Pipeline([("pcc",
                        ProbabilisticClassifierChain(
                            RandomForestClassifier()))])
        pl.fit(X, y)

    def test_pcc_gridsearch(self):
        clf = ProbabilisticClassifierChain(RandomForestClassifier())
        cv = GridSearchCV(clf,
                          {'estimator__n_estimators': [10, 20]},
                          n_jobs=-1)
        cv.fit(X, y)

    def test_pcc_always_present(self):
        # Test that cc works with classes that are always present or absent.
        clf = ProbabilisticClassifierChain(LogisticRegression())
        X_2 = np.array([[2, 3], [4, 0]])
        y_2 = np.array([[1, 1], [1, 0]])
        clf.fit(X, y)

    def test_pcc_predict_multi_instances(self):
        clf = ProbabilisticClassifierChain(LogisticRegression())
        clf.fit(X, y)

        y_pred = clf.predict(X)
        assert_true(y_pred.shape == y.shape)

    def test_pcc_fit_predict_sparse(self):
        # test fit/predict of sparse matrices
        for sparse in [sp.csr_matrix, sp.csc_matrix, sp.coo_matrix,
                       sp.dok_matrix, sp.lil_matrix]:
            clf = ProbabilisticClassifierChain(LogisticRegression())
            clf.fit(X, sparse(y))
            y_pred = clf.predict(X)
            assert_true(sp.issparse(y_pred))
