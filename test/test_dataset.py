from chai import Chai
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from skml.problem_transformation.probabilistic_classifier_chain \
   import ProbabilisticClassifierChain

from skml.datasets import load_dataset, sample_down_label_space


class TestDataset(Chai):
    def test_load_yeast(self):
        X, y = load_dataset('yeast')

    def test_sample_down_label_space(self):
        _, y = load_dataset('yeast')

        sample10 = sample_down_label_space(y, 10)
        assert sample10.shape[1] == 10

        sample5 = sample_down_label_space(y, 5)
        assert sample5.shape[1] == 5

        self.assert_raises(ValueError, sample_down_label_space, y, 20)

    def test_sparse_sample_down_label_space(self):
        y = sparse.rand(200, 20, format='csc')
        sample10 = sample_down_label_space(y, 10)

        assert sample10.shape[1] == 10

    def test_sparse_sample_down_label_space_classification(self):
        clf = ProbabilisticClassifierChain(LogisticRegression())
        # LogisticRegression needs dense
        X = sparse.random(100, 15, format='csc').toarray()
        _y = sparse.random(100, 20, format='csc')
        y = sample_down_label_space(_y, 10)
        y = y > 0.1
        y = y.toarray().astype(int)

        clf.fit(X, y)
        y_pred = clf.predict(X)

        assert y_pred.shape == y.shape


    def test_load_enron(self):
        X, y = load_dataset('enron', 'undivided')
