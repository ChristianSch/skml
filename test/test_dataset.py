from chai import Chai
from scipy import sparse

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
        y = sparse.rand((200, 20))
        sample10 = sample_down_label_space(y, 10)

        assert sample10.shape[1] == 10

    def test_load_enron(self):
        X, y = load_dataset('enron', 'undivided')
