from chai import Chai

from skml.datasets import load_dataset


class TestDataset(Chai):
    def load_yeast(self):
        X, y = load_dataset('yeast')
