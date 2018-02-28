"""
=================================
Ensemble Binary Relevance Example
=================================

An example of :class:`skml.problem_transformation.BinaryRelevance`
"""
from __future__ import print_function

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

from skml.problem_transformation import BinaryRelevance
from skml.datasets import load_dataset

X, y = load_dataset('yeast')
X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = BinaryRelevance(LogisticRegression())
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


print("hamming loss: ")
print(hamming_loss(y_test, y_pred))

print("accuracy:")
print(accuracy_score(y_test, y_pred))

print("f1 score:")
print("micro")
print(f1_score(y_test, y_pred, average='micro'))
print("macro")
print(f1_score(y_test, y_pred, average='macro'))

print("precision:")
print("micro")
print(precision_score(y_test, y_pred, average='micro'))
print("macro")
print(precision_score(y_test, y_pred, average='macro'))

print("recall:")
print("micro")
print(recall_score(y_test, y_pred, average='micro'))
print("macro")
print(recall_score(y_test, y_pred, average='macro'))
