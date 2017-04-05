"""
=================================
Classifier Chain Example
=================================

An example of :class:`skml.problem_transformation.ClassifierChain`
"""

from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np


from skml.problem_transformation import ClassifierChain
from skml.datasets import load_dataset

X, y = load_dataset('yeast')
cc = ClassifierChain(RandomForestClassifier(),
                     threshold=.5)
cc.fit(X, np.array(y))
y_pred = cc.predict(X)
y_pred_proba = cc.predict_proba(X)

print("real: ", y.shape)
print("y_pred: ", y_pred.shape)

print("hamming loss: ")
print(hamming_loss(y, y_pred))

print("accuracy:")
print(accuracy_score(y, y_pred))

print("f1 score:")
print("micro")
print(f1_score(y, y_pred, average='micro'))
print("macro")
print(f1_score(y, y_pred, average='macro'))

print("precision:")
print("micro")
print(precision_score(y, y_pred, average='micro'))
print("macro")
print(precision_score(y, y_pred, average='macro'))

print("recall:")
print("micro")
print(recall_score(y, y_pred, average='micro'))
print("macro")
print(recall_score(y, y_pred, average='macro'))
