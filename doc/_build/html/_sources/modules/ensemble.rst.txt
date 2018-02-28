Ensemble Methods
======================
Ensemble methods provide multi-label classification compatible ensemble methods,
where a number of estimators (or classifiers) are used to gather a number of
predictions, and then obtain votes by majority vote or averaging. This is
expected to achieve better results, as the diversity of classifiers (optimally)
works as an error correction to the other classifiers.

.. automodule:: skml.ensemble
  :members:

.. autoclass:: EnsembleClassifierChain
   :members:

   .. automethod:: __init__
