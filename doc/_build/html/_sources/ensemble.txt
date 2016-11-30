.. _ensemble:

================
Ensemble Methods
================

.. currentmodule:: skml.ensemble

The :mod:`skml.ensemble` module implements ensembles to be used for multi-label
classification.


Ensemble Classifier Chains
==========================
Ensemble of classifier chains (ECC) trains an ensemble of bagged
classifier chains. Each chain is trained on a randomly sampled subset
of the training data (with replacement, also known as bagging).

.. topic:: References:

    .. [1] "Classifier chains for multi-label classification",
       Read, J., Pfahringer, B., Holmes, G. & Frank, E. (2009).
       In Proceedings of European conference on Machine Learning and
       Knowledge Discovery in Databases 2009 (ECML PKDD 2009), Part II,
       LNAI 5782(pp. 254-269). Berlin: Springer.
