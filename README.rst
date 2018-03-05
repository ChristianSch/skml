====
skml
====
.. image:: https://travis-ci.org/ChristianSch/skml.svg?branch=master
   :target: https://travis-ci.org/ChristianSch/skml

scikit-learn compatible multi-label classification implementations.

A multi-class classification (MLC) problem is given, if a subset of labels
(picture of equation) shall be predicted for an example.

Currently Supported
-------------------
* Problem Transformations:
    * Binary Relevance
    * Label Powerset
    * Classifier Chains
    * Probabilistic Classifier Chains
* Ensembles:
    * Ensemble Classifier Chain

Installation
------------
For production install via pip:
```
pip install skml
```

For development, clone this repo, change to the directory of skml
and inside of the skml directory run the following:
```
pip install -e .[dev]
python setup.
```


Python Supported
----------------
Due to dependencies we do not check for a working distribution of skml for the
following Python versions:

* 3.2
