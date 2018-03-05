from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
   INSTALL_DEV_REQUIRES = [l.strip() for l in f.readlines() if l]

INSTALL_REQUIRES = [
    'numpy',
    'sklearn'
]


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)

setup(name='skml',
      version='0.2.1',
      description='scikit-learn compatibel multi-label classification',
      author='Christian Schulze',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      extras_require={
        'dev': [r for r in INSTALL_DEV_REQUIRES if not r in INSTALL_REQUIRES]
      })
