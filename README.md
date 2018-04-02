# AutoTF: Automatic Machine Learning Toolkit for TensorFlow

![License](http://img.shields.io/badge/license-BSD3-blue.svg)
![Version](https://img.shields.io/badge/version-0.0.1-green.svg)

## AutoTF
For hyper-parameter tuning, this project implements a flexible, distributed and parallel framework for robust Bayesian Optimization, and the relative module `Tuner` is based on the project [RoBo](https://github.com/automl/RoBO) from AutoML.

## Programming style

- [Google Python Style](http://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/contents/)

## Referenced project
- [Ray](https://github.com/ray-project/ray)
- [HyperOpt](https://github.com/hyperopt)
- [AutoML](https://github.com/automl)
- [Spearmint](https://github.com/JasperSnoek/spearmint)

## Overview
```python
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

from tuner.tuner import Tuner
from test_model import TestModel

# Defining the bounds and dimensions of the input space
lower = np.array([0, 2, 1])
upper = np.array([6, 5, 9])

# Start Bayesian optimization to optimize the objective function

tuners = Tuner(TestModel.train, lower, upper, num_iter=10, num_worker=4)
results = tuners.run()

```

## Installation

`AutoTF` is based on project RoBo, and uses the Gaussian processes library `george` and the random forests library `pyrfr`. In order to use this library make sure the libeigen and swig are installed:
```
sudo apt-get install libeigen3-dev swig gfortran
```

Before you install `AutoTF` you have to install the required dependencies. We use a for loop because we want to preserve the installation order of the list of dependencies in the requirments file.
```
for req in $(cat requirements.txt); do pip install $req; done
```

This will install the basis requirements that you need to run `AutoTF`’s core functionality. If you want to make use of the full functionality (for instance Bohamiann, Fabolas, …) you can install all necessary dependencies by:
```
for req in $(cat all_requirements.txt); do pip install $req; done
```
Note: This may take a while to install all dependencies.
