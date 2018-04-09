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
