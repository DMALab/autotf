import numpy as np
import time

import logging
logging.basicConfig(level=logging.INFO)

from tuner.tuner import Tuner
from test_model import TestModel

# Defining the bounds and dimensions of the input space
lower = np.array([0, 2, 1])
upper = np.array([6, 5, 9])

start_time = time.time()
# Start Bayesian optimization to optimize the objective function

results = Tuner(TestModel, lower, upper, num_iterations=20)
print(results["x_opt"])
print(results["f_opt"])
print("it takes: %f seconds" % (time.time() - start_time))
