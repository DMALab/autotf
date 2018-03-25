import numpy as np
import time

from tuner.fmin import bayesian_optimization


import logging
logging.basicConfig(level=logging.INFO)


# The optimization function that we want to optimize.
# It gets a numpy array with shape (1,D) where D is the number of input dimensions
def objective_function(x):
    # y = np.sin(3 * x[0]) * 4 * (x[0] - 1) * (x[0] + 2)
    res = 0
    for i in range(100000000):
        res += i

    y = sum(x * x)
    return y


# Defining the bounds and dimensions of the input space
lower = np.array([0, 2, 1])
upper = np.array([6, 5, 9])
start_time = time.time()

# Start Bayesian optimization to optimize the objective function
results = bayesian_optimization(objective_function, lower, upper, num_iterations=10, n_workers=4, parallel_type="async")

print(results["x_opt"])
print(results["f_opt"])
print("it takes: %f seconds" % (time.time() - start_time))
