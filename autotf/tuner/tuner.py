from __future__ import division, print_function, absolute_import

import time
import numpy as np

from tuner.fmin import bayesian_optimization
from tuner.tune_config import TuneConfig

import logging
logging.basicConfig(level=logging.INFO)


class Tuner(object):
    """A Tuner is the entry of automatically tuning hyper-parameters.
    user can define: 1. n_workers (default to n_cpu_kernel)
                     2. parallel_strategy (async and sync)
                     3. aggresive_degree
                     4. acquisition_function
                     5. maximizer
                     6. model_type
                     7. n_iteration
                     8. random_seed
                     9. n_init
                     10. dataset
                     11. hyper-parameter definition

    ---
    tips:
      1. support user defined function
      2. support autotf machine learning model

    """

    def __init__(self, obj_func, lower, upper, num_iter=10, num_worker=1, config=None, maximizer="random",
                 acquisition_func="log_ei",
                 model_type="gp_mcmc", n_init=3, rng=None, output_path=None,
                 parallel_type="sync"):

        if config is not None: self._config = TuneConfig(config)
        # self._strategy = tune_strategy.get_strategy(self._config.strategy_name)
        # self._trials = []

        self.maximizer = maximizer
        self.objective_function = obj_func
        self.acquisition_func = acquisition_func
        self.model_type = model_type
        self.parallel_type = parallel_type
        self.n_init = n_init
        self.rng = rng
        self.lower = lower
        self.upper = upper
        self.num_iterations = num_iter
        self.num_worker = num_worker
        self.output_path = output_path

    def run(self):
        start_time = time.time()
        results = bayesian_optimization(
            self.objective_function, self.lower,
            self.upper, num_iterations=self.num_iterations,
            n_workers=self.num_worker, maximizer=self.maximizer,
            acquisition_func=self.acquisition_func, model_type=self.model_type,
            n_init=self.n_init, rng=self.rng, output_path=self.output_path,
            parallel_type=self.parallel_type
        )

        print(results["x_opt"])
        print(results["f_opt"])
        print("it takes: %f seconds" % (time.time() - start_time))

        return results
