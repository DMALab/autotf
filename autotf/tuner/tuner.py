from __future__ import division, print_function, absolute_import

from autotf.tuner.tune_config import TuneConfig
from autotf.tuner import tune_strategy


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

    def __init__(self, config):
        self._config = TuneConfig(config)
        self._strategy = tune_strategy.get_strategy(self._config.strategy_name)
        self._trials = []

    def run(self):
        return
