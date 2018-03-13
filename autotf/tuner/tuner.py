from __future__ import division, print_function, absolute_import

from autotf.tuner.tune_config import TuneConfig
from autotf.tuner import tune_strategy


class Tuner(object):
    """A Tuner is the entry of automatically tuning hyper-parameters.

    """

    def __init__(self, config):
        self._config = TuneConfig(config)
        self._strategy = tune_strategy.get_strategy(self._config.strategy_name)
        self._trials = []

    def run(self):
        return
