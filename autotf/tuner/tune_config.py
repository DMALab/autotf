from __future__ import division, print_function, absolute_import


class TuneConfig(object):
    """A TuneConfig defines the setting of a Tuner, e.g., data path, ML model, tune strategy, hyper-parameter settings.
    """

    def __init__(self, config):
        self._config_str = config
        self.data_path = ""
        self.model_name = ""
        self.strategy_name = ""

    def parse(self):
        return