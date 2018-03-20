from __future__ import division, print_function, absolute_import

from autotf.model.base_model import Model


class Trial(object):
    """A Trial trains a machine learning model with given hyper-parameters.
    """

    def __init__(self):
        self._model = Model()

    def run(self):
        return

    def stop(self):
        return