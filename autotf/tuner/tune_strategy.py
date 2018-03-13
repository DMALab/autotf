from __future__ import division, print_function, absolute_import

from autotf.tuner.acquisition_functions.base_acquisition import BaseAcquisitionFunction


def get_strategy(name=None):
    if not name:
        return TuneStrategy()


class TuneStrategy(object):
    """A TuneStrategy controls how to create and schedule trials.

    """

    def __init__(self):
        self._acquistion = BaseAcquisitionFunction()
