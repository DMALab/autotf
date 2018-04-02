import numpy as np


class TestModel(object):
    def __init__(self):
        pass

    @staticmethod
    def train(x):
        # y = np.sin(3 * x[0]) * 4 * (x[0] - 1) * (x[0] + 2)
        res = 0
        for i in range(100000000):
            res += i

        y = sum(x * x)
        return y
