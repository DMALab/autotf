from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesRegressor

from autotf.selector.base_selector import BaseSelector


models = [LogisticRegression(), DecisionTreeClassifier(),
          GradientBoostingClassifier(), ExtraTreesRegressor()]


class FastSelector(BaseSelector):

    def __init__(self):
        super().__init__()

    def select_model(self, X, y):
        return daub(learners=models, X_train=X, y_train=y)


def daub(learners, X_train, y_train, ratio, b):
    """
    DAUB(Data Allocation using Upper Bounds) is a method for searching the best algorithm within
    limited time.
    Reference:
    Sabharwal A, Samulowitz H, Tesauro G (2016) Selecting near-optimal learners via incremental data allocation.
    Proc. AAAI 2016.
    """
    pass
