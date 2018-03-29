from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesRegressor

from autotf.selector.base_selector import BaseSelector
from autotf.selector.daub import DAUB


class FastSelector(BaseSelector):

    def __init__(self):
        super().__init__()

    def select_model(self, X, y, total_time,
                     metric=None,
                     learners=None,
                     save_directory=None):
        daub = DAUB(learners, X, y)

        learner, score = daub.fit()

        return learner, score




