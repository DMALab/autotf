from autotf.selector.base_selector import BaseSelector
from autotf.selector.daub import DAUB


class FastSelector(BaseSelector):

    def select_model(self, X, y, metric=None):
        if self.learners is None:
            learners = BaseSelector.ALL_LEARNERS
        else:
            learners = self.learners
        daub = DAUB(learners, X, y)

        learner, score = daub.fit()

        return learner, score




