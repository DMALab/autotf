from autotf.selector.base_selector import BaseSelector


class AccurateSelector(BaseSelector):

    def __init__(self):
        super().__init__()

    def select_model(self, X, y,
                     total_time,
                     metric=None,
                     save_directory=None):
        """
        Find the best model with its hyperparameters from the autotf's model zool
        """

        return "the best model object"


