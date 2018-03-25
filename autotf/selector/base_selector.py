class BaseSelector:

    def __init__(self):
        pass

    def select_model(self, X, y,
                     total_time,
                     metric=None,
                     save_directory=None):
        """
        Find the best model with its hyperparameters from the autotf's model zool

        Parameters
        ----------
        X: array-like or sparse matrix
        y: the target classes
        total_time: the training time
        metric: the eay to evaluate the model
        save_directory: the path to save the

        """

        return "the best model object"

    def fit(self, X, y):
        """
        Train with the best model
        """
        pass

    def predict(self, X):
        """
        Predict with the best model
        """

    def best_score(self):
        """
        Return the best score such as cross-validation accuracy or f1.
        :return:
        """
        return "best score"

    def show_models(self):
        """
        Display the models which the selector has found.
        :return:
        """
        pass