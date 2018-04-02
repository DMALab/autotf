from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB


class BaseSelector:

    ALL_LEARNERS = [LogisticRegression(), SVC(), KNeighborsClassifier()]

    def __init__(self,
                 task_type,
                 total_time=None,
                 learners=None,
                 save_directory=None):
        self.task_type = task_type
        self.total_time = total_time
        self.learners = learners
        self.save_directory = save_directory

        self.X = None
        self.y = None

    def select_model(self, X, y, metric=None):
        """
        Find the best model with its hyperparameters from the autotf's model zool

        Parameters
        ----------
        X: array-like or sparse matrix
        y: the target classes
        metric: the eay to evaluate the model

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

    def show_models(self, params, accu):
        """
        Display the models which the selector has found.
        :return:
        """
        pass
