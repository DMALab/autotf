from sklearn.ensemble.forest import ExtraTreesClassifier as ExtremeRandomizedTrees
from sklearn.neighbors import KNeighborsClassifier as kNearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier

from xgboost.sklearn import XGBClassifier
import numpy as np
from sklearn import datasets
from ensemble.Deep_super_learner import DeepSuperLearner
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    ERT_learner = ExtremeRandomizedTrees(n_estimators=200, max_depth=None, max_features=1)
    kNN_learner = kNearestNeighbors(n_neighbors=11)
    LR_learner = LogisticRegression()
    RFC_learner = RandomForestClassifier(n_estimators=200, max_depth=None)
    XGB_learner = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=1.)
    models = [ERT_learner, kNN_learner,
                     LR_learner,
                     RFC_learner, XGB_learner]

    np.random.seed(100)
    X, y = datasets.make_classification(n_samples=1000, n_features=12,
                                        n_informative=2, n_redundant=6)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = DeepSuperLearner(models, shuffle=True, k=5, max_iter=20)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)