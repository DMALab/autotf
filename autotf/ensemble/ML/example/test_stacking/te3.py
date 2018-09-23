from ensemble.superknn3 import Superknn
from ensemble import Stacking
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
import time
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

# Prepare data
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=0)

print('Train shape:', X_train.shape)
print('Test shape: ', X_test.shape)


models = [ExtraTreesClassifier(n_jobs=1, random_state=0),
    RandomForestClassifier(n_jobs=1, random_state=0),

    XGBClassifier(n_jobs=1, random_state=0)
]
ensemble = Stacking(X_train, y_train, X_test, bagged_pred=False,
                               regression=False,
                               n_folds=5, needs_proba=True,
                               shuffle=True, random_state=0, verbose=0)

ensemble.add(models)
print(ensemble.next_test[0:5])

fitted_models = []
for model in models:
    model.fit(X_train,y_train)
    fitted_models.append(model)

y_pred2 = []

for vector in X_test:
    y_pred3 = []
    for model in fitted_models:
        y_pred3.append(model.predict_proba(vector.reshape(1, -1)))
    y_pred2.append(y_pred3)

print(y_pred2[0:5])
