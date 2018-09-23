from ensemble.superknn import Superknn
import numpy as np
from sklearn.ensemble.forest import ExtraTreesClassifier as ExtremeRandomizedTrees
from sklearn.neighbors import KNeighborsClassifier as kNearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from ensemble import Stacking
from ensemble.Deep_super_learner import DeepSuperLearner
import time
import warnings
warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

# Prepare data
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=0)
# '''实验1：不对model调参'''
# '''1.just use the XGB'''
# model = XGBClassifier(n_jobs=-1)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('Final prediction score with optimal(use XGB): [%.8f]' % accuracy_score(y_test, y_pred))
#
# '''2.just use the Extra'''
# model = ExtraTreesClassifier(n_jobs=-1)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('Final prediction score with optimal(use Extra): [%.8f]' % accuracy_score(y_test, y_pred))
#
# '''3.just use the GBDT'''
# model = GradientBoostingClassifier()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('Final prediction score with optimal(use GBDT): [%.8f]' % accuracy_score(y_test, y_pred))
#
# '''4.just use the RF'''
# model = RandomForestClassifier(n_jobs=-1)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('Final prediction score with optimal(use RF): [%.8f]' % accuracy_score(y_test, y_pred))
#
'''5.use Deep super learner'''
# Make train/test split
# As usual in machine learning task we have X_train, y_train, and X_test
ERT_learner = ExtremeRandomizedTrees(n_estimators=200, max_depth=None, max_features=1)
kNN_learner = kNearestNeighbors(n_neighbors=11)
LR_learner = LogisticRegression()
RFC_learner = RandomForestClassifier(n_estimators=200, max_depth=None)
XGB_learner = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=1.)
models= [ERT_learner,  kNN_learner,LR_learner,
        RFC_learner, XGB_learner]
model = DeepSuperLearner(models, shuffle=True, k=5, weight_on_model=False, max_iter=20, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Final prediction score with optimal(use Deepsuper): [%.8f]' % accuracy_score(y_test, y_pred))

# '''6.use Stacking'''
# # Make train/test split
# # As usual in machine learning task we have X_train, y_train, and X_test
# models = [ExtraTreesClassifier(random_state=0),
#     RandomForestClassifier(random_state=0),
#     GradientBoostingClassifier(random_state=0),
#     XGBClassifier(random_state=0)
# ]
# meta_model = LogisticRegression()
# model = Stacking(X_train, y_train, X_test, regression=False, bagged_pred=True,
#                     needs_proba=True, save_dir=None, metric=accuracy_score,
#                     n_folds=5, stratified=True, shuffle=True,
#                     random_state=0, verbose=0)
# model.add(models)
# y_pred = model.add_meta(meta_model)
# print('Final prediction score with optimal(use Stacking): [%.8f]' % accuracy_score(y_test, y_pred))
#
#
