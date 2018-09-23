from ensemble.superknn4 import Superknn
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import time
# Prepare data
boston = load_boston()
X, y = boston.data, boston.target

# Make train/test split
# As usual in machine learning task we have X_train, y_train, and X_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# print(X_test.shape)
# '''1. just use the Lasso'''
# model = linear_model.LassoCV(random_state=0, n_jobs=-1)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('Final prediction score with optimal: [%.8f](use lasso)' % mean_absolute_error(y_test, y_pred))
#
# '''2. just use the GradientBoostingRegressor'''
# model = GradientBoostingRegressor(random_state=0)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('Final prediction score with optimal: [%.8f](use GBDT)' % mean_absolute_error(y_test, y_pred))
#
# '''3. just use the SVR'''
# model = SVR()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('Final prediction score with optimal: [%.8f](use SVR)' % mean_absolute_error(y_test, y_pred))
#
# '''4. just use the RF'''
# model = RandomForestRegressor(random_state=0, n_jobs=-1,
#                           n_estimators=100, max_depth=3)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('Final prediction score with optimal: [%.8f](use RF)' % mean_absolute_error(y_test, y_pred))
#
# '''5. just use the EXTRA'''
# model = ExtraTreesRegressor(random_state=0, n_estimators=100)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('Final prediction score with optimal: [%.8f](use EXTRA)' % mean_absolute_error(y_test, y_pred))
#
# '''6. just use the XGB'''
# model = XGBRegressor(random_state=0,n_estimators=100, n_jobs=-1)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print('Final prediction score with optimal: [%.8f](use XGB)' % mean_absolute_error(y_test, y_pred))


models = [linear_model.LassoCV(random_state=0, n_jobs=1), GradientBoostingRegressor(random_state=0),
          SVR(),RandomForestRegressor(random_state=0, n_jobs=1,
                          n_estimators=100, max_depth=3),
          ExtraTreesRegressor(random_state=0, n_estimators=100),
          XGBRegressor(random_state=0,n_estimators=100, n_jobs=1)]
# # '''With optimal'''
model = Superknn(models=models, metric=mean_absolute_error, n_jobs=10, random_state=0, folds=5)
# '''1.get the approximate value of epsilon and alpha'''
model.opmimal_parameters(X=X_train, y=y_train, eps_size=0.02, grid_points=10)
#'''2.using grid_search or random_search to find the best parameters around the approximate value '''
# erange = [3,4,4.5]
# #alpha = [5,6]
# # 'folds': folds, 'alpha':alpha
# tuned_parameters = {'epsilon': erange}
# clf = GridSearchCV(Superknn(models=models, metric=mean_absolute_error,
#                          regression=True, random_state=0,
#                          folds=4, alpha=3
#                          ),
#                         param_grid=tuned_parameters, cv=4,
#                    scoring="neg_mean_absolute_error")
# clf.fit(X_train, y_train)
# print(clf.cv_results_, '\n')
# print(clf.best_params_)
# model.epsilon = 3.65
# model.alpha = 6
# start_time = time.time()
# model.fit(X_train, y_train)
# print('fit took %fs!' % (time.time() - start_time))
#
# start_time = time.time()
# y_pred = model.predict(X_test)
#
# # print(y_pred)
# print('predict took %fs!\n' % (time.time() - start_time))
# print('Final prediction score with optimal(use superknn): [%.8f]' % mean_absolute_error(y_test, y_pred))

# epsilons = np.arange(6, 11, 0.5)
# for epsilon in epsilons:
#     print('epsilon:', epsilon)
#     model.epsilon = epsilon
#     model.alpha = 5
#     start_time = time.time()
#     model.fit(X_train, y_train)
#     print('fit took %fs!' % (time.time() - start_time))
#     start_time = time.time()
#     y_pred = model.predict(X_test)
#     print('predict took %fs!' % (time.time() - start_time))
#     # print(y_test)
#     # print(y_pred)
#     print('Final prediction score with optimal(use superknn): [%.8f]\n' % mean_absolute_error(y_test, y_pred))


# '''Without optimal'''
# model = Superknn(models=models,  random_state=0, shuffle=True)
# model.set_epsilon(X_epsilon=X_eps, y_epsilon=y_eps, grid_points=5)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
#
# print('Final prediction score without optimal: [%.8f]' % mean_absolute_error(y_test, y_pred))