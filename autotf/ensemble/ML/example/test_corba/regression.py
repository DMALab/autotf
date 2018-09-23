from ensemble import Cobra
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

# Prepare data
boston = load_boston()
X, y = boston.data, boston.target

# Make train/test split
# As usual in machine learning task we have X_train, y_train, and X_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
_, X_eps, _, y_eps = train_test_split(X, y, test_size=0.7, random_state=0)


models = [KNeighborsRegressor(weights='distance'),
          DecisionTreeRegressor(random_state=0),
          svm.SVR(),
          BayesianRidge()
]
'''With optimal'''
cobra = Cobra(models=models, metric=mean_absolute_error, random_state=0)
# cobra.epsilon = 10.381571140368733
# cobra.alpha = 4
# cobra.k = 0.6
cobra.opmimal_parameters(X=X_train, y=y_train, eps_size=0.8, grid_points=10,verbose=2)
cobra.fit(X_train, y_train)
y_pred = cobra.predict(X_test)
# print(y_test)
# print(y_pred)
print('Final prediction score with optimal: [%.8f]' % mean_absolute_error(y_test, y_pred))


# '''Without optimal'''
# cobra = Cobra(models=models,  random_state=0, shuffle=True)
# cobra.set_epsilon(X_epsilon=X_eps, y_epsilon=y_eps, grid_points=5)
# cobra.fit(X_train, y_train)
# y_pred = cobra.predict(X_test)
#
# print('Final prediction score without optimal: [%.8f]' % mean_absolute_error(y_test, y_pred))