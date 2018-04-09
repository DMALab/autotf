from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from ensemble import subsemble

# Prepare data
boston = load_boston()
X, y = boston.data, boston.target

# Make train/test split
# As usual in machine learning task we have X_train, y_train, and X_test
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=0)

print(X_train.shape)

# First-layer model
models_1 = [
    ExtraTreesRegressor(random_state=0, n_jobs=-1,
                        n_estimators=100, max_depth=3),

    XGBRegressor(random_state=0, n_jobs=-1, learning_rate=0.1,
                 n_estimators=100, max_depth=3)
]
# Second-layer model
models_2 = [GradientBoostingRegressor(), SVR(), RandomForestRegressor()]

# Meta-layer model
meta_model = XGBRegressor(seed=0, n_jobs=-1, learning_rate=0.1, n_estimators=100, max_depth=3)

# Construct stacking
ensemble = subsemble(X_train, y_train, X_test, num_splits=3, save_dir='.',
                    regression=True, metric=mean_absolute_error, n_folds=3,
                    shuffle=True, random_state=0, verbose=2)

# First layer
ensemble.add(models_1, propagate_features=[0, 1])
print(ensemble.next_train[:5]) # we expect 5 columns as we propagate the 0 and 1 column of the first layer data
print("test the function of the propagate:")
print(ensemble.next_train.shape)
if ensemble.next_train.shape[1] == 8:  # n_models(2) * n_subset(3) + progate(2)
    print("     pass the test!")


# Second layer
ensemble.add(models_2, subset=[0, 1], propagate_features=[0])
print("test the function of the subset")
if ensemble.next_train.shape[1] == 12:  # n_models(3) * n_subset(3) + subset(2) + progate(1)
    print("     pass the test!")
print(ensemble.next_train[:5])

# Meta layer
y_pred = ensemble.add_meta(meta_model)

print('Final prediction score: [%.8f]' % mean_absolute_error(y_test, y_pred))