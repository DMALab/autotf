from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ensemble import Stacking
from sklearn.datasets import make_classification
from sklearn.metrics import log_loss

n_classes = 3

# Create data: 500 example, 5 feature, 3 classes
X, y = make_classification(n_samples=500, n_features=5,
                           n_informative=3, n_redundant=1,
                           n_classes=n_classes, flip_y=0,
                           random_state=0)

# Make train/test split
# As usual in machine learning task we have X_train, y_train, and X_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('Train shape:', X_train.shape)
print('Test shape: ', X_test.shape)


models = [ExtraTreesClassifier(n_jobs=1, random_state=0),
    RandomForestClassifier(n_jobs=1, random_state=0),

    XGBClassifier(n_jobs=1, random_state=0)
]

ensemble = Stacking(X_train, y_train, X_test, regression=False, bagged_pred=False,
                    needs_proba=True, save_dir=None, metric=log_loss,
                    n_folds=4, stratified=True, shuffle=True,
                    random_state=0, verbose=0)

ensemble.add(models)
print(ensemble.next_test[0:3])

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

print(y_pred2[0:3])
