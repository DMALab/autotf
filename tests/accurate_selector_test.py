import pandas as pd

from autotf.selector.accurate_selector import AccurateSelector

df = pd.read_csv('~/datasets/MagicTelescope.csv')
df.drop(labels=['ID'], axis=1, inplace=True)

X = df.values[:, :-1]
y = df.values[:, -1]

selector = AccurateSelector(task_type='classification')

params, performance = selector.select_model(X, y)

print(params)
print(performance)
