import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from autotf.selector.accurate_selector import AccurateSelector

cod_rna = load_svmlight_file("/home/daim_gpu/xuehuanran/datasets/codrna/cod-rna", n_features=8)
cod_rna_test = load_svmlight_file("/home/daim_gpu/xuehuanran/datasets/codrna/cod-rna.t", n_features=8)

x, y = cod_rna[0].toarray().astype(np.float32), cod_rna[1].astype(int)
y[np.where(y == -1)] = 0
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, stratify=y)
x_test, y_test = cod_rna_test[0].toarray().astype(np.float32), cod_rna_test[1].astype(int)
y_test[np.where(y_test == -1)] = 0

selector = AccurateSelector("common_classification")
selector.select_model([x_train, x_valid, x_test], [y_train, y_valid, y_test], feature_num=8, class_num=2)
