import pandas as pd
import numpy as np
import math
from sklearn.neighbors import NearestNeighbors


def get_rank(algo_list, dataset):
    """
    Return a rank of algorithms in algo_list with the given dataset.
    Reference:
    Brazdil P, Soares C, da Costa JP (2003)
    Ranking learning algorithms: using IBL and meta-learning on accuracy and time results.
    Machine Learning 50(3):251-77.
    """
    return []


# def get_real_rank(dataset_id):
#     accuracy = benchmark.loc[dataset_id, algorithm_list].values
#     algo_indices = np.argsort(accuracy)[::-1]
#     rank = [-1] * 7
#     order = 1
#     for indice in algo_indices:
#         rank[indice] = order
#         order += 1
#     return rank
#
#
# def get_recommend_rank(k_dataset):
#     accuracy = np.array([-1.0] * 7)
#     for i in range(7):
#         sum = 0.0
#         for j in range(7):
#             product = 1.0
#             for k in k_dataset[0]:
#                 product *= arr[k][i][j]
#             sum += math.pow(product, 1.0 / len(k_dataset[0]))
#         accuracy[i] = float(sum) / 7.0
#
#     algo_indices = np.argsort(accuracy)[::-1]
#     rank = [-1] * 7
#     order = 1
#     for indice in algo_indices:
#         rank[indice] = order
#         order += 1
#     return rank
#
#
# def evaluate_rank(rank1, rank2):
#     m = len(rank1)
#     rank1 = np.array(rank1)
#     rank2 = np.array(rank2)
#     return 1 - 6 * sum((rank1 - rank2) ** 2) / (m ** 3 - m)
#
#
# benchmark = pd.read_csv('./benchmark.csv')
#
# dataset_num = benchmark.shape[0]
#
# algorithm_list = ['LogisticRegressionAccuracy', 'SvmAccuracy', 'NaiveBayesAccuracy', 'DecisionTreeAccuracy',
#                   'AdaBoostAccuracy', 'RandomTreeAccuracy', 'GbdtAccuracy']
#
# arr = []
# for i in range(dataset_num):
#     arr_pq = np.zeros(shape=(7, 7))
#     for j in range(7):
#         for k in range(7):
#             sr_p = benchmark.loc[i, algorithm_list[j]]
#             sr_q = benchmark.loc[i, algorithm_list[k]]
#             arr_pq[j][k] = sr_p / sr_q
#     arr.append(arr_pq)
#
#
# knn = NearestNeighbors(n_neighbors=10)
# feat_list = ['NumberOfInstances', 'NumberOfClasses', 'NumberOfFeatures',
#              'PercentageOfSymbolicFeatures', 'PercentageOfMissingValues', 'ClassEntropy']
# rank_evaluation = []
# for dataset_id in range(dataset_num):
#     train_list = list(range(dataset_num))
#     train_list.remove(dataset_id)
#     X = benchmark.loc[train_list, feat_list].values
#     test_point = benchmark.loc[dataset_id, feat_list].values.reshape((1, 6))
#     knn.fit(X)
#     _, indices = knn.kneighbors(test_point)
#     real_rank = get_real_rank(dataset_id)
#     recommend_rank = get_recommend_rank(k_dataset=indices)
#     print(real_rank, recommend_rank)
#     rank_evaluation.append(evaluate_rank(real_rank, recommend_rank))
#
#
# cnt = 0
# for e in rank_evaluation:
#     if e > 0.5:
#         cnt += 1
# print(cnt)
# print(cnt / 68)
# print(rank_evaluation)
#
# print(np.average(rank_evaluation))
