import numpy as np
from sklearn.neighbors import NearestNeighbors


# 相似度矩阵
def similarity(X):
    n_samples, n_features = X.shape
    # 计算每对样本之间的曼哈顿距离
    diff = np.abs(X[:, np.newaxis, :] - X[np.newaxis, :, :])
    manhattan_distances = np.sum(diff, axis=2) / n_features
    # 转化为相似性度量
    manhattan_matrix = (1 - manhattan_distances) ** 2
    return manhattan_matrix


#####################
# NDP
#####################
def neighbor_discrepancy_penalty(X, labels, k=6):

    similarity_matrix = similarity(X)

    N = X.shape[0]
    penalty = 0

    for i in range(N):
        neighbors = np.argsort(-similarity_matrix[i])[:k + 1]
        neighbors = neighbors[neighbors != i]

        # 计算惩罚项
        weights = similarity_matrix[i, neighbors]
        diff_cluster = (labels[neighbors] != labels[i]).astype(float)
        penalty += np.sum(weights * diff_cluster)

    return 1 - penalty / N
