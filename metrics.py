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


# 局部一致性（越高越好）
def local_consistency_score(dataset, cluster_labels, k):
    neighbors = NearestNeighbors(n_neighbors=k).fit(dataset)
    _, indices = neighbors.kneighbors(dataset)

    total_consistency = 0
    N = dataset.shape[0]

    for i in range(N):
        target_label = cluster_labels[i]
        neighbor_labels = cluster_labels[indices[i]]
        # 计算邻居中与目标样本属于同一簇的样本比例
        same_cluster_count = np.sum(neighbor_labels == target_label)
        consistency_score = same_cluster_count / k
        total_consistency += consistency_score

    # 平均局部一致性得分
    return total_consistency / N


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
