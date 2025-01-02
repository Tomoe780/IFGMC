import os
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from IFGMC import IFGMC
import pandas as pd
from visualization import visualize_clusters
from metrics import local_consistency_score, neighbor_discrepancy_penalty

# X = pd.read_csv(r"./dataset/bank.csv")
# X = X[['balance', 'duration']].values
X = pd.read_csv(r"./dataset/gap.csv", sep="\t")
# 取一部分数据点
# num_samples = 2000
# random_indices = np.random.choice(X.shape[0], num_samples, replace=False)
# X = X[random_indices, :]
# 缩放到[0, 1]
normalizer = MinMaxScaler()
X = normalizer.fit_transform(X)

# 设置聚类数量
K = 6

# GMM using Standard EM Algorithm
gmm = GaussianMixture(K)
gmm.fit(X)
labels1 = gmm.predict(X)
# 计算轮廓系数
silhouette_score_GMC = silhouette_score(X, labels1)
print(f"silhouette_score_GMC: {silhouette_score_GMC}")
# 个体公平指标
score1 = local_consistency_score(X, labels1, 6)
score11 = neighbor_discrepancy_penalty(X, labels1)
print("score1:", score1)
print("score11:", score11)
print("-----------------------------------------")

gmm2 = IFGMC(K)
gmm2.fit(X)
labels2 = gmm2.predict(X)
silhouette_score_IFGMC = silhouette_score(X, labels2)
print(f"silhouette_score_IFGMC: {silhouette_score_IFGMC}")
score2 = local_consistency_score(X, labels2, 6)
score22 = neighbor_discrepancy_penalty(X, labels2)
print("score2:", score2)
print("score22:", score22)

visualize_clusters(X, labels1, labels2)
