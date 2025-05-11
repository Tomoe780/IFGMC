import os
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from IFGMC import IFGMC
from sklearn.mixture import GaussianMixture
import pandas as pd
from metrics import neighbor_discrepancy_penalty


X = pd.read_csv(r"dataset/abalone.csv", sep="\t")
normalizer = MinMaxScaler()
X = normalizer.fit_transform(X)
# 设置聚类数量
K = 6
# 传统EM聚类
EM = GaussianMixture(K)
EM.fit(X)
labels1 = EM.predict(X)


# 计算轮廓系数
silhouette_score_GMC = silhouette_score(X, labels1)
print(f"silhouette_score_GMC: {silhouette_score_GMC}")
# 个体公平指标
NDP_GMC = neighbor_discrepancy_penalty(X, labels1)
print("NDP_GMC:", NDP_GMC)
print("-----------------------------------------")

# IFGMC
gmm2 = IFGMC(K)
gmm2.fit(X)
labels2 = gmm2.predict(X)
silhouette_score_IFGMC = silhouette_score(X, labels2)
print(f"silhouette_score_IFGMC: {silhouette_score_IFGMC}")
NDP_IFGMC = neighbor_discrepancy_penalty(X, labels2)
print("NDP_IFGMC:", NDP_IFGMC)

# visualize_clusters(X, labels1, labels2)
