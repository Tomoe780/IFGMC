import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def compute_similarity_matrix(X):
    # 计算最近邻相似性
    nbrs = NearestNeighbors(n_neighbors=6).fit(X)
    distances, indices = nbrs.kneighbors(X)
    N = X.shape[0]
    W = np.zeros((N, N))
    for i in range(N):
        for j, dist in zip(indices[i], distances[i]):
            W[i, j] = np.exp(-dist ** 2)
    return W


class IFGMC:
    def __init__(self, n_components, tol=1e-6, max_iter=100, n_init=2, reg_covar=1e-6,
                 init_params='kmeans', lambda_penalty=0.3):
        self.n_components = n_components
        self.tol = tol
        self.max_iter = max_iter
        self.n_init = n_init
        self.reg_covar = reg_covar
        self.init_params = init_params
        self.lambda_penalty = lambda_penalty

    def _initialize_parameters(self, X):
        N, D = X.shape
        if self.init_params == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_components, n_init=self.n_init, init='k-means++').fit(X)
            self.means_ = kmeans.cluster_centers_
            self.weights_ = np.array([np.mean(kmeans.labels_ == i) for i in range(self.n_components)])
        else:
            self.means_ = X[np.random.choice(N, self.n_components, replace=False)]
            self.weights_ = np.ones(self.n_components) / self.n_components

        self.covariances_ = np.array(
            [np.cov(X, rowvar=False) + self.reg_covar * np.eye(D) for _ in range(self.n_components)]
        )

        # 初始化相似性矩阵
        self.neighbor_graph_ = compute_similarity_matrix(X)

    def _e_step(self, X):
        N, D = X.shape
        self.resp_ = np.zeros((N, self.n_components))

        for k in range(self.n_components):
            cov_matrix = self.covariances_[k]
            norm = multivariate_normal(mean=self.means_[k], cov=cov_matrix, allow_singular=True)
            self.resp_[:, k] = self.weights_[k] * norm.pdf(X)

        # 加入公平性约束的修正
        for i in range(N):
            for k in range(self.n_components):
                fairness_term = self.lambda_penalty * np.sum(
                    self.neighbor_graph_[i, :] * (self.resp_[:, k] - self.resp_[i, k])
                )
                self.resp_[i, k] += fairness_term

        # 归一化责任度
        self.resp_ /= self.resp_.sum(axis=1, keepdims=True)
        return np.sum(np.log(self.resp_.sum(axis=1)))

    def _m_step(self, X):
        N, D = X.shape
        Nk = self.resp_.sum(axis=0)

        self.weights_ = Nk / N
        self.weights_ = np.maximum(self.weights_, 1e-6)
        self.weights_ /= self.weights_.sum()

        self.means_ = np.dot(self.resp_.T, X) / Nk[:, np.newaxis]

        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = np.dot(self.resp_[:, k] * diff.T, diff) / Nk[k] + self.reg_covar * np.eye(D)

    def fit(self, X):
        best_log_likelihood = -np.inf
        best_params = None

        for _ in range(self.n_init):
            self._initialize_parameters(X)
            log_likelihood = None

            for i in range(self.max_iter):
                prev_log_likelihood = log_likelihood
                log_likelihood = self._e_step(X)
                self._m_step(X)

                if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < self.tol:
                    break

            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_params = (self.weights_, self.means_, self.covariances_)

        self.weights_, self.means_, self.covariances_ = best_params

    def predict_proba(self, X):
        N, D = X.shape
        resp = np.zeros((N, self.n_components))

        for k in range(self.n_components):
            cov_matrix = self.covariances_[k]
            norm = multivariate_normal(mean=self.means_[k], cov=cov_matrix, allow_singular=True)
            resp[:, k] = self.weights_[k] * norm.pdf(X)

        resp /= resp.sum(axis=1, keepdims=True)
        print(resp)
        return resp

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
