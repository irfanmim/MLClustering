"""
DBSCAN Clustering
"""

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances

class DBSCAN(BaseEstimator, ClusterMixin):
    def __init__(self, eps=0.1, min_samples=5, metric='euclidean', metric_params=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params

    def fit(self, X, y=None):
        """Fit the clustering model

        Parameters
        ----------
        X : array_like
            the data to be clustered: shape = [n_samples, n_features]
        """
        # metric_kwargs = self.metric_params or {}

        # distances = pairwise_distances(X, metric=self.metric, **metric_kwargs)
        # size = distances.shape[0]

        # # Build groups
        # groups = [None] * size

        # for i in range(0, size):
        #     for j in range(0, size):
        #         if X[i][j] < self.eps:
        #             groups[i].append(j)

        # # Clamp groups
        # clamped_groups = []

        self.labels_ = [0] * size
        return self
