"""
DBSCAN Clustering
"""

from numpy import full

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances


class DBSCAN(BaseEstimator, ClusterMixin):
    OUTLIER = 0
    CORE_POINT = 1
    BORDER_POINT = 2

    def __init__(self,
                 eps=0.1,
                 min_samples=5,
                 metric='euclidean',
                 metric_params=None):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params

    def _discover(self, distances):
        return [i for i, dist in enumerate(distances) if dist <= self.eps]

    def _dfs(self, distance_m, status_list, node_ids_to_visit, member_list,
             member_id):
        while node_ids_to_visit:
            node_id = node_ids_to_visit.pop()

            # Set membership status
            member_list[node_id] = member_id

            # Discover adjacent nodes
            adjacent_node_ids = self._discover(distance_m[node_id])

            if len(adjacent_node_ids) < self.min_samples:
                status_list[node_id] = self.BORDER_POINT
            else:
                status_list[node_id] = self.CORE_POINT

                for adjacent_node_id in adjacent_node_ids:
                    # Skip if node has been visited before
                    if status_list[adjacent_node_id]:
                        continue

                    node_ids_to_visit.append(adjacent_node_id)

    def fit(self, X, y=None):
        """Fit the clustering model

        Parameters
        ----------
        X : array_like
            the data to be clustered: shape = [n_samples, n_features]
        """
        metric_kwargs = self.metric_params or {}

        distance_m = pairwise_distances(X, metric=self.metric, **metric_kwargs)

        nodes_size = distance_m.shape[0]

        member_list = full((nodes_size, ), -1)
        status_list = full((nodes_size, ), self.OUTLIER)

        new_member_id = 0

        for node_id in range(0, nodes_size):
            # Not visiting visited nodes
            if not status_list[node_id]:
                # Discover adjacent nodes
                adjacent_node_ids = self._discover(distance_m[node_id])

                # Not visiting outliers
                if len(adjacent_node_ids) >= self.min_samples:
                    member_id = new_member_id
                    # Increment new member id
                    new_member_id = member_id + 1

                    self._dfs(distance_m, status_list, adjacent_node_ids,
                              member_list, member_id)

        self.labels_ = member_list
        self.status_ = status_list
        return self
