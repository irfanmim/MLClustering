"""
Agglomerative Clustering
"""
import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from scipy.spatial.distance import pdist, squareform

class Agglomerative(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=4, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.dist_mat = []
        self.clusters = []
        self.agglomerative_step = []
        self.data = []

    def fit(self, X, y=None):
        # Initiate index clusters
        for i in range(len(X)):
            temp = [i]
            self.clusters.append(temp)
            self.data.append([X[i]])

        # Initiate empty labels
        self.labels_ = np.zeros(len(X))

        # Calculate initial distance matrix
        self.dist_mat = squareform(pdist(X, 'euclidean'))

        # Loop until desired number of clusters
        for i in range(len(self.clusters) - self.n_clusters):
            self._join_cluster()
            self._update_dist_mat()

        for i in range(len(self.clusters)):
            for j in range(len(self.clusters[i])):
                self.labels_[self.clusters[i][j]] = i

        return self

    def _join_cluster(self):
        # Find minimal distance from distance matrix
        min = 999
        for i in range(0, len(self.dist_mat)):
            for j in range(i, len(self.dist_mat[i])):
                if self.dist_mat[i][j] > 0 and min > self.dist_mat[i][j] and i != j:
                    min = self.dist_mat[i][j]
                    idxmin = [i, j]

        self.agglomerative_step.append(idxmin)

        # Append cluster with minimum distance
        for i in range(len(self.clusters[idxmin[1]])):
            self.clusters[idxmin[0]].append(self.clusters[idxmin[1]][i])
            self.data[idxmin[0]].append(self.data[idxmin[1]][i])
        
        # Delete the merged cluster and corresponding distance matrix
        del self.clusters[idxmin[1]]
        del self.data[idxmin[1]]
        self.dist_mat = np.delete(self.dist_mat, idxmin[1], axis=0)
        self.dist_mat = np.delete(self.dist_mat, idxmin[1], axis=1)
        # del self.dist_mat[idxmin[1]]
        # del self.dist_mat[:, idxmin[1]]
        return None

    def _update_dist_mat(self):
        last = self.agglomerative_step[-1]
        updated_cluster = last[0]
        new_dist = []
        for x in range(0, self.dist_mat.shape[0]):
            dist = self._calc_distance(self.data[updated_cluster], self.data[x])
            new_dist.append(dist)
        self.dist_mat[updated_cluster] = new_dist
        self.dist_mat[:, updated_cluster] = new_dist
        return None

    def _calc_distance(self, clust_one, clust_two):
        linkage = self.linkage
        if linkage == "single" :
            min = -1
            for i in clust_one:
                for j in clust_two:
                    dist = np.linalg.norm(i - j)
                    if min == -1 or dist < min:
                        min = dist
            return min

        elif linkage == "complete" :
            max = -1
            for i in clust_one:
                for j in clust_two:
                    dist = np.linalg.norm(i - j)
                    if max == -1 or dist > max:
                        max = dist
            return max  

        elif linkage == "average_group" :
            sum = 0
            for i in clust_one:
                sum += i[0]
            avg_one = float(sum) / float(len(clust_one))

            sum = 0
            for i in clust_two:
                sum += i[0]
            avg_two = float(sum) / float(len(clust_two))
            dist = np.linalg.norm(avg_one - avg_two)
            return dist

        elif linkage == "average" :
            sum = 0
            for i in clust_one:
                for j in clust_two:
                    sum += np.linalg.norm(i - j)
            divisor = len(clust_one) * len(clust_two)
            dist = float(sum) / float(divisor)
            return dist
          
    

