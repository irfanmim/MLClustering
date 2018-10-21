"""
Agglomerative Clustering
"""
import numpy as np
from scipy.spatial.distance import pdist

class Agglomerative():
    def __init__(self, n_clusters, linkage)
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.dist_mat = []
        self.clusters = []
        self.agglomerative_step = []

    def fit(self, X, y=None):
        clusters = X
        dist_mat = pdist(X, 'euclidean')
        
        for i in range(clusters.shape[0] - n_clusters):
            self._join_cluster()
            self._update_dist_mat

        return

    def _join_cluster(self):
        # Find minimal distance from distance matrix
        min = -1
        for i in range(0, len(dist_mat)):
            for j in range(i, len(dist_mat[i])):
                if min > 0 and min > dist_mat[i][j]
                    min = dist_mat[i][j]
                    idxmin = [i, j]

        agglomerative_step.append[idxmin[0], idxmin[1]]

        # Append cluster with minimum distance
        for i in range(len(self.clusters[idxmin[1]])):
            self.clusters[idxmin[0]].append(self.clusters[idxmin[1]][i])
        del self.clusters[idxmin[1]]

    def _update_dist_mat(self):
        last = agglomerative_step[-1]
        updated_cluster = last[0]
        new_dist = []
        for x in range(dist_mat.shape[0]-1)
            dist = self._calc_distance(dist_mat[updated_cluster], dist_mat[x])
            new_dist.append(dist)
        dist_mat[updated_cluster] = new_dist
        dist_mat[:, updated_cluster] = new_dist
        return None

    def _calc_distance(self, clust_one, clust_two):
        linkage = self.linkage
        if linkage == "single" :
            min = -1
            for i in clust_one:
                for j in clust_two:
                    dist = np.linalg.norm(clust_one[i] - clust_two[j])
                    if min == -1 or dist < min:
                        min = dist
            return min

        elif linkage == "complete" :
            max = -1
            for i in clust_one:
                for j in clust_two:
                    dist = np.linalg.norm(clust_one[i] - clust_two[j])
                    if max == -1 or dist > max:
                        max = dist
            return max  

        elif linkage "average_group" :
            sum = 0
            for i in clust_one:
                sum += clust_one[i]
            avg_one = float(sum) / len(clust_one)

            sum = 0
            for i in clust_two:
                sum += clust_two[i]
            avg_two = float(sum) / len(clust_two)
            dist = np.linalg.norm(avg_one - avg_two)
            return dist

        elif linkage = "average" :
            sum = 0
            for i in clust_one:
                for j in clust_two:
                    sum += np.linalg.norm(clust_one[i] - clust_two[j])
            divisor = len(clust_one) * len(clust_two)
            dist = float(sum) / divisor
            return dist
          
    

