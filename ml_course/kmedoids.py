"""
KMedoids Clustering
"""

import numpy as np
import random

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances


class KMEDOIDS(BaseEstimator, ClusterMixin):

    def __init__(self,
                 k=2,
                 tolerance=0.0001,
                 max_iteration=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iteration = max_iteration

    def _predict1(self,data):
        distances = [np.sum(np.abs(data-self.centroids[centroid]), axis=-1) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    def fit(self, data):

        self.centroids = {}
        self.label = []

        m, n = data.shape

        for i in range(self.k):
            self.centroids[i] = data[random.randint(0,m-1)]

        newCentroid = np.copy(self.centroids)


        for i in range(self.max_iteration):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            total = 0
            for featureset in data:
                distances = [np.sum(np.abs(featureset-self.centroids[centroid]), axis=-1) for centroid in self.centroids]
                classification = distances.index(min(distances))
                total = total + np.sum(distances)
                self.classifications[classification].append(featureset)
            
            prev_total = total

            prev_centroids = dict(self.centroids)

            intRandom = random.randint(0, self.k-1)
            self.centroids[intRandom] = data[random.randint(0,m-1)]

            total = 0
            for featureset in data:
                distances = [np.sum(np.abs(featureset-self.centroids[centroid]), axis=-1) for centroid in self.centroids]
                classification = distances.index(min(distances))
                total = total + np.sum(distances)
                self.classifications[classification].append(featureset)

            current_total = total

            optimized = False
            index = 0
            if (current_total > prev_total):
                self.centroids = prev_centroids
                if (current_total-prev_total<self.tolerance):
                    index = i
                    optimized = True
                    break

        count = 0
        for dat in data:
            self.label.append(dat)
            self.label[count] = self._predict1(dat)
            count = count + 1


        self.labels_ = self.label
        return self
