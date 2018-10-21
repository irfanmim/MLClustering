"""
KMeans Clustering
"""

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import pairwise_distances


class KMEANS(BaseEstimator, ClusterMixin):

    def __init__(self,
                 k=2,
                 tolerance=0.0001,
                 max_iteration=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iteration = max_iteration

    def _predict1(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    def fit(self, data):

        self.centroids = {}
        self.label = []

        for i in range(self.k):
            self.centroids[i] = data[i]

            for i in range(self.max_iteration):
                self.classifications = {}

                for i in range(self.k):
                    self.classifications[i] = []

                for featureset in data:
                    distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                    classification = distances.index(min(distances))
                    # print(classification)
                    self.classifications[classification].append(featureset)

                prev_centroids = dict(self.centroids)

                for classification in self.classifications:
                    self.centroids[classification] = np.average(self.classifications[classification],axis=0)

                optimized = True

                for c in self.centroids:
                    original_centroid = dict(self.centroids)[c]
                    current_centroid = self.centroids[c]
                    if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tolerance:
                        print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                        optimized = False

                if optimized:
                    break

        count = 0
        for dat in data:
            # print(self._predict1(dat))
            self.label.append(dat)
            self.label[count] = self._predict1(dat)
            count = count + 1

        print()
        print("centroids")
        print(self.centroids)

        print()
        print("label")
        print(self.label)


        self.labels_ = self.label
        return self
