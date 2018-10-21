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

            for featureset in data:
                distances = [np.sum(np.abs(featureset-self.centroids[centroid]), axis=-1) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)


            # print()
            # print("prev_centroid")
            # print(prev_centroids)

            # harusnya cuman ganti sentroid 1 kali dan itu dijadiin random
            intRandom = random.randint(0, self.k-1)
            self.centroids[intRandom] = np.average(self.classifications[intRandom],axis=0)

            # print(intRandom)
            # print()
            # print("next_centroid")
            # print(self.centroids)

            # cek errornya 


            # cek anggotanya sama atau engga


            # cek tolerance

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tolerance:
                    optimized = optimized and False

            if optimized:
                break


        count = 0
        for dat in data:
            self.label.append(dat)
            self.label[count] = self._predict1(dat)
            count = count + 1

        # print()
        # print("centroids")
        # print(self.centroids)

        print()
        print("label")
        print(self.label)

        print()
        print("optimized")
        print(optimized)


        self.labels_ = self.label
        return self
