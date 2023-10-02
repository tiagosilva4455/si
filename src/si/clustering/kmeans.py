from si.data.dataset import Dataset
import numpy as np
from si.statistics.euclidean_distance import euclidean_distance

class KMeans():
    def __init__(self, k:int, max_iter:int, distance = euclidean_distance) -> None:
        self.k = k
        self.max_iter = max_iter
        self.distance = distance

        self.centroids = None
        self.labels = None 

    def _init_centroids(self, dataset:Dataset) -> None:
        seeds = np.random.permutation(dataset.shape()[0])[:self.k]
        self.centroids =dataset.X[seeds]    

    def _get_closest_centroid (self, sample:np.ndarray) -> None:
        distances = self.distance(sample, self.centroids)
        closest = np.argmin(distances)
        return self.centroids[closest]
    
    def _get_distances(self, sample:np.ndarray) -> None:
        return self.distance (sample, self.centroids)

    def fit(self, dataset:Dataset) -> "KMeans":
        self._init_centroids(dataset)

        convergence = False
        i = 0
        labels = np.zeros (dataset.shape()[0])

        while not convergence and i < self.max_iter:
            closest_centroids = np.apply_over_axes(self._get_closest_centroid, axis=1, arr= dataset.X)
            centroids = []

            for j in range (self.k):
                centroid = np.mean (dataset.X[closest_centroids==j], axis=0)
                centroids.append(centroid)
            centroids = np.array(centroid)

            convergence = not np.any(labels != closest_centroids)
            labels = closest_centroids
            self.labels = labels
            return self

    def transform(self):
        centroid_distances

    def fit_transform (self, dataset):
        self.fit(dataset)
        return self.transform(dataset)
    
    def predict():