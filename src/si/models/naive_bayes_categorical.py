import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class NaiveBayesCategorical:
    
    def __init__(self, smothing:float = 1.0) -> None:

        self.smothing = smothing

        self.class_prior = None
        self.features_prob = None

    def fit(self, dataset:Dataset) -> "NaiveBayesCategorical":
        n_samples = dataset.shape()[0]
        n_features = dataset.shape()[1]
        n_classes = len(np.unique(dataset.label))

        class_count = np.zeros(n_classes)
        feature_count = np.zeros((n_classes, n_features))
        class_prior = np.zeros(n_classes)


    def predict(self, dataset:Dataset) -> np.ndarray:

    def score(self, dataset:Dataset, metric:float = accuracy) -> float:
