import numpy as np
from si.data.dataset import Dataset

class PCA:

    def __init__(self, n_components:int = 10) -> None:
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset:Dataset) -> "PCA":
        self.mean = np.mean(dataset.X, axis = 0)
        centered_data = dataset.X - self.mean

        U,S,Vt =np.linalg.svd (centered_data, full_matrices=False)

        self.components = Vt[:self.n_components]

        explained_variance =(S ** 2)/(len(dataset.X)-1)
        self.explained_variance = explained_variance[:self.n_components]
        return self

    def transform (self, dataset: Dataset) -> Dataset:
        centered_data = dataset.X - self.mean
        V = np.transpose(self.components)
        principal_components = np.dot(centered_data, V)
        return Dataset(principal_components, dataset.y, dataset.features, )

    def fit_transform(self,dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)