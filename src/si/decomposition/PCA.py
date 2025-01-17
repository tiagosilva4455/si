import numpy as np
from si.data.dataset import Dataset

class PCA:
"""
    Principal Component Analysis (PCA) is a linear dimensionality reduction technique that can be utilized for extracting information from a high-dimensional space by projecting it into a lower-dimensional sub-space.
"""
    def __init__(self, n_components:int = 10) -> None:
        """
        Constructor for PCA class.

        Parameters
        ----------
        n_components: int
            The number of components to keep.

        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, dataset:Dataset) -> "PCA":
        """
        Fit the PCA class.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit.

        Returns
        -------
        self: PCA
            The fitted PCA class.

        """
        self.mean = np.mean(dataset.X, axis = 0)
        centered_data = dataset.X - self.mean

        U,S,Vt =np.linalg.svd (centered_data, full_matrices=False)

        self.components = Vt[:self.n_components]

        explained_variance =(S ** 2)/(len(dataset.X)-1)
        self.explained_variance = explained_variance[:self.n_components]
        return self

    def transform (self, dataset: Dataset) -> Dataset:
        """
        Reduce X to the selected number of components.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.

        """
        centered_data = dataset.X - self.mean
        V = np.transpose(self.components)
        principal_components = np.dot(centered_data, V)
        return Dataset(principal_components, dataset.y, dataset.features[:self.n_components], dataset.label)

    def fit_transform(self,dataset: Dataset) -> Dataset:
        """
        Fit the PCA class and reduce X to the selected number of components.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit and transform.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.

        """
        self.fit(dataset)
        return self.transform(dataset)
    
if __name__ == '__main__':

    from si.data.dataset import Dataset

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    print(X.shape)
    
    label = 'target'
    features = ['feature1', 'feature2', 'feature3']
    dataset = Dataset(X, y, features, label)

    sklearn_pca = PCA(n_components=2)
    sklearn_pca.fit(dataset)

    sklearn_transformed_data = sklearn_pca.transform(dataset)

    your_pca = PCA(n_components=2)
    your_pca.fit(dataset)
    your_transformed_dataset = your_pca.transform(dataset)

    print("scikit-learn Transformed Data:")
    print(sklearn_transformed_data.)
    print("Your PCA Transformed Data:")
    print(your_transformed_dataset.X)