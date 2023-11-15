import numpy as np 
from typing import Callable
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance

class KNNRegressor:
"""
    K-Nearest Neighbors regressor
"""
    def __init__ (self,k:int = 1, distance:Callable = euclidean_distance):
        """
        Constructor for KNNRegressor class.

        Parameters
        ----------
        k: int
            The number of neighbors to use.
        distance: Callable
            The distance function to use.

        """
        self.k = k
        self.distance =distance

        self.dataset = None

    def fit(self, dataset:Dataset) -> "KNNRegressor":
        """
        Fit the KNNRegressor class.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit.

        Returns
        -------
        self: KNNRegressor
            The fitted KNNRegressor class.
        """
        self.dataset=dataset
        return self
    
    def closest_value(self, dataset:Dataset) -> np.ndarray:
        """
        Find the closest value to a given sample.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit.

        Returns
        -------
        np.ndarray
            The closest value to a given sample.

        """
        distances = self.distance(dataset.X,dataset.y)
        k_indices = np.argsort(distances)[:self.k]
        return np.mean(self.dataset.y[k_indices])
        
    def predict(self, dataset:Dataset) -> np.ndarray:
        """
        Predict the values for a given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict.
        Returns
        -------
        predictions: np.ndarray
            The predicted values.
        """
        predictions = np.apply_along_axis(self.closest_value(dataset), axis = 1, arr=dataset.X)
        return predictions

    def score(self,dataset:Dataset) -> float:
        """
        Evaluate the model on a given dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate.

        Returns
        -------
        score: float
            The score of the model.
        """

        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions) 


if __name__ == "__main__":
    from si.model_selection.split import train_test_split
    num_samples = 600
    num_features = 100

    X = np.random.rand(num_samples, num_features)
    y = np.random.rand(num_samples) 

    dataset_ = Dataset(X=X, y=y)

    #features and class name 
    dataset_.features = ["feature_" + str(i) for i in range(num_features)]
    dataset_.label = "target"

    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # regressor KNN
    knn_regressor = KNNRegressor(k=5)  

    # fit the model to the train dataset
    knn_regressor.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn_regressor.score(dataset_test)
    print(f'The rmse of the model is: {score}')







