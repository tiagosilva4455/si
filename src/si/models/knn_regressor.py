import numpy as np 
from typing import Callable
from si.data.dataset import Dataset
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance

class KNNRegressor:

    def __init__ (self,k:int = 1, distance:Callable = euclidean_distance):
        self.k = k
        self.distance =distance

        self.dataset = None

    def fit(self, dataset:Dataset) -> "KNNRegressor":
        self.dataset=dataset
        return self
    
    def closest_value(self, dataset:Dataset) -> np.ndarray:
        distances = self.distance(dataset.X,dataset.y)
        k_indices = np.argsort(distances)[:self.k]
        return np.mean(self.dataset.y[k_indices])
        
    def predict(self, dataset:Dataset) -> np.ndarray:
        predictions = np.apply_along_axis(self.closest_value(dataset), axis = 1, arr=dataset.X)
        return predictions

    def score(self,dataset:Dataset) -> float:
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







