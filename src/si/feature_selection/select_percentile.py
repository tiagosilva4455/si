
import numpy as np
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile():

    def __init__ (self, percentile:float = 0.0, score_func = f_classification) -> None:
        
        self.score_func = score_func
        self.percentile = percentile
        self.F = None 
        self.p = None

    def fit(self, dataset: Dataset) -> "SelectPercentile":
       self.percentile = np.percentile(dataset.X, axis=0)

    def transform(self, dataset: Dataset) -> Dataset: 
        X= dataset.X

        features_mask = self.percentile

    def fit_transform(self,dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)
    
if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                    [0, 1, 4, 3],
                                    [0, 1, 1, 3]]),
                        y=np.array([0, 1, 0]),
                        features=["f1", "f2", "f3", "f4"],
                        label="y")

    selector = SelectPercentile()
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)

