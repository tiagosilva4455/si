
import numpy as np
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile():

    def __init__ (self, percentile:float = 0.5, score_func = f_classification) -> None:
        
        self.score_func = score_func
        self.percentile = percentile
        self.F = None 
        self.p = None

    def fit(self, dataset: Dataset) -> "SelectPercentile":
       self.F, self.p = self.score_func(dataset)
       return self

    def transform(self, dataset: Dataset) -> Dataset:
        n_selection = int(len(dataset.features)*self.percentile) 
        idxs = np.argsort(self.F)[-n_selection:]
        feats_percentile = dataset.X[:, idxs]
        feats_percentile_name = [dataset.features[idx]for idx in idxs]
        return Dataset(feats_percentile, dataset.y, feats_percentile_name, dataset.X)
    
        

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

