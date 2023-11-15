
import numpy as np
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile():
"""
    Select features according to a percentile of the highest scores.
"""
    def __init__ (self, percentile:float = 0.5, score_func = f_classification) -> None:
        """
        Initialize the SelectPercentile class

        Parameters
        ----------
        percentile: float, default=0.5
            Percent of features to keep.
        score_func: callable, default=f_classification
            Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores. Default is f_classification.

        """
        self.score_func = score_func
        self.percentile = percentile
        self.F = None 
        self.p = None

    def fit(self, dataset: Dataset) -> "SelectPercentile":
        """
        Fit the SelectPercentile class.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit.

        Returns
        -------
        self: SelectPercentile
            The fitted SelectPercentile class.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Reduce X to the selected features.

        Parameters
        ----------
        dataset: Dataset
            The dataset to transform.

        Returns
        -------
        dataset: Dataset
            The transformed dataset.
        """
        n_selection = int(len(dataset.features)*self.percentile) 
        idxs = np.argsort(self.F)[-n_selection:]
        feats_percentile = dataset.X[:, idxs]
        feats_percentile_name = [dataset.features[idx]for idx in idxs]
        return Dataset(feats_percentile, dataset.y, feats_percentile_name, dataset.X)

        

    def fit_transform(self,dataset: Dataset) -> Dataset:
        """
        Fit the SelectPercentile class and reduce X to the selected features.
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

