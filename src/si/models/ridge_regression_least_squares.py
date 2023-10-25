import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares:

    def __init__(self, l2_penalty:float ,scale:bool = True) -> None:

        self.l2_penalty = l2_penalty
        self.scale = scale

        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, dataset:Dataset)-> "RidgeRegressionLeastSquares":
        if self.scale  == True:
            self.mean = np.nanmean (dataset.X)
            self.std = np.nanstd (dataset.X)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        X=np.c_[np.ones(X.shape[0]),X]

        penalty_m = self.l2_penalty * np.eye(X.shape[1])
        penalty_m [0,0] = 0

        transposed_x = np.transpose(X)

        XtX= np.linalg.inv(np.dot(transposed_x,X) + penalty_m) 
        Xty= np.dot(transposed_x,dataset.y)

        thetas = np.dot(XtX,Xty)
        self.theta_zero = thetas[0]
        self.theta = thetas[1:]

        return self
    
    def predict (self, dataset:Dataset) -> np.ndarray:
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        X = np.c_[np.ones(X.shape[0]),X]
        return np.dot(X, np.r_[self.theta, self.theta_zero])
    
    def score (self, dataset:Dataset) -> float:
        predictions = self.predict(dataset)
        return mse(dataset.y, predictions)

# This is how you can test it against sklearn to check if everything is fine
if __name__ == '__main__':
    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegressionLeastSquares(l2_penalty=2.0)
    model.fit(dataset_)
    print(model.theta)
    print(model.theta_zero)

    # compute the score
    print(model.score(dataset_))

    # compare with sklearn
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=2.0)
    # scale data
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    print(model.coef_) # should be the same as theta
    print(model.intercept_) # should be the same as theta_zero
    print(mse(dataset_.y, model.predict(X)))
