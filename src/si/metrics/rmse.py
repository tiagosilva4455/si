import numpy as np
from si.metrics.mse import mse

def rmse(y_true:np.ndarray , y_pred:np.ndarray) ->float:
    """
    Calculates the root mean squared error between two arrays.

    Parameters
    ----------
    y_true: numpy.ndarray
        The true values.
    y_pred: numpy.ndarray
        The predicted values.

    Returns
    -------
    float
        The root mean squared error between the two arrays.
    """
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))
    #return np.sqrt(mse) #Uma vez que a rmse é so a raiz quadrada do mse
