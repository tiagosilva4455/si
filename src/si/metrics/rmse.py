import numpy as np
from si.metrics.mse import mse

def rmse(y_true:np.ndarray , y_pred:np.ndarray) ->float:
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))
    #return np.sqrt(mse) #Uma vez que a rmse é so a raiz quadrada do mse
