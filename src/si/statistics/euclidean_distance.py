import numpy as np

def euclidean_distance(X:np.ndarray, y:np.ndarray) -> np.ndarray:
  return np.sqrt(((X - y)**2).sum(axis=1))