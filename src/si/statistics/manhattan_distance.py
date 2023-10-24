import numpy as np

def manhattan_distance (x:np.ndarray,y:np.ndarray)-> np.ndarray:
    return np.abs((x - y).sum(axis = 1))

if __name__ == '__main__':
    x = np.array([1, 2, 3])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    my_manhattan_distance = manhattan_distance(x, y)
    
    from sklearn.metrics.pairwise import manhattan_distances
    sklearn_distance = manhattan_distances(x.reshape(1, -1), y)
    assert np.allclose(my_manhattan_distance, sklearn_distance)
    print(my_manhattan_distance, sklearn_distance)