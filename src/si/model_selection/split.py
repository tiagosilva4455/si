from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split (dataset:Dataset, test_size:float = 0.2, random_state:int = 42) -> Tuple[Dataset,Dataset]:
    """
    Split the dataset into training and testing sets.

    Parameters
    ----------
    dataset: Dataset
        The dataset to split.
    test_size: float
        The proportion of the dataset to include in the test split.
    random_state: int
        The seed of the random number generator.

    Returns
    -------
    train: Dataset
        The training dataset.
    test: Dataset
        The testing dataset.

    """
    unique_labels, label_counts = np.unique(dataset.y, return_counts=True)

    train_idxs =[]
    test_idxs=[]

    for label, count in zip(unique_labels, label_counts):
        num_test_samples = int(test_size * count)
        np.random.seed(random_state)
        idxs = np.where(dataset.y == label)[0] #identifica o indice onde a variavel target y é igual ao label atual, encontra os pontos qque pertencem a uma dada classe
        np.random.shuffle(idxs)

        test_idxs.extend(idxs[:num_test_samples])
        train_idxs.extend(idxs[num_test_samples:])

    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

    
    #test with iris dataset
if __name__ == "__main__":
    from sklearn import datasets

    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    dataset = Dataset(X, y, features=feature_names, label='target')
    
    #test using your stratified_train_test_split function
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

    print(train_dataset.shape())
    print(test_dataset.shape())


    #test using your stratified_train_test_split function
    train_dataset, test_dataset = stratified_train_test_split(dataset, test_size=0.2, random_state=42)

    print(train_dataset.shape())
    print(test_dataset.shape())
