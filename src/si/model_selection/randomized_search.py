import itertools
from typing import Callable, Tuple, Dict, Any
import numpy as np
from si.data.dataset import Dataset
from si.model_selection.cross_validation import k_fold_cross_validation

def randomized_search_cv(model, dataset:"Dataset", hyperparameter_grid: Dict[str, Tuple], scoring:Callable = None, cv:int = 5, n_iter:int = 10) -> dict:
    """
    Performs randomized search cross validation.

    Create an efficient parameter optimization strategy with cross-validation by randomly selecting 'n' hyperparameter combinations from a distribution. This approach is well-suited for large datasets, offering a good solution in less time, although it may not be the optimal one.

    Parameters
    ----------
    model
        Model to cross validate
    dataset: Dataset
        The dataset to cross validate
    hyperparameter_grid: dict[str, Tuple]
        The hyperparameter grid
    scoring: Callable
        The scoring function
    cv: int
        The number of folds
    n_iter: int
        number of iterations

    Returns
    -------
    results: dict
        The results of the grid search cross validation. Gives the score, hyperparameters, best hyperparameters, and best score.
    """

    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise ValueError(f"Model {model} does not have attribute {parameter}") # check if model has hyperparameter to test

    results = {"scores": [], "hyperparameters": []}

    for x in range(n_iter):
        parameters = {}
        for keys, values in hyperparameter_grid.items():  # i'm separating keys from possible values so that I can choose random values later
            valores_random = np.random.choice(values)
            parameters[keys] = valores_random
            setattr(model, keys, valores_random)

        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)# crossvalidation of the model

        results['scores'].append(np.mean(score))  # model with all combination of hyperparameters saved and respective scores

        results['hyperparameters'].append(parameters)  # each hyperparameter to the value respective

    results['best_hyperparameters'] = results['hyperparameters'][np.argmax(results['scores'])]  # hyperparameters with higher scores
    results['best_score'] = np.max(results['scores'])  # choosing max score
    return results

if __name__ == '__main__':
    # import dataset
    from si.models.logistic_regression import LogisticRegression

    num_samples = 600
    num_features = 100
    num_classes = 2

    # random data
    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(0, num_classes, size=num_samples)  # classe aleatórios

    dataset_ = Dataset(X=X, y=y)

    #  features and class name
    dataset_.features = ["feature_" + str(i) for i in range(num_features)]
    dataset_.label = "class_label"

    # initialize the Logistic Regression model
    knn = LogisticRegression()

    # parameter grid
    parameter_grid_ = {
        'l2_penalty': (1, 10),
        'alpha': (0.001, 0.0001),
        'max_iter': (1000, 2000)
    }

    # cross validate the model
    results_ = randomized_search_cv(knn,dataset_, hyperparameter_grid=parameter_grid_, cv=3,n_iter=8)

    # print the results
    print(results_)

    # get the best hyperparameters
    best_hyperparameters = results_['best_hyperparameters']
    print(f"Best hyperparameters: {best_hyperparameters}")

    # get the best score
    best_score = results_['best_score']
    print(f"Best score: {best_score}")