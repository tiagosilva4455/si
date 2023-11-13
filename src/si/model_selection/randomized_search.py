import numpy as np
from si.data.dataset import Dataset
from si.model_selection.cross_validation import k_fold_cross_validation

def randomized_search_cv(model, dataset, hyperparameter_grid: dict, scoring:str = "k_fold_cross_validation", cv, n_iter:int = 10) -> dict:

    pass