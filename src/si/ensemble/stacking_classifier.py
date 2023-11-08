import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy 

class StackingClassifier():

    def __init__ (self, models, final_model:str):

        self.models = models
        self.final_model = final_model

    def fit (self, dataset:Dataset)-> "StackingClassifier":
        predictions = []
        for model in self.models:
            model.fit(dataset)
            pred = model.predict(dataset)
            predictions.append(pred)

        predictions = np.array(predictions).T
        self.final_model.fit(Dataset(dataset.X, predictions))

        return self

    def predict (self, dataset:Dataset)->np.ndarray:
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))
        
        final_pred = self.final_model.predict(Dataset(dataset.X, np.array(predictions).T))
        return final_pred

    def score (self, dataset:Dataset)->float:
        final_pred = self.predict(dataset) 
        score = accuracy(dataset.y, final_pred)
        return score

if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import stratified_train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    from si.models.decision_tree_classifier import DecisionTreeClassifier

    filename = "/Users/tiago_silva/Documents/GitHub/si/datasets/breast_bin/breast-bin.csv"
    breast = read_csv(filename, sep=",", features=True, label=True)
    train_data, test_data = stratified_train_test_split(breast, test_size=0.20, random_state=42)

    #knnregressor
    knn = KNNClassifier(k=3)
    
    #logistic regression
    LG=LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    #decisiontreee
    DT=DecisionTreeClassifier(min_sample_split=3, max_depth=3, mode='gini')

    #final model
    final_model=knn
    models=[knn,LG,DT]
    stack = StackingClassifier(models,final_model)
    stack.fit(train_data)
    print(stack.score(test_data))