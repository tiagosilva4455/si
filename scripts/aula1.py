import si
from si.data.dataset import Dataset

import si.io.csv_file
import si.io.data_file

iris = si.io.csv_file.read_csv("/Users/tiago_silva/Documents/GitHub/si/datasets/iris/iris.csv")

print(iris)