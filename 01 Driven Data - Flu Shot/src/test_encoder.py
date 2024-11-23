# from skmultilearn.dataset import load_dataset
# X, y, _, _ = load_dataset('emotion', 'train')
import numpy as np
import pandas as pd


data = pd.read_parquet('../data/train-00000-of-00001.parquet', engine='fastparquet')
X = data.iloc[:,0].to_numpy()
y = data.iloc[:, 1]

print(X, type(X))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
print(X_train.shape, X_test.shape)
