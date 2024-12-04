# import pandas for data wrangling
import pandas as pd


# import numpy for Scientific computations
import numpy as np

from sklearn.model_selection import train_test_split

# import machine learning libraries
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from hyperopt import fmin, tpe, hp

# Input data
data = '../data/Wholesale customers data.csv'
df = pd.read_csv(data)

# Declare feature vector and target variable ¶
X = df.drop('Channel', axis=1)
y = df['Channel']

# convert labels into binary values
y[y == 2] = 0
y[y == 1] = 1

# Split data into separate training and test set ¶
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import scipy.stats as stats

# Define the hyperparameter distributions
param_dist = {
    'max_depth': stats.randint(3, 10),
    'learning_rate': stats.uniform(0.01, 0.1),
    'subsample': stats.uniform(0.5, 0.5),
    'n_estimators':stats.randint(50, 200)
}

# Create the XGBoost model object
xgb_model = xgb.XGBClassifier()

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy')

# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, y_train)

# Print the best set of hyperparameters and the corresponding score
print("Best set of hyperparameters: ", random_search.best_params_)
print("Best score: ", random_search.best_score_)
