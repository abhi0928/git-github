# Importing the libraries for computation and analysis
import numpy as np 
import pandas as pd 

# Loading the Real Estates dataset
housing = pd.read_csv('housing.csv')

# show a small amount of DateFrame
def show_data_head():
    return housing.head()

# show shape of the DataFrame
def show_data_shape():
    return housing.shape 

# show information about DataFrame
def show_data_info():
    return housing.info()

# Describe the DataFrame
def show_description_of_data():
    return housing.describe() 

# Split the data into training and testing set
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)

for train_split, test_split in split.split(housing, housing['CHAS']):
    train_set = housing.loc[train_split]
    test_set = housing.loc[test_split]

# train features/labels
train_features = train_set.iloc[:, : 13]
train_labels = train_set['PRICE']

# test features/labels
test_features = test_set.iloc[:, : 13]
test_labels = test_set['PRICE']

# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

train_features = sc.fit_transform(train_features)
test_features = sc.transform(test_features)

# Fit the dataset in Regression Models

# Linear Regressor 
'''
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(train_features, train_labels)

housing_predictions = model.predict(test_features)
'''

# RandomForest Regressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

model.fit(train_features, train_labels)

housing_predictions = model.predict(test_features)

# print(housing_predictions[: 5])
# print(list(test_labels[: 5]))

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(test_labels, housing_predictions)
rmse = np.sqrt(mse)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, train_features, train_labels, scoring = 'neg_mean_squared_error', cv = 10)
rmse_scores = np.sqrt(-scores)

# print(rmse_scores)

accuracy_mean = rmse_scores.mean()
standard_deviation = rmse_scores.std()

from joblib import dump, load
dump(model, 'dragon.joblib')

