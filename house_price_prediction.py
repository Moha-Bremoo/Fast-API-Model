import joblib 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

"""Importing the Boston House Price Dataset"""

house_price_dataset = sklearn.datasets.fetch_california_housing()

# Loading the dataset to a pandas dataframe
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)

# add the target column to the dataframe
house_price_dataframe['price'] = house_price_dataset.target

"""Splitting the data and target"""

X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']

"""Splitting the data into training data and test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

"""Model Training

XGBoost Regressor
"""

# load the model
model = XGBRegressor()

#training the model with X_train
model.fit(X_train, Y_train)

joblib.dump(model, "our_model_Task2.joblib")

"""Evaluation

Prediction on training data
"""

# accuracy for prediction on training data
# training_data_prediction = model.predict(X_train)
# 
# R Squared Error
# score_1 = metrics.r2_score(Y_train, training_data_prediction)
# 
# Mean Absolute Error
# score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)
# 
# """Prediction on test data"""
# 
# accuracy for prediction on test data
# test_data_prediction = model.predict(X_test)
# 
# R Squared Error
# score_1 = metrics.r2_score(Y_test, test_data_prediction)
# 
# Mean Absolute Error
# score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)
# 
# print('R Sqaured Error:', score_1)
# print('Mean Absolute Error:', score_2)

