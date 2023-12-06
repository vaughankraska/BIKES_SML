# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:11:53 2023

@author: ANAND
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV

data = pd.read_csv('../data/all_interactions.csv')
data = data.drop(columns=data.columns[0])

X = data.drop(columns=["is_high_demand"])
Y = data["is_high_demand"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

# Scale the train and test features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
###############################################################################
# Bagging classifier
bag = BaggingClassifier(random_state=123, n_estimators=20)
bag.fit(X_train_scaled, Y_train)
Y_predict = bag.predict(X_test_scaled)
# print(Y_predict)

accuracy = bag.score(X_test_scaled, Y_test)

pred_accuracy_score = accuracy_score(Y_test, Y_predict)
report = classification_report(Y_test, Y_predict)

print('Bagging Classifier')
print("Model Score = " + str(accuracy))
print("Prediction Score = " + str(pred_accuracy_score))
print("Report = ", report)
###############################################################################
# Adaboost classifier
boost = AdaBoostClassifier(random_state=123, n_estimators=100)
boost.fit(X_train_scaled, Y_train)
Y_predict = boost.predict(X_test_scaled)
# print(Y_predict)

accuracy = boost.score(X_test_scaled, Y_test)

pred_accuracy_score = accuracy_score(Y_test, Y_predict)
report = classification_report(Y_test, Y_predict)

print('Adaboost Classifier')
print("Model Score = " + str(accuracy))
print("Prediction Score = " + str(pred_accuracy_score))
print("Report = ", report)
###############################################################################
# Get best parameters for Adaboost using Halving Grid Search
# param_grid = {'learning_rate': [1, 2, 5, 10, 25]}
# best_boost = AdaBoostClassifier(random_state=123)
# hgs = HalvingGridSearchCV(best_boost, param_grid, 
#                           resource='n_estimators',
#                           max_resources=1000).fit(X_train_scaled, Y_train)
# print("Best Estimator: ", hgs.best_estimator_)
###############################################################################
# Best Adaboost classifier
boost = AdaBoostClassifier(random_state=123, n_estimators=999, learning_rate=1)
boost.fit(X_train_scaled, Y_train)
Y_predict = boost.predict(X_test_scaled)
# print(Y_predict)

accuracy = boost.score(X_test_scaled, Y_test)

pred_accuracy_score = accuracy_score(Y_test, Y_predict)
report = classification_report(Y_test, Y_predict)

print('Best Adaboost Classifier')
print("Model Score = " + str(accuracy))
print("Prediction Score = " + str(pred_accuracy_score))
print("Report = ", report)
