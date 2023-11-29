# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:53:15 2023

@author: ANAND
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV

# Load data, set features and labels and split into training and testing set
data = pd.read_csv('../data/data_copy.csv')
data = data.drop(columns=data.columns[0])
data = data.drop(columns=["holiday", "snowdepth"])

X = data.drop(columns=["increase_stock"])
Y = data["increase_stock"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=123)

# Scale the train and test features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
###############################################################################
# Decision Tree classifier
d_tree = tree.DecisionTreeClassifier(random_state=123)
model = d_tree.fit(X_train_scaled, Y_train)
Y_predict = model.predict(X_test_scaled)
# print(Y_predict)

accuracy = model.score(X_test_scaled, Y_test)

pred_accuracy_score = accuracy_score(Y_test, Y_predict)
report = classification_report(Y_test, Y_predict)

print("Model Score = " + str(accuracy))
print("Prediction Score = " + str(pred_accuracy_score))
print("Report = ", report)
###############################################################################
# Random Forest Classifier
forest = RandomForestClassifier(random_state=123)
forest.fit(X_train_scaled, Y_train)
forest_accuracy = forest.score(X_test_scaled, Y_test)
Y_forest_predict = forest.predict(X_test_scaled)
forest_report = classification_report(Y_test, Y_forest_predict)

print("Model Score = " + str(forest_accuracy))
print("Report = ", forest_report)
###############################################################################
# Get best parameters for Random Forest using Halving Grid Search
# param_grid = {'max_depth': [2, 3, 5, 7, 10, 20, 30, 50, 100, 250, 500, 1000, 1500, 1750, 2000],
#               'min_samples_split': [2, 3, 5, 7, 10, 20, 30, 40, 50, 55, 60],
#               'criterion': ['gini', 'entropy', 'log_loss']}
# best_forest = RandomForestClassifier(random_state=123)
# hgs = HalvingGridSearchCV(best_forest, param_grid, 
#                           resource='n_estimators',
#                           max_resources=30).fit(X_train_scaled, Y_train)
# print("Best Estimator: ", hgs.best_estimator_)
# Best Estimator:  RandomForestClassifier(criterion='entropy', max_depth=7, min_samples_split=5,
#                        n_estimators=27, random_state=123)

best_forest = RandomForestClassifier(n_estimators=27,max_depth=7,
                                     min_samples_split=5,criterion='entropy',random_state=123)
best_forest.fit(X_train_scaled, Y_train)
best_forest_accuracy = best_forest.score(X_test_scaled, Y_test)
Y_best_forest_predict = best_forest.predict(X_test_scaled)
best_forest_report = classification_report(Y_test, Y_best_forest_predict)

print("Model Score = " + str(best_forest_accuracy))
print("Report = \n", best_forest_report)
print("Important features = ", best_forest.feature_importances_)

feature_names = X.columns
importances = best_forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in best_forest.estimators_], axis=0)
forest_importances = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)