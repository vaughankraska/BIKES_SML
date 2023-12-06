# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:34:45 2023

@author: ANAND
"""
import numpy as np
import sys
sys.path.append("../src")
from utils import initialize_model_pipeline, load_data, cross_validate_model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


model = initialize_model_pipeline(RandomForestClassifier(n_estimators=27,max_depth=7, 
                                                         min_samples_split=5,criterion='entropy',
                                                         random_state=123))

data = load_data()
model.fit(data.drop('increase_stock',axis=1), data['increase_stock'])
print(model.score(X=data, y=data['increase_stock']))

leave_out_one = cross_validate_model(model, data, data['increase_stock'],n_splits=len(data))
print("Leave out one score = ", np.mean(leave_out_one))
###############################################################################
# Best Adaboost classifier
boost_model = initialize_model_pipeline(AdaBoostClassifier(random_state=123, n_estimators=20, learning_rate=1))
boost_model.fit(data.drop('increase_stock',axis=1), data['increase_stock'])
print(boost_model.score(X=data, y=data['increase_stock']))

leave_out_one_boost = cross_validate_model(boost_model, data, data['increase_stock'],n_splits=len(data))
print("Leave out one score = ", np.mean(leave_out_one_boost))