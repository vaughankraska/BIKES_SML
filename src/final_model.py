# final_model.py

import numpy as np
import pandas as pd
import utils as utils

# XGBoost is the selected final model
# load saved bootstrapped model results
results_xg = pd.read_pickle('../data/xg_boost_optimal_results.pkl')
# extract extra stats from classification report
single_class_report = results_xg['class_report'].loc[3]
keys = utils.parse_classification_report(single_class_report,
                                         target_row='macro_avg',
                                         extract_dict=True).keys()
# add other classification stats as columns in the df
for key in keys:
    get_value = lambda x, tr: utils.parse_classification_report(x, target_row=tr, extract_dict=True).get(key)
    results_xg[key + '_macro'] = results_xg['class_report'].apply(get_value, tr='macro_avg')
    results_xg[key + '_0'] = results_xg['class_report'].apply(get_value, tr='zero')
    results_xg[key + '_1'] = results_xg['class_report'].apply(get_value, tr='one')
    results_xg[key + '_weighted'] = results_xg['class_report'].apply(get_value, tr='weighted')

print(results_xg.drop(['optimistic_accuracy', 'class_report', 'best_params', 'model'], axis=1).columns.tolist())
# show more detailed report of scores
print('########SUMMARY')
print('Low bike demand (0):')
print(f'precision: mew={np.round(np.mean(results_xg["precision_0"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["precision_0"])}')
print(f'recall: mew={np.round(np.mean(results_xg["recall_0"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["recall_0"])}')
print(f'F1: mew={np.round(np.mean(results_xg["f1-score_0"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["f1-score_0"])}')
print(f'support: mew={np.round(np.mean(results_xg["support_0"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["support_0"])}')
print('\nHigh Bike Demand (1)')
print(f'precision: mew={np.round(np.mean(results_xg["precision_1"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["precision_1"])}')
print(f'recall: mew={np.round(np.mean(results_xg["recall_1"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["recall_1"])}')
print(f'F1: mew={np.round(np.mean(results_xg["f1-score_1"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["f1-score_1"])}')
print(f'support: mew={np.round(np.mean(results_xg["support_1"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["support_1"])}')
print(f'\nAccuracy: {np.round(np.mean(results_xg["accuracy"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["accuracy"])}')
print('\nMacro Avg:')
print(f'precision: mew={np.round(np.mean(results_xg["precision_macro"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["precision_macro"])}')
print(f'recall: mew={np.round(np.mean(results_xg["recall_macro"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["recall_macro"])}')
print(f'F1: mew={np.round(np.mean(results_xg["f1-score_macro"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["f1-score_macro"])}')
print(f'support: mew={np.round(np.mean(results_xg["support_macro"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["support_macro"])}')
print('\nWeighted:')
print(f'precision: mew={np.round(np.mean(results_xg["precision_weighted"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["precision_weighted"])}')
print(f'recall: mew={np.round(np.mean(results_xg["recall_weighted"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["recall_weighted"])}')
print(f'F1: mew={np.round(np.mean(results_xg["f1-score_weighted"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["f1-score_weighted"])}')
print(f'support: mew={np.round(np.mean(results_xg["support_weighted"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["support_weighted"])}')

###############################################################
# now run final optimization of model on all data
# (instead of on subset to do out of sample testing bootstrap)
from pickle import dump
from os import cpu_count
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import classification_report

# Load OG data with increase_stock as binary
data = utils.load_data()
# Init model pipeline with xg boost as final step
optimized_model = utils.initialize_model_pipeline(XGBClassifier())
# Define our Kfolds process to be used
kf = KFold(n_splits=160, shuffle=True, random_state=123)  # use 160 instead of 150 since we have all data
# Define our hyperparameter window we are allowing to optimize on
param_grid = {
    'model__subsample': [.75],
    'model__colsample_bytree': [.75],
    'model__gamma': [0],
    'model__min_child_weight': [3],
    'model__reg_lambda': [.1],
    'model__learning_rate': [0.05, .1, .15],
    'model__n_estimators': [50, 100, 150],
    'model__max_depth': [4, 5],
}

# Define search grid
search = GridSearchCV(optimized_model, param_grid=param_grid, cv=kf,
                      scoring='accuracy', n_jobs=cpu_count() - 1)
# Run grid search
search.fit(X=data, y=data['increase_stock'])
# Print best params and score
print(f'Best Score (optimistic) = {search.best_score_}')
best_params_ = search.best_params_
print(f'Best Params = {best_params_}')

# Train final model on all data using optimal params
final_model = utils.initialize_model_pipeline(
    XGBClassifier(**best_params_)
)
final_model.fit(data, y=data['increase_stock'])
print('Optimistic Classification report of final model')
print(classification_report(data['increase_stock'], final_model.predict(data)))

# Save trained final model to pickle in order to call fit on new data
with open('./final_model.pkl', 'wb') as file:
    dump(final_model, file)