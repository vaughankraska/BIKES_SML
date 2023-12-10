# final_model.py

import numpy as np
import pandas as pd

import utils as utils

# XGBoost is the selected final model
# load saved bootstrapped model results
results_xg = pd.read_pickle('../data/xg_boost_optimal_results.pkl')
# extract extra stats from classification report
single_class_report = results_xg['class_report'].loc[0]
keys = utils.parse_classification_report(single_class_report,
                                         target_row='macro_avg',
                                         extract_dict=True).keys()
# add other classification stats as columns in the df
for key in keys:
    get_value = lambda x, tr: utils.parse_classification_report(x, target_row=tr, extract_dict=True).get(key)
    results_xg[key + '_macro'] = results_xg['class_report'].apply(get_value, tr='macro_avg')
    results_xg[key + '_0'] = results_xg['class_report'].apply(get_value, tr='zero')
    results_xg[key + '_1'] = results_xg['class_report'].apply(get_value, tr='one')
print(single_class_report)
print(results_xg.drop(['optimistic_accuracy', 'class_report', 'best_params', 'model'], axis=1).columns.tolist())
# show more detailed report of
print('########SUMMARY')
print('Low bike demand (0):')
print(f'precision: mew={np.round(np.mean(results_xg["precision_0"]),3)}, '
      f'{utils.get_linear_CI(results_xg["precision_0"])}')
print(f'recall: mew={np.round(np.mean(results_xg["recall_0"]),3)}, '
      f'{utils.get_linear_CI(results_xg["recall_0"])}')
print(f'F1: mew={np.round(np.mean(results_xg["f1-score_0"]),3)}, '
      f'{utils.get_linear_CI(results_xg["f1-score_0"])}')
print(f'support: mew={np.round(np.mean(results_xg["support_0"]),3)}, '
      f'{utils.get_linear_CI(results_xg["support_0"])}')
print('\nHigh Bike Demand (1)')
print(f'precision: mew={np.round(np.mean(results_xg["precision_1"]),3)}, '
      f'{utils.get_linear_CI(results_xg["precision_1"])}')
print(f'recall: mew={np.round(np.mean(results_xg["recall_1"]),3)}, '
      f'{utils.get_linear_CI(results_xg["recall_1"])}')
print(f'F1: mew={np.round(np.mean(results_xg["f1-score_1"]),3)}, '
      f'{utils.get_linear_CI(results_xg["f1-score_1"])}')
print(f'support: mew={np.round(np.mean(results_xg["support_1"]),3)}, '
      f'{utils.get_linear_CI(results_xg["support_1"])}')
print(f'\nAccuracy: {np.round(np.mean(results_xg["accuracy"]), 3)}, '
      f'{utils.get_linear_CI(results_xg["accuracy"])}')
print('\nMacro Avg:')
print(f'precision: mew={np.round(np.mean(results_xg["precision_macro"]),3)}, '
      f'{utils.get_linear_CI(results_xg["precision_macro"])}')
print(f'recall: mew={np.round(np.mean(results_xg["recall_macro"]),3)}, '
      f'{utils.get_linear_CI(results_xg["recall_macro"])}')
print(f'F1: mew={np.round(np.mean(results_xg["f1-score_macro"]),3)}, '
      f'{utils.get_linear_CI(results_xg["f1-score_macro"])}')
print(f'support: mew={np.round(np.mean(results_xg["support_macro"]),3)}, '
      f'{utils.get_linear_CI(results_xg["support_macro"])}')


