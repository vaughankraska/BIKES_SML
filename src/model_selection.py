import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

import utils as utils

OG_data = utils.load_data()
maf = 'macro_average_f1'  # col name for f1 macro avg score
# load saved bootstrapped model results
results_xg = pd.read_pickle('../data/xg_boost_optimal_results.pkl')
results_xg[maf] = results_xg['class_report'].apply(utils.parse_classification_report, target_stat='f1-score')

results_ada = pd.read_pickle('../data/ada_optimal_results.pkl')
results_ada[maf] = results_ada['class_report'].apply(utils.parse_classification_report, target_stat='f1-score')

results_knn = pd.read_pickle('../data/knn_optimal_results.pkl')
results_knn[maf] = results_knn['class_report'].apply(utils.parse_classification_report, target_stat='f1-score')

results_forest = pd.read_pickle('../data/forest_optimal_results.pkl')
results_forest[maf] = results_forest['class_report'].apply(utils.parse_classification_report, target_stat='f1-score')

# get equivalent bootstrapped dataframes for non parametric models
log = utils.initialize_model_pipeline(LogisticRegression())
results_log = utils.bootstrap_model(log, OG_data, OG_data['increase_stock'])
results_log[maf] = results_log['class_report'].apply(utils.parse_classification_report, target_stat='f1-score')

qda = utils.initialize_model_pipeline(QuadraticDiscriminantAnalysis())
results_qda = utils.bootstrap_model(qda, OG_data, OG_data['increase_stock'])
results_qda[maf] = results_qda['class_report'].apply(utils.parse_classification_report, target_stat='f1-score')

lda = utils.initialize_model_pipeline(LinearDiscriminantAnalysis())
results_lda = utils.bootstrap_model(lda, OG_data, OG_data['increase_stock'])
results_lda[maf] = results_lda['class_report'].apply(utils.parse_classification_report, target_stat='f1-score')

# do naive model (always choose low bike demand)
results_naive = []
for i in range(1000):
    r = {}
    x_train, x_test, y_train, y_test = train_test_split(
        OG_data.drop('increase_stock', axis=1),
        OG_data['increase_stock'],
        test_size=1 / 16, random_state=i
    )
    zeros = np.zeros(len(y_test))
    r['accuracy'] = accuracy_score(y_test, zeros)
    r['optimistic_accuracy'] = accuracy_score(y_train, np.zeros(len(y_train)))
    r[maf] = f1_score(y_test, zeros, average='macro')
    results_naive.append(r)
results_naive = pd.DataFrame(results_naive)

print(f'XGB \nEnew = {np.mean(results_xg["accuracy"])}, Enew CI = {utils.get_linear_CI(results_xg["accuracy"])}, '
      f'\nOpt Acc = {np.mean(results_xg["optimistic_accuracy"])}, fma = {np.mean(results_xg[maf])}, fmaCI = {utils.get_linear_CI(results_xg[maf])}')

print(f'ADA\n Enew = {np.mean(results_ada["accuracy"])}, Enew CI = {utils.get_linear_CI(results_ada["accuracy"])}, '
      f'\nOpt Acc = {np.mean(results_ada["optimistic_accuracy"])}, fma = {np.mean(results_ada[maf])}, fmaCI = {utils.get_linear_CI(results_ada[maf])}')

print(f'KNN \nEnew = {np.mean(results_knn["accuracy"])}, Enew CI = {utils.get_linear_CI(results_knn["accuracy"])}, '
      f'\nOpt Acc = {np.mean(results_knn["optimistic_accuracy"])}, fma = {np.mean(results_knn[maf])}, fmaCI = {utils.get_linear_CI(results_knn[maf])}')

print(f'LOG \nEnew = {np.mean(results_log["accuracy"])}, Enew CI = {utils.get_linear_CI(results_log["accuracy"])}, '
      f'\nOpt Acc = {np.mean(results_log["optimistic_accuracy"])}, fma = {np.mean(results_log[maf])}, fmaCI = {utils.get_linear_CI(results_log[maf])}')

print(f'LDA \nEnew = {np.mean(results_lda["accuracy"])}, Enew CI = {utils.get_linear_CI(results_lda["accuracy"])}, '
      f'\nOpt Acc = {np.mean(results_lda["optimistic_accuracy"])}, fma = {np.mean(results_lda[maf])}, fmaCI = {utils.get_linear_CI(results_lda[maf])}')

print(f'QDA \nEnew = {np.mean(results_qda["accuracy"])}, Enew CI = {utils.get_linear_CI(results_qda["accuracy"])}, '
      f'\nOpt Acc = {np.mean(results_qda["optimistic_accuracy"])}, fma = {np.mean(results_qda[maf])}, fmaCI = {utils.get_linear_CI(results_qda[maf])}')

print(
    f'FRST \nEnew = {np.mean(results_forest["accuracy"])}, Enew CI = {utils.get_linear_CI(results_forest["accuracy"])}, '
    f'\nOpt Acc = {np.mean(results_forest["optimistic_accuracy"])}, fma = {np.mean(results_forest[maf])}, fmaCI = {utils.get_linear_CI(results_forest[maf])}')

print(
    f'NAIVE \nEnew = {np.mean(results_naive["accuracy"])}, Enew CI = {utils.get_linear_CI(results_naive["accuracy"])}, '
    f'\nOpt Acc = {np.mean(results_naive["optimistic_accuracy"])}, fma = {np.mean(results_naive[maf])}, fmaCI = {utils.get_linear_CI(results_naive[maf])}')
