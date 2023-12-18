# model_selection.py
def run_model_selection(use_cached_results=True):  # function to choose runtime of model selection criteria
    import numpy as np
    import pandas as pd
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    import utils as utils
    from src.optimal_xgboost import optimal_xgboost
    from src.optimal_adaboost import optimal_adaboost
    from src.optimal_knn import optimal_knn
    from src.optimal_forest import optimal_forest

    og_data = utils.load_data()
    maf = 'macro_average_f1'  # col name for f1 macro avg score
    # load saved bootstrapped model results
    results_xg = pd.read_pickle('./store/xg_boost_optimal_results.pkl') if use_cached_results else optimal_xgboost()
    results_xg[maf] = results_xg['class_report'].apply(utils.parse_classification_report, target_stat='f1-score')

    results_ada = pd.read_pickle('./store/ada_optimal_results.pkl') if use_cached_results else optimal_adaboost()
    results_ada[maf] = results_ada['class_report'].apply(utils.parse_classification_report, target_stat='f1-score')

    results_knn = pd.read_pickle('./store/knn_optimal_results.pkl') if use_cached_results else optimal_knn()
    results_knn[maf] = results_knn['class_report'].apply(utils.parse_classification_report, target_stat='f1-score')

    results_forest = pd.read_pickle('./store/forest_optimal_results.pkl') if use_cached_results else optimal_forest()
    results_forest[maf] = results_forest['class_report'].apply(utils.parse_classification_report,
                                                               target_stat='f1-score')

    # get equivalent bootstrapped dataframes for non parametric models
    log = utils.initialize_model_pipeline(LogisticRegression())
    results_log = pd.read_pickle('./store/log_reg_results.pkl') if use_cached_results\
        else utils.bootstrap_model(log, og_data, og_data['increase_stock'])
    if not use_cached_results:
        results_log.to_pickle('./store/log_reg_results.pkl')
    results_log[maf] = results_log['class_report'].apply(utils.parse_classification_report, target_stat='f1-score')

    qda = utils.initialize_model_pipeline(QuadraticDiscriminantAnalysis())
    results_qda = pd.read_pickle('./store/qda_results.pkl') if use_cached_results\
        else utils.bootstrap_model(qda, og_data, og_data['increase_stock'])
    if not use_cached_results:
        results_qda.to_pickle('./store/qda_results.pkl')
    results_qda[maf] = results_qda['class_report'].apply(utils.parse_classification_report, target_stat='f1-score')

    lda = utils.initialize_model_pipeline(LinearDiscriminantAnalysis())
    results_lda = pd.read_pickle('./store/lda_results.pkl') if use_cached_results \
        else utils.bootstrap_model(lda, og_data, og_data['increase_stock'])
    if not use_cached_results:
        results_lda.to_pickle('./store/lda_results.pkl')
    results_lda[maf] = results_lda['class_report'].apply(utils.parse_classification_report, target_stat='f1-score')

    # do naive model (always choose low bike demand)
    results_naive = []
    for i in range(1000): # do 1000 instead of 100 because it's a really cheap bootstrap
        r = {}
        x_train, x_test, y_train, y_test = train_test_split(
            og_data.drop('increase_stock', axis=1),
            og_data['increase_stock'],
            test_size=1 / 16, random_state=i
        )
        zeros = np.zeros(len(y_test))
        r['accuracy'] = accuracy_score(y_test, zeros)
        r['optimistic_accuracy'] = accuracy_score(y_train, np.zeros(len(y_train)))
        r[maf] = f1_score(y_test, zeros, average='macro')
        results_naive.append(r)
    results_naive = pd.DataFrame(results_naive)

    print(f'XGB \nEnew = {np.mean(results_xg["accuracy"])}, Enew CI = {utils.get_linear_CI(results_xg["accuracy"])}, '
          f'\nOpt Acc = {np.mean(results_xg["optimistic_accuracy"])}, fma = {np.mean(results_xg[maf])}, '
          f'fmaCI = {utils.get_linear_CI(results_xg[maf])}')

    print(f'ADA\n Enew = {np.mean(results_ada["accuracy"])}, Enew CI = {utils.get_linear_CI(results_ada["accuracy"])}, '
          f'\nOpt Acc = {np.mean(results_ada["optimistic_accuracy"])}, fma = {np.mean(results_ada[maf])}, '
          f'fmaCI = {utils.get_linear_CI(results_ada[maf])}')

    print(f'KNN \nEnew = {np.mean(results_knn["accuracy"])}, Enew CI = {utils.get_linear_CI(results_knn["accuracy"])}, '
          f'\nOpt Acc = {np.mean(results_knn["optimistic_accuracy"])}, fma = {np.mean(results_knn[maf])}, '
          f'fmaCI = {utils.get_linear_CI(results_knn[maf])}')

    print(f'LOG \nEnew = {np.mean(results_log["accuracy"])}, Enew CI = {utils.get_linear_CI(results_log["accuracy"])}, '
          f'\nOpt Acc = {np.mean(results_log["optimistic_accuracy"])}, fma = {np.mean(results_log[maf])}, '
          f'fmaCI = {utils.get_linear_CI(results_log[maf])}')

    print(f'LDA \nEnew = {np.mean(results_lda["accuracy"])}, Enew CI = {utils.get_linear_CI(results_lda["accuracy"])}, '
          f'\nOpt Acc = {np.mean(results_lda["optimistic_accuracy"])}, fma = {np.mean(results_lda[maf])}, '
          f'fmaCI = {utils.get_linear_CI(results_lda[maf])}')

    print(f'QDA \nEnew = {np.mean(results_qda["accuracy"])}, Enew CI = {utils.get_linear_CI(results_qda["accuracy"])}, '
          f'\nOpt Acc = {np.mean(results_qda["optimistic_accuracy"])}, fma = {np.mean(results_qda[maf])}, fmaCI = {utils.get_linear_CI(results_qda[maf])}')

    print(
        f'FRST \nEnew = {np.mean(results_forest["accuracy"])}, Enew CI = {utils.get_linear_CI(results_forest["accuracy"])}, '
        f'\nOpt Acc = {np.mean(results_forest["optimistic_accuracy"])}, fma = {np.mean(results_forest[maf])}, fmaCI = {utils.get_linear_CI(results_forest[maf])}')

    print(
        f'NAIVE \nEnew = {np.mean(results_naive["accuracy"])}, Enew CI = {utils.get_linear_CI(results_naive["accuracy"])}, '
        f'\nOpt Acc = {np.mean(results_naive["optimistic_accuracy"])}, fma = {np.mean(results_naive[maf])}, fmaCI = {utils.get_linear_CI(results_naive[maf])}')


# Run model selection
# WARNING: running with use_cached_results=False will take very long to run
run_model_selection()
