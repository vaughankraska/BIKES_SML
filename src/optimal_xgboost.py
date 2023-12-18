# optimal_xgboost.py
def optimal_xgboost():
    from os import cpu_count
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split, GridSearchCV, KFold
    from xgboost import XGBClassifier

    print("Running optimal XGBoost")
    # Load preprocessed dataset from a CSV file
    data = pd.read_csv('./store/final.csv')
    X = data.drop('increase_stock', axis=1)  # Separate features
    Y = data['increase_stock']  # Separate target variable

    # Prepare for model evaluation
    bs_results = []
    # Define hyperparameter grid to optimize certain parameters of XGBoost
    param_grid = {
        'learning_rate': [0.05, .1, .15],
        'n_estimators': [50, 100, 150],
        'max_depth': [4, 5],
        'subsample': [.75],
        'colsample_bytree': [0.75],
        'gamma': [0],
        'min_child_weight': [3],
        'reg_lambda': [0.1]
    }

    # Loop to perform model training and evaluation 100 times
    for i in range(100):
        print(f'{i + 1}/100')
        result = {}
        # Split dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 / 16, random_state=i + 1)
        kf = KFold(n_splits=150, shuffle=True, random_state=123)  # K-Fold cross-validation setup
        m = XGBClassifier()  # Initialize XGBoost Classifier
        # Grid Search for hyperparameter tuning
        search = GridSearchCV(m, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=cpu_count() - 1)
        search.fit(x_train, y_train)  # Fit the model
        # Store various metrics in result dictionary
        result['accuracy'] = accuracy_score(y_test, search.best_estimator_.predict(x_test))
        result['optimistic_accuracy'] = search.best_score_
        result['class_report'] = classification_report(y_test, search.best_estimator_.predict(x_test))
        result['best_params'] = search.best_params_
        result['model'] = search.best_estimator_
        bs_results.append(result)  # Append results to the list

    # Create a new DataFrame
    bs_results = pd.DataFrame(bs_results)
    bs_results.to_pickle('./store/xg_boost_optimal_results.pkl')

    # Calculate mean difference between out-of-sample accuracy and optimistic accuracy
    print(
        f'mew out of sample accuracy - mew optimistic accuracy = {np.mean(bs_results["accuracy"] - bs_results["optimistic_accuracy"])}')
    # Calculate mean accuracy and optimistic accuracy
    bs_results[['accuracy', 'optimistic_accuracy']].mean()

    # Calculate 5th and 95th percentiles for out-of-sample accuracy
    quantile_5 = np.percentile(bs_results['accuracy'], 5, interpolation="linear")
    quantile_95 = np.percentile(bs_results['accuracy'], 95, interpolation="linear")
    print(f'~95% CI on out of sample accuracy [{quantile_5}, {quantile_95}]')

    # Plot histogram of accuracy distribution
    from seaborn import histplot
    histplot(data=bs_results[['accuracy']], kde=True, palette='viridis')
    plt.xlabel('Accuracy distribution')
    # Uncomment below line to save the plot as an image
    # plt.savefig('../figures/xg_boost_accuracy.png', format='png', transparent=True, dpi=300)

    return bs_results
