# optimal_forest.py
def optimal_forest():
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from os import cpu_count
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, KFold, HalvingGridSearchCV
    from sklearn.metrics import accuracy_score, classification_report

    # Load the dataset
    data = pd.read_csv('./store/final.csv')
    X = data.drop('increase_stock', axis=1)  # Separate features
    Y = data['increase_stock']  # Separate target variable

    # Prepare for model evaluation
    bs_results = []
    param_grid = {
        'max_depth': [7],  # Depth of the trees in the forest
        'min_samples_split': [3, 5, 8],  # Minimum number of samples required to split an internal node
        'criterion': ['gini', 'entropy']  # Function to measure the quality of a split
    }

    # Loop to perform model training and evaluation 100 times
    for i in range(100):
        print(f'{i + 1}/100')
        result = {}
        # Split dataset into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1 / 16, random_state=i + 1)

        kf = KFold(n_splits=150, shuffle=False)  # K-Fold cross-validation setup
        model = RandomForestClassifier(100)  # Initialize RandomForest model
        # Halving Grid Search for hyperparameter tuning
        search = HalvingGridSearchCV(model, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=cpu_count() - 1)
        search.fit(x_train, y_train)  # Fit the model
        # Store various metrics in result dictionary
        result['accuracy'] = accuracy_score(y_test, search.best_estimator_.predict(x_test))
        result['optimistic_accuracy'] = search.best_score_
        result['class_report'] = classification_report(y_test, search.best_estimator_.predict(x_test))
        result['best_params'] = search.best_params_
        result['model'] = search.best_estimator_
        bs_results.append(result)  # Append results to the list

    # Convert results into a DataFrame for analysis
    results_df = pd.DataFrame(bs_results)
    results_df.to_pickle('./store/forest_optimal_results.pkl')  # Save results to a file

    # Calculate mean difference between out-of-sample accuracy and optimistic accuracy
    print(f'mew out of sample accuracy - mew optimistic accuracy = '
          f'{np.mean(results_df["accuracy"] - results_df["optimistic_accuracy"])}')

    # Calculate 5th and 95th percentiles for out-of-sample accuracy
    quantile_5 = np.percentile(results_df['accuracy'], 5, interpolation="linear")
    quantile_95 = np.percentile(results_df['accuracy'], 95, interpolation="linear")
    print(f'~95% CI on out of sample accuracy [{quantile_5}, {quantile_95}]')

    # Plotting histogram for accuracy comparison
    from seaborn import histplot
    histplot(data=results_df[['accuracy', 'optimistic_accuracy']], kde=True, palette='viridis')
    plt.title('Density: Out of Sample Accuracy vs. Optimistic Accuracy')

    return results_df
