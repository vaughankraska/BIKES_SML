# optimal_xgboost.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import cpu_count
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from src.utils import load_data, initialize_model_pipeline, cross_validate_model
from sklearn.metrics import accuracy_score, classification_report

# Load dataset from a CSV file
data = pd.read_csv('../data/final.csv')
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
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/16, random_state=i + 1)
    kf = KFold(n_splits=150, shuffle=True)  # K-Fold cross-validation setup
    m = XGBClassifier()  # Initialize XGBoost Classifier
    # Grid Search for hyperparameter tuning
    search = GridSearchCV(m, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=cpu_count() - 1, verbose=2)
    search.fit(x_train, y_train)  # Fit the model
    # Store various metrics in result dictionary
    result['accuracy'] = accuracy_score(y_test, search.best_estimator_.predict(x_test))
    result['optimistic_accuracy'] = search.best_score_
    result['class_report'] = classification_report(y_test, search.best_estimator_.predict(x_test))
    result['best_params'] = search.best_params_
    result['model'] = search.best_estimator_
    bs_results.append(result)  # Append results to the list

# Create a new DataFrame
bs_results = pd.read_pickle('../data/xg_boost_optimal_results.pkl')  # Load saved results

# Calculate mean difference between out-of-sample accuracy and optimistic accuracy
print(f'mew out of sample accuracy - mew optimistic accuracy = {np.mean(bs_results["accuracy"] - bs_results["optimistic_accuracy"])}')
# Calculate mean accuracy and optimistic accuracy
bs_results[['accuracy','optimistic_accuracy']].mean()

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

# Plot histogram of difference between actual and optimistic accuracy
bs_results['accuracy_diff'] = bs_results['accuracy'] - bs_results['optimistic_accuracy']
histplot(data=bs_results['accuracy_diff'], palette='viridis')

# Analyze and describe hyperparameters of the best models
hyper_params_df = pd.DataFrame(bs_results['best_params'].tolist())
hyper_params_df.describe()  # Statistical summary of hyperparameters

# Count occurrences of each 'max_depth' value
hyper_params_df['max_depth'].value_counts()

# Count occurrences of each 'learning_rate' value
hyper_params_df['learning_rate'].value_counts()

# Count occurrences of each 'n_estimators' value
hyper_params_df['n_estimators'].value_counts()

# Analyze correlation of numerical hyperparameters with accuracy difference
hyper_params_df['accuracy'] = bs_results['accuracy']
hyper_params_df['accuracy_diff'] = bs_results['accuracy_diff']
hyper_params_df.select_dtypes(include='number').corr()['accuracy_diff']

# Uncomment below line to save the new results to a file
# bs_results.to_pickle('../xg_boost_optimal_results.pkl')
