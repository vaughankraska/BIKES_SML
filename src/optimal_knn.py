# optimal_knn.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from os import cpu_count
import pandas as pd

# Load dataset from a CSV file
data = pd.read_csv('../data/final.csv')
X = data.drop('increase_stock', axis=1)  # Separating features
Y = data['increase_stock']  # Separating target variable

# Prepare for model evaluation
bs_results = []
param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Number of neighbors
    'weights': ['uniform', 'distance'],  # Weight function used in prediction
    'metric': ['euclidean', 'manhattan']  # Metric for distance computation
}

# Loop to perform model training and evaluation 100 times
for i in range(100):
    print(f'{i + 1}/100')
    result = {}
    # Split dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/16, random_state=i + 1)
    knn = KNeighborsClassifier()  # Initialize KNN model
    kf = KFold(n_splits=150, shuffle=True)  # K-Fold cross-validation setup
    # Grid Search for hyperparameter tuning
    search = GridSearchCV(knn, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=cpu_count())
    search.fit(x_train, y_train)  # Fit the model
    # Store various metrics in result dictionary
    result['accuracy'] = accuracy_score(y_test, search.best_estimator_.predict(x_test))
    result['optimistic_accuracy'] = search.best_score_
    result['class_report'] = classification_report(y_test, search.best_estimator_.predict(x_test))
    result['best_params'] = search.best_params_
    result['model'] = search.best_estimator_
    bs_results.append(result)  # Append results to the list

# Create a new DataFrame
results_df = pd.read_pickle('../data/knn_optimal_results.pkl')  # Load saved results
results_df  # Display the DataFrame

# Analyze the distribution of 'n_neighbors' parameter in the best models
parameter_df = pd.DataFrame(results_df['best_params'].tolist())
parameter_df['n_neighbors'].value_counts()

# Uncomment below line to save the new results to a file
# results_df.to_pickle('../data/knn_optimal_results.pkl')

# Custom utility functions for loading data, initializing models, and cross-validation
from src.utils import load_data, initialize_model_pipeline, cross_validate_model

data = load_data()  # Load data using custom function
# Initialize model pipeline with specific hyperparameters
model = initialize_model_pipeline(KNeighborsClassifier(metric='manhattan', n_neighbors=14, weights='uniform'))

# Perform cross-validation and store the scores
cvs = cross_validate_model(model, X=data, y=data['increase_stock'], n_splits=160)

import numpy as np

np.mean(cvs)  # Calculate the mean of cross-validation scores
