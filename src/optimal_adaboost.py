# optimal_adaboost.py

# Importing necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os import cpu_count
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, HalvingGridSearchCV

# Custom utilities for loading data and initializing models
from src.utils import load_data, initialize_model_pipeline, cross_validate_model
from sklearn.metrics import accuracy_score, classification_report

# Reading the dataset
data = pd.read_csv('../data/final.csv')
X = data.drop('increase_stock', axis=1)  # Dropping the target column to isolate features
Y = data['increase_stock']  # Isolating the target column
data.shape  # Getting the shape of the data

# Preparing for model evaluation
bs_results = []
param_grid = {
    'n_estimators': [100, 200],  # Number of trees in AdaBoost
    'learning_rate': [0.01, 0.1],  # Learning rate for AdaBoost
    'estimator__max_depth': [2, 3, 4],  # Max depth of each tree
}

# Loop to perform model training and evaluation 100 times
for i in range(100):
    print(f'{i+1}/100')
    result = {}
    # Splitting the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=1/16, random_state=i+1)
    kf = KFold(n_splits=150, shuffle=False)  # Setting up K-Fold cross-validation
    base_m = DecisionTreeClassifier(random_state=123)  # Base model for AdaBoost
    ada_model = AdaBoostClassifier(estimator=base_m)  # AdaBoost with a decision tree base model
    # Setting up Halving Grid Search for hyperparameter tuning
    search = HalvingGridSearchCV(ada_model, param_grid=param_grid, cv=kf, scoring='accuracy', n_jobs=cpu_count()-1)
    search.fit(x_train, y_train)  # Fitting the model
    # Recording metrics
    result['accuracy'] = accuracy_score(y_test, search.best_estimator_.predict(x_test))
    result['optimistic_accuracy'] = search.best_score_
    result['class_report'] = classification_report(y_test, search.best_estimator_.predict(x_test))
    result['best_params'] = search.best_params_
    result['model'] = search.best_estimator_
    bs_results.append(result)  # Appending results to the list
    
# Converting results to DataFrame for analysis
results_df = pd.DataFrame(bs_results)
# Uncomment below line to save the results to a file
# results_df.to_pickle('../data/ada_optimal_results.pkl')
results_df

# Calculating mean difference between out of sample and optimistic accuracy
print(f'mew out of sample accuracy - mew optimistic accuracy = {np.mean(results_df["accuracy"] - results_df["optimistic_accuracy"])}')
# Calculating the mean accuracy and optimistic accuracy
results_df[['accuracy','optimistic_accuracy']].mean()

# Calculating the 5th and 95th percentiles for out-of-sample accuracy
quantile_5 = np.percentile(results_df['accuracy'], 5, interpolation="linear")
quantile_95 = np.percentile(results_df['accuracy'], 95, interpolation="linear")
print(f'~95% CI on out of sample accuracy [{quantile_5}, {quantile_95}]')

# Plotting histogram for accuracy comparison
from seaborn import histplot
import matplotlib.pyplot as plt
histplot(data=results_df[['accuracy', 'optimistic_accuracy']], kde=True, palette='viridis')
plt.title('Density: Out of Sample Accuracy vs. Optimistic Accuracy')

# Analyzing the optimal results of hyperparameters
hyper_params_df = pd.DataFrame(results_df['best_params'].tolist())
hyper_params_df.describe()  # Describing the DataFrame for statistical analysis
