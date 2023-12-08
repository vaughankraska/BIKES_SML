# utils.py

# Function to create a weather score from a row of a DataFrame
def create_weather_score(df_row):
    import numpy as np

    # Coefficients for the linear regression model
    b0 = -0.8138192466580598  # Intercept term
    # Coefficients for each feature
    betas = np.array([-5.56042822e-04, 6.31851025e-02, -5.21309866e-02, 6.49687899e-03,
                      -1.10383706e-02, 6.86816015e-03, -5.95565495e-05, 7.78956846e-04,
                      -1.72291318e-03])
    # Extracting the relevant features from the DataFrame row
    exes = np.matrix([df_row['summertime'], df_row['temp'], df_row['dew'],
                      df_row['humidity'], df_row['precip'], df_row['snowdepth'],
                      df_row['windspeed'], df_row['cloudcover'], df_row['visibility']])
    # Ensuring the feature vector is in the correct shape
    if exes.shape[0] != 9:
        exes = exes.T
    # Calculating the linear combination of features and coefficients
    return b0 + np.dot(betas, exes.tolist())

# Function to load and preprocess data
def load_data():
    import pandas as pd

    # Reads the training data into a pandas DataFrame
    data = pd.read_csv('../data/training_data.csv')
    # Creating a copy of the DataFrame
    data = data.copy()
    # Transforming 'increase_stock' column based on a condition
    data['increase_stock'] = data['increase_stock'].apply(lambda entity: 1 if entity == 'high_bike_demand' else 0)
    return data  # Returning the modified DataFrame

# Function to initialize the model pipeline
def initialize_model_pipeline(sklearn_model):
    from sklearn.preprocessing import OneHotEncoder
    from WeatherScaler import WeatherTransformer
    from Generator import Generator
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # Defining a column transformer for preprocessing
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(), ['hour_of_day', 'month']),  # Encoding categorical features
        ('gen', Generator(), ['hour_of_day']),  # Applying custom transformation
        ('bin', 'passthrough', ['weekday', 'summertime']),  # Passing through without changes
        ('weather', WeatherTransformer(), ['summertime', 'temp', 'dew', 'humidity', 'precip',
                                           'snowdepth', 'windspeed', 'cloudcover', 'visibility'])
    ], remainder='drop', sparse_threshold=0)  # Dropping other columns and setting sparse threshold

    # Creating a pipeline with the preprocessing steps and the model
    model_pipline = Pipeline([
        ('pre', pre),  # Preprocessing
        ('model', sklearn_model)  # The machine learning model
    ])

    return model_pipline  # Returning the pipeline

# Function to perform cross-validation on the model pipeline
def cross_validate_model(pipeline, X, y, n_splits=5, scoring='accuracy', cpu_count=1):
    from sklearn.model_selection import cross_val_score, KFold

    # Setting up KFold for cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
    # Performing cross-validation and returning the scores
    cvs = cross_val_score(pipeline, X, y, cv=kf, scoring=scoring, n_jobs=cpu_count)

    return cvs  # Returning the cross-validation scores

# Function to perform cross-validation on the model pipeline
def cross_validate_model(pipeline, X, y, n_splits=5, scoring='accuracy', cpu_count=1):
    from sklearn.model_selection import cross_val_score, KFold

    # Setting up KFold for cross-validation with n_splits, shuffle and fixed random state
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
    # Performing cross-validation using the pipeline, input data, and specified scoring metric; 
    # parallelizing across available CPUs as specified by cpu_count
    cvs = cross_val_score(pipeline, X, y, cv=kf, scoring=scoring, n_jobs=cpu_count)

    return cvs  # Returning the cross-validation scores

# Function to perform bootstrap analysis on the model pipeline
def bootstrap_model(pipeline, X, y, num_bs=100):
    from sklearn.model_selection import train_test_split
    from numpy import ravel
    from sklearn.metrics import accuracy_score, classification_report
    from pandas import DataFrame

    # Conversion of input data into DataFrame and flattening target variable array
    X = DataFrame(X)
    y = ravel(y)
    results = []
    for _ in range(num_bs):
        result = {}
        # Splitting data into training and testing sets for each bootstrap iteration
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1/16, random_state=_)
        # Fitting the model pipeline to the training data
        pipeline.fit(x_train, y_train)

        # Evaluating the model on test data and storing various metrics
        result['accuracy'] = accuracy_score(y_test, pipeline.predict(x_test))
        result['optimistic_accuracy'] = accuracy_score(y_train, pipeline.predict(x_train))
        result['class_report'] = classification_report(y_test, pipeline.predict(x_test))
        result['model'] = pipeline
        results.append(result)

    return DataFrame(results)

# Function to calculate confidence intervals
def get_linear_CI(arraylike):
    from numpy import percentile
    # Calculating the 5th and 95th percentiles for the input array using linear interpolation
    quantile_5 = percentile(arraylike, 5, method="linear")
    quantile_95 = percentile(arraylike, 95, method="linear")
    return [quantile_5, quantile_95]

# Function to parse a classification report into a summary metric
def parse_classification_report(report_str):
    from re import findall
    # Splitting the report string into lines
    lines = report_str.strip().split('\n')

    # Extracting macro-average scores from the second-to-last line
    macro_avg_line = lines[-2]
    macro_avg_scores = findall(r'\d+\.\d+|\d+', macro_avg_line)
    temp = {
        'precision': float(macro_avg_scores[0]),
        'recall': float(macro_avg_scores[1]),
        'f1-score': float(macro_avg_scores[2]),
        'support': int(macro_avg_scores[3])
    }

    return float(macro_avg_scores[2])  # Returning the F1-score as a summary metric