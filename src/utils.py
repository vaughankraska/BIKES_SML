def create_weather_score(df_row):
    import numpy as np

    b0 = -0.8138192466580598
    betas = np.array([-5.56042822e-04, 6.31851025e-02, -5.21309866e-02, 6.49687899e-03,
                      -1.10383706e-02, 6.86816015e-03, -5.95565495e-05, 7.78956846e-04,
                      -1.72291318e-03])
    exes = np.matrix([df_row['summertime'], df_row['temp'], df_row['dew'],
                      df_row['humidity'], df_row['precip'], df_row['snowdepth'],
                      df_row['windspeed'], df_row['cloudcover'], df_row['visibility']])
    if exes.shape[0] != 9:
        exes = exes.T
    return b0 + np.dot(betas, exes.tolist())


def load_data():
    """
   Reads original data (tranining.csv) into pandas dataframe and returns it. Welp, sorry I couldn't transform the Y
   variable (increase stock = 1 for is_high_demand).

   Returns:
   a copy of training_data.csv
   """
    import pandas as pd
    data = pd.read_csv('../data/training_data.csv')
    data = data.copy()
    data['increase_stock'] = data['increase_stock'].apply(lambda entity: 1 if entity == 'high_bike_demand' else 0)
    return data


def initialize_model_pipeline(sklearn_model):
    """
   Creates a unified processing pipeline for our data. All you have to do is tell what model you would like to
   implement, and it handles all the preprocessing when fit is called. New data points also get processed in the
   same manner.

   Parameters:
       - sklearn_model: The model you wish to implement the training on

   Returns:
   Pipeline for model
   """
    from sklearn.preprocessing import OneHotEncoder
    from WeatherScaler import WeatherTransformer
    from Generator import Generator
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    pre = ColumnTransformer([
        ('cat', OneHotEncoder(), ['hour_of_day', 'month']),
        ('gen', Generator(), ['hour_of_day']),
        ('bin', 'passthrough', ['weekday', 'summertime']),
        ('weather', WeatherTransformer(), ['summertime', 'temp', 'dew', 'humidity', 'precip',
                                           'snowdepth', 'windspeed', 'cloudcover', 'visibility'])
    ], remainder='drop', sparse_threshold=0)

    model_pipline = Pipeline([
        ('pre', pre),
        ('model', sklearn_model)
    ])

    return model_pipline


def cross_validate_model(pipeline, X, y, n_splits=5, scoring='accuracy', cpu_count=1):
    from sklearn.model_selection import cross_val_score, KFold

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
    cvs = cross_val_score(pipeline, X, y, cv=kf, scoring=scoring, n_jobs=cpu_count)

    return cvs


def bootstrap_model(pipeline, X, y, num_bs=100):
    from sklearn.model_selection import train_test_split
    from pandas import DataFrame

    X = DataFrame(X)
    y = DataFrame(y)
    results = []
    for _ in range(num_bs):
        result = {}
        X_bs = X.sample(frac=1, replace=True, random_state=_)
        y_bs = y.loc[X_bs.index]
        pipeline.fit(X_bs, y_bs)

        coefs = pipeline.coef_.tolist()
        coefs.append(pipeline.intercept_)
        result['score'] = pipeline.score()
        result['coefs'] = coefs
        result['labels'] = pipeline.get_feature_names_out()
        result['model'] = pipeline
        results.append(result)

    return results

#################################################################################################

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

# Function to initialize a model pipeline
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

# Function to perform cross-validation on a model pipeline
def cross_validate_model(pipeline, X, y, n_splits=5, scoring='accuracy', cpu_count=1):
    from sklearn.model_selection import cross_val_score, KFold  

    # Setting up KFold for cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)
    # Performing cross-validation and returning the scores
    cvs = cross_val_score(pipeline, X, y, cv=kf, scoring=scoring, n_jobs=cpu_count)

    return cvs  # Returning the cross-validation scores

# Function to perform bootstrap analysis on a model pipeline
def bootstrap_model(pipeline, X, y, num_bs=100):
    from sklearn.model_selection import train_test_split  # For splitting data
    from pandas import DataFrame  # For DataFrame handling

    X = DataFrame(X)  # Ensuring X is a DataFrame
    y = DataFrame(y)  # Ensuring y is a DataFrame
    results = []  # List to store results of each bootstrap iteration
    for _ in range(num_bs):
        result = {}  # Dictionary to store results of current iteration
        # Sampling with replacement to create a bootstrap sample
        X_bs = X.sample(frac=1, replace=True, random_state=_)
        y_bs = y.loc[X_bs.index]
        pipeline.fit(X_bs, y

