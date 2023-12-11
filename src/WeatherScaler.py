# WeatherScaler.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


# Define a custom transformer class named 'WeatherTransformer'
class WeatherTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.equation = LinearRegression()
        self.scaler = StandardScaler()

    # Fit method for the transformer, doesn't learn anything, returns self
    def fit(self, X, y=None):
        self.equation.fit(X[['summertime', 'temp', 'dew',
                             'humidity', 'precip', 'snowdepth',
                             'windspeed', 'cloudcover', 'visibility']],
                          y)
        output_y = self.equation.predict(X)  # weather score unscaled
        self.scaler.fit(X=output_y.reshape(-1, 1))  # scale weather score (stores mean and sigma)
        return self

    # Transform method to apply the weather equation to each row in DataFrame X
    def transform(self, X, y=None):
        # Calculate weather score for each row
        score = self.equation.predict(X)
        # Normalize the weather score
        score = self.scaler.transform(score.reshape(-1, 1))

        # Return the resulting weather score as a DataFrame
        return pd.DataFrame(score, columns=['weather_score'])

    # Method to return the names of the features output by this transformer
    def get_feature_names_out(self, input_features=None):
        return ['score']  # Returns the name of the output feature
