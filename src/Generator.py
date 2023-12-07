# Generator.py

import pandas as pd  

from sklearn.base import BaseEstimator, TransformerMixin  

# Defining a custom transformer class named 'Generator', inheriting from scikit-learn's BaseEstimator and TransformerMixin
class Generator(BaseEstimator, TransformerMixin):

    # The fit method for the transformer, which doesn't learn anything so simply returns self
    def fit(self, X, y=None):
        return self

    # The transform method that actually processes the data
    def transform(self, X, y=None):
        # Creating a new column 'daytime' in the DataFrame X
        # It sets to 1 during daytime hours (7 to 19) and 0 otherwise
        X["daytime"] = X.hour_of_day.apply(lambda h: 1 if h in range(7, 20) else 0)

        # Creating a new column 'rushhour' in the DataFrame X
        # It sets to 1 during rush hour (15 to 18) and 0 otherwise
        X["rushhour"] = X.hour_of_day.apply(lambda h: 1 if h in range(15, 19) else 0)

        # Returning a new DataFrame with only 'daytime' and 'rushhour' columns
        return pd.DataFrame(X[['daytime', 'rushhour']],columns=['daytime','rushhour'])

    # Method to return the names of the features output by this transformer
    def get_feature_names_out(self, input_features=None):
        return ['daytime', 'rushhour']
