import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class Generator(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["daytime"] = X.hour_of_day.apply(lambda h: 1 if h in range(7, 20) else 0)
        X["rushhour"] = X.hour_of_day.apply(lambda h: 1 if h in range(15, 19) else 0)

        return pd.DataFrame(X[['daytime', 'rushhour']],columns=['daytime','rushhour'])

    def get_feature_names_out(self, input_features=None):
        return ['daytime', 'rushhour']