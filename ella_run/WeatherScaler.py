import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WeatherTransformer(BaseEstimator, TransformerMixin):

    # def __init__(self):
    #     self.scaler = CustomScaler()

    def fit(self, X, y=None):
        return self

    def _weather_equation(self, X):
        return (-0.8138192466580598 + -5.56042822e-04 * X['summertime'] +
                6.31851025e-02 * X['temp'] + -5.21309866e-02 * X['dew'] +
                6.49687899e-03 * X['humidity'] + -1.10383706e-02 * X['precip'] +
                6.86816015e-03 * X['snowdepth'] + -5.95565495e-05 * X['windspeed'] +
                7.78956846e-04 * X['cloudcover'] + -1.72291318e-03 * X['visibility'])

    def transform(self, X, y=None):

        X['weather_score'] = X.apply(self._weather_equation, axis=1)
        X['weather_score'] = (X['weather_score'] - 0.18) / 0.184322

        return pd.DataFrame(X['weather_score'])

    def get_feature_names_out(self, input_features=None):
        return ['score']
