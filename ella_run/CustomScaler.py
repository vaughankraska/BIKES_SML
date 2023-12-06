from sklearn.preprocessing import StandardScaler
from pandas import DataFrame


class CustomScaler(StandardScaler):
    """
    Returns a custom scaler so that we are always working with dataframes

    Returns:
    Class: CustomScaler
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def transform(self, X, y=None):
        transformed = super().transform(X)
        transformed_as_df = DataFrame(transformed, columns=self.feature_names_in_)
        return transformed_as_df
