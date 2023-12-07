from sklearn.preprocessing import StandardScaler
from pandas import DataFrame


# Defining a custom class CustomScaler, which inherits from StandardScaler
class CustomScaler(StandardScaler):
    """
    A custom scaler class that extends the StandardScaler from scikit-learn.
    The main purpose of this class is to ensure that the output of the transformation
    is always a pandas DataFrame.

    Returns:
        Class: CustomScaler
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize the StandardScaler with any additional arguments

    def transform(self, X, y=None):
        transformed = super().transform(X)  # Applying the standard scaling transformation
        # Converting the numpy array result back into a pandas DataFrame
        # and ensuring the column names are retained
        transformed_as_df = DataFrame(transformed, columns=self.feature_names_in_)
        return transformed_as_df  # Returning the DataFrame

