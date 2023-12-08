# weather_score_creation.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load training data 
data = pd.read_csv(r"../data/training_data.csv")
# Remove the 'snow' column as it contains no useful information.
data = data.drop('snow', axis=1) 
# Create a new binary column 'is_high_demand' based on the 'increase_stock' column values.
data['is_high_demand'] = data['increase_stock'].apply(lambda entity: 1 if entity == 'high_bike_demand' else 0)
# Convert 'is_high_demand' to integer type for consistency.
data['is_high_demand'] = data['is_high_demand'].astype('int')

# Train a linear regression model to get the coefficients used in weather_score
scaler = MinMaxScaler() # Initialize a MinMaxScaler for normalizing data.
y = data['is_high_demand'] # Target variable.
# Select relevant weather-related features, dropping non-weather columns.
data_weather = data.drop(
    columns=['is_high_demand', 'increase_stock', 'hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday'])
X = scaler.fit_transform(data_weather) # Normalize the weather data.
# Split the data into training and test sets, with 10% of data as test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=123)

# Create and train the linear regression model.
weather_model = LinearRegression()
weather_model.fit(X_train, y_train)

# Intercept and coefficients are used to create the variable wather_score
print(weather_model.intercept_) # Intercept of the regression line.
print(weather_model.coef_) # Coefficients for each feature in the model.
