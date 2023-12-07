# Load packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load training data
data = pd.read_csv(r"../data/training_data.csv")

data = data.drop('snow', axis=1) #no information in this column
data['is_high_demand'] = data['increase_stock'].apply(lambda entity: 1 if entity == 'high_bike_demand' else 0)
data['is_high_demand'] = data['is_high_demand'].astype('int')

# Train a linear regression model to get the coefficients used in weather_score
scaler = MinMaxScaler()
y = data['is_high_demand']
data_weather = data.drop(
    columns=['is_high_demand', 'increase_stock', 'hour_of_day', 'day_of_week', 'month', 'holiday', 'weekday'])
X = scaler.fit_transform(data_weather)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=123)

weather_model = LinearRegression()
weather_model.fit(X_train, y_train)

# Intercept and coefficients are used to create the variable wather_score
print(weather_model.intercept_)
print(weather_model.coef_)