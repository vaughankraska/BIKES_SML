# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 10:53:50 2023

@author: ANAND
"""

import pandas as pd
from pandas.plotting import scatter_matrix

data = pd.read_csv('../data/training_data.csv')

data_copy = data.copy()

data_copy["month"] = data_copy["month"].astype('category')
data_copy["day_of_week"] = data_copy["day_of_week"].astype('category')
data_copy["hour_of_day"] = data_copy["hour_of_day"].astype('category')
data_copy["holiday"] = data_copy["holiday"].astype('category')
data_copy["weekday"] = data_copy["weekday"].astype('category')
data_copy["increase_stock"] = data_copy["increase_stock"].astype('category')

data_copy = data_copy.drop(columns=["snow"])
'''
Time of day
7-19: morning
otherwise: night
'''
data_copy["daytime"] = data_copy.hour_of_day.apply(lambda h: 1 if h in range(7, 20) else 0).astype('category')

'''
Rushhour
15-19: rush
otherwise: no rush
'''
data_copy["rushhour"] = data_copy.hour_of_day.apply(lambda h: 1 if h in range(15, 20) else 0).astype('category')

'''
Need a good weather variable

Finn inputs pending...
'''