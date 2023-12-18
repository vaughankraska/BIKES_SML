# plots.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

data = pd.read_csv('./store/training_data.csv')

# create bar plot of proportion of high demand by month
data_time = data[["month", "hour_of_day", "day_of_week", "increase_stock"]]
df_month = data_time.groupby(['month', "increase_stock"]).count().reset_index().drop(columns=["day_of_week"])
df_help = data_time["month"].value_counts().reset_index()
df_month = pd.merge(df_month, df_help, left_on=df_month["month"], right_on=df_help["month"], how="inner").drop(
    columns=["month_y", "key_0"])
df_month = df_month.rename(columns={"hour_of_day": "count_stock", "count": "count_month", "month_x": "month"})
df_month["perc_stock"] = df_month["count_stock"] / df_month["count_month"]
_ = sns.barplot(data=df_month, x="month", y="perc_stock", hue="increase_stock", palette='viridis')

# Create bar plot of percentage of high/low bike demand hour of the day
df_hour = data_time.groupby(['hour_of_day', "increase_stock"]).count().reset_index().drop(columns=["month"])
df_help = data_time["hour_of_day"].value_counts().reset_index()
df_hour = pd.merge(df_hour, df_help, left_on=df_hour["hour_of_day"], right_on=df_help["hour_of_day"], how="inner").drop(
    columns=["hour_of_day_y", "key_0"])
df_hour = df_hour.rename(columns={"day_of_week": "count_stock", "count": "count_hour", "hour_of_day_x": "hour_of_day"})
df_hour["perc_stock"] = df_hour["count_stock"] / df_hour["count_hour"]
_ = sns.barplot(data=df_hour, x="hour_of_day", y="perc_stock", hue="increase_stock", palette='viridis')

# create corr matrix heat map (reset data load to get weather vars as proper types)
data = pd.read_csv('./store/training_data.csv')
data['increase_stock'] = data['increase_stock'].astype('category')
data['month'] = data['month'].astype('category')
data['hour_of_day'] = data['hour_of_day'].astype('category')
data['holiday'] = data['holiday'].astype('category')
data['weekday'] = data['weekday'].astype('category')
data['summertime'] = data['summertime'].astype('category')
data['day_of_week'] = data['day_of_week'].astype('category')
data = data.drop('snow', axis=1) #no information in this column
data['is_high_demand'] = data['increase_stock'].apply(lambda entity: 1 if entity == 'high_bike_demand' else 0)
data['is_high_demand'] = data['is_high_demand'].astype('int')
data_numerical = data.select_dtypes(include='number')
corr_matrix = data_numerical.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(10, 10), dpi=500)
sns.heatmap(corr_matrix, mask=mask, cmap='viridis', vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

# Create plots of all numerical data
data_weather = data_numerical.drop(['precip', 'visibility', 'snowdepth'], axis=1)
covariates = data_weather.columns[data_weather.columns != 'is_high_demand']
sns.set(style="ticks")
pp = sns.pairplot(data_numerical,
                  vars=covariates,
                  hue='is_high_demand',
                  markers=["o", "s"],
                  plot_kws={"s": 8, 'alpha': .5},
                  palette='viridis')
pp.map_lower(sns.kdeplot, fill=True)
pp.map_diag(sns.histplot, kde=True)

# Create Violin plot of month by temp
data_violin = data[['is_high_demand','temp','humidity','hour_of_day','summertime','month', 'weekday']]
plt.figure(figsize=(16,4))
sns.violinplot(data=data_violin,x='month',y='temp', hue='is_high_demand', palette='viridis', linewidth=.3)

# Create violin plot of hour by temp
plt.figure(figsize=(16,2))
sns.violinplot(data=data_violin,x='hour_of_day',y='temp', hue='is_high_demand', palette='viridis', linewidth=.2)

# Create plot for accuracy dist of
xg_results = pd.read_pickle('./store/xg_boost_optimal_results.pkl')
from seaborn import histplot
histplot(data=xg_results[['accuracy']], kde=True, palette='viridis')
plt.xlabel('Accuracy distribution')

