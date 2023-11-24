# Washington DC Bike Data Analysis

## Overview
Welcome to the Washington DC Bike Data Analysis project! This repository contains the code and data related to our exploration of bike-related information in Washington DC.

## Project Structure
The repository is organized as follows:

1. `data/`: This directory contains the raw and processed datasets used in the analysis.
    - `raw_data/`: Raw data files go here.
    - `processed_data/`: Processed data files go here.
    - Eventually stick `predictions.csv` in here (final guesses our model gives)

2. `notebooks/`: Jupyter notebooks used for data cleaning, exploration, and analysis.
    - `exploratory_analysis.ipynb`: Notebook for exploratory data analysis.
    - `data_preprocessing.ipynb`: Notebook for data preprocessing.
    - ...

3. `src/`: Source code for the project.
    - We will put our final model in here with finalized proccessing and runnables.
    - `main.py`: Main script for the analysis.
    - `data_processing.py`: Module for data processing functions.
    - ...

4. `environment.yml`: YAML file specifying the project's conda environment.
5. `README.md`: Documentation and instructions for the project.
6. https://www.overleaf.com/project/655db32b849e2e21ba3487c5

## Getting Started
To get started with the project, follow these steps:
**Clone the Repository:**
   ```bash
   git clone https://github.com/vaughankraska/BIKES_SML.git
   ```
## The data
**Features**
   ```
| Feature         | Description                                                                      |
|-----------------|----------------------------------------------------------------------            |
| increase_stock  | Prediction label indicating bike demand: `low_bike_demand` or `high_bike_demand`.|
| hour_of_day     | Hour of the day (0 to 23).                                                       |
| day_of_week     | Day of the week (0 - Monday to 6 - Sunday).                                      |
| month           | Month of the year (0 - January to 12 - December).                                |
| holiday         | Indicates if it's a holiday: `0` (No holiday) or `1` (Holiday).                  |
| weekday         | Indicates if it's a weekday or weekend: `0` (Weekend) or `1` (Weekday).          |
| summertime      | Indicates if it's summertime: `0` (No summertime) or `1` (Summertime).           |
| temp            | Temperature in Celsius degrees.                                                  |
| dew             | Dew point in Celsius degrees.                                                    |
| humidity        | Relative humidity (percentage).                                                  |
| precip          | Precipitation in mm.                                                             |
| snow            | Amount of snow in the last hour in mm.                                           |
| snow_depth      | Accumulated amount of snow in mm.                                                |
| windspeed       | Wind speed in km/h.                                                              |
| cloudcover      | Percentage of the city covered in clouds.                                        |
| visibility      | Distance in km for clear object identification.                                  |
```