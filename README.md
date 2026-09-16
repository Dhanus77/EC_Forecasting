# Energy Consumption Forecasting Using Time Series and Weather Features

A machine learning-based time-series forecasting system that predicts household appliance energy consumption one hour ahead using historical consumption patterns, temporal features, and environmental data.

## Project Overview

Energy consumption changes throughout the day based on usage patterns, time-related behavior, and environmental conditions. This project develops a forecasting pipeline to predict future household appliance energy consumption using historical observations from the UCI Appliances Energy Prediction dataset.

The project covers the complete machine learning workflow:

- Data understanding and preprocessing
- Time-series target construction
- Temporal and cyclical feature engineering
- Lag and rolling-window features
- Model training and comparison
- Hyperparameter tuning
- Final model evaluation on unseen test data
- Streamlit-based forecasting application

## Dataset

**Dataset:** UCI Appliances Energy Prediction

The dataset contains:

- 19,735 observations
- 29 original variables
- 10-minute sampling intervals
- Appliance energy consumption measured in Wh
- Indoor temperature and humidity measurements
- Outdoor weather measurements
- Lighting consumption

The forecasting target is defined as appliance energy consumption **one hour ahead (t + 60 minutes)**.

Dataset source:

https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction

## Methodology

### 1. Data Preprocessing

The raw time-series data was:

- Converted to datetime format
- Chronologically sorted
- Checked for missing values and duplicates
- Processed using the original 10-minute sampling frequency
- Checked for redundant variables
- Prepared for one-hour-ahead forecasting

The random variables `rv1` and `rv2` were removed as they were redundant for the forecasting workflow.

### 2. Feature Engineering

The forecasting model uses several categories of features.

**Temporal Features**

- Hour
- Day
- Month
- Day of week
- Weekend indicator
- Week of year

**Cyclical Features**

- Hour sine
- Hour cosine

**Lag Features**

- 10-minute lag
- 20-minute lag
- 30-minute lag
- 1-hour lag
- 2-hour lag
- 1-day lag
- 1-week lag

**Rolling Features**

- 30-minute rolling mean and standard deviation
- 1-hour rolling mean and standard deviation
- 2-hour rolling mean and standard deviation

**Weather Features**

- Outdoor temperature
- Outdoor humidity
- Wind speed
- Visibility
- Atmospheric pressure
- Dew point

**Weather Change Features**

- 1-hour outdoor temperature change
- 1-hour outdoor humidity change

All lag and rolling features are constructed using information available at or before the prediction time to avoid future-data leakage.

## Time-Series Split

The dataset was divided chronologically rather than randomly:

| Dataset | Proportion |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Test | 15% |

The test set was kept untouched during model selection and hyperparameter tuning.

## Models Evaluated

The following approaches were evaluated:

### Baseline Models

- Persistence baseline
- 24-hour lag baseline

### Linear Models

- Linear Regression
- Ridge Regression

### Tree-Based Models

- Random Forest
- Gradient Boosting
- XGBoost

Models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Model Selection

Ridge Regression was tuned using the chronological validation set.

The selected final configuration uses:

```text
Ridge Regression
Alpha = 4000
