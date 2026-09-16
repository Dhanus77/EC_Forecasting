# Energy Consumption Forecasting Using Time Series and Weather Features

A machine learning-based time-series forecasting system that predicts household appliance energy consumption **one hour ahead** using historical energy usage, temporal patterns, and environmental features.

## Overview

Energy consumption varies throughout the day based on usage patterns, time, and environmental conditions. This project develops an end-to-end forecasting pipeline using the **UCI Appliances Energy Prediction Dataset** and integrates the trained model into an interactive **Streamlit dashboard**.

### Key Components

- Time-series data preprocessing
- One-hour-ahead forecasting
- Temporal and cyclical feature engineering
- Lag and rolling-window features
- Weather and environmental features
- Chronological train/validation/test split
- Machine learning model comparison
- Ridge Regression hyperparameter tuning
- Streamlit deployment

## Dataset

**UCI Appliances Energy Prediction Dataset**

- 19,735 observations
- 29 original variables
- 10-minute sampling interval
- Appliance energy consumption measured in Wh
- Indoor temperature and humidity measurements
- Outdoor weather measurements
- Lighting consumption

**Forecast Horizon:** 1 hour (6 × 10-minute intervals)

**Source:**  
https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction

## Feature Engineering

The forecasting pipeline uses:

- Temporal features: hour, day, month, weekday, weekend, week of year
- Cyclical features: sine and cosine encoding of hour
- Lag features: 10-minute to 1-week historical lags
- Rolling statistics: 30-minute, 1-hour, and 2-hour windows
- Indoor temperature and humidity
- Outdoor weather conditions
- One-hour weather change features

Lag and rolling features use only information available at or before the prediction time to prevent future-data leakage.

## Model Development

The following models were evaluated:

| Category | Models |
|---|---|
| Baselines | Persistence, 24-Hour Lag |
| Linear | Linear Regression, Ridge Regression |
| Tree-Based | Random Forest, Gradient Boosting, XGBoost |

Evaluation metrics:

- MAE — Mean Absolute Error
- RMSE — Root Mean Squared Error
- R² — Coefficient of Determination

The final Ridge Regression model was tuned using the chronological validation dataset.

**Final configuration:**

```text
Model: Ridge Regression
Alpha: 4000
```

## Final Test Performance

The final model was retrained using the combined training and validation datasets and evaluated on the untouched test dataset.

| Metric | Result |
|---|---:|
| MAE | **39.63 Wh** |
| RMSE | **80.08 Wh** |
| R² | **0.233** |

The model achieved an average absolute forecasting error of approximately **39.63 Wh** on the unseen test period.

## Streamlit Dashboard

The trained forecasting pipeline is integrated into a Streamlit application with four modes:

### LIVE

Designed for integration with real energy sources such as smart meters, smart plugs, IoT sensors, Home Assistant, or energy APIs.

### LIVE SIMULATION

Provides a synthetic real-time demonstration for testing and presentation purposes.

### MANUAL PREDICTION

Allows users to enter energy, date/time, indoor environmental, and outdoor weather values to generate a one-hour-ahead forecast.

### HISTORICAL DATASET

Uses the original UCI dataset to demonstrate the forecasting pipeline and model results.

## Project Structure

```text
EC_Forecasting/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── energydata_complete.csv
│   └── processed/
│
├── models/
│   ├── final_ridge_model.pkl
│   ├── final_scaler.pkl
│   └── model_features.pkl
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   └── 05_machine_learning_models.ipynb
│
├── src/
│   ├── feature_engineering.py
│   └── predict.py
│
└── presentation/
    ├── ENERGY CONSUMPTION FORECASTING USING TIME SERIES AND WEATHER FEATURES.pdf
    ├── Energy_Consumption_Forecasting_Abstract.docx
    ├── Energy_Consumption_Forecasting_Abstract.pdf
    └── model_comparison_validation_mae.png
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Dhanus77/EC_Forecasting.git
cd EC_Forecasting
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Technologies

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Matplotlib**
- **Plotly**
- **Streamlit**
- **Jupyter Notebook**
- **Git & GitHub**

## Key Findings

Recent energy consumption and time-based patterns provided useful information for forecasting future demand. Lag and rolling-window features captured recent consumption behavior and provided historical context for the prediction.

Weather features were included and evaluated. For this dataset and model setup, adding weather produced only a small change in validation performance, while historical consumption and temporal features remained important predictors.

## Future Scope

- Smart-meter and IoT integration
- Automated real-time energy data collection
- Live weather integration
- Longer-term datasets
- Advanced time-series forecasting models
- Deep learning-based forecasting
- Personalized energy forecasting
- Multi-step forecasting
- Energy monitoring and alerts

## Presentation & Documentation

The `presentation/` directory contains the project presentation, abstract, and model comparison visualization.

## Author

**Dhanus D**  
B.Tech — Artificial Intelligence and Machine Learning

## License

This project is distributed under the license included in this repository.
