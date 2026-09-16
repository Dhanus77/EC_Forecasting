import os
import joblib
import pandas as pd

from feature_engineering import create_forecast_features



# Project Paths

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "energydata_complete.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_ridge_model.pkl"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_scaler.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "model_features.pkl"
)



# Load Model Artifacts


model = joblib.load(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

model_features = joblib.load(FEATURES_PATH)



# Forecast Function


def forecast_energy(data_path=DATA_PATH):
   
    # Load raw data
    

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}"
        )

    raw_data = pd.read_csv(data_path)

    required_columns = [
        "date",
        "Appliances"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in raw_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    raw_data["date"] = pd.to_datetime(
        raw_data["date"]
    )
    """
    Generate a one-hour-ahead energy consumption forecast.

    Parameters
    ----------
    data_path : str
        Path to the raw energy consumption CSV file.

    Returns
    -------
    dict
        Forecast information including timestamp,
        current consumption, predicted consumption,
        and expected trend.
    """

  
    # Load raw data
    

    raw_data = pd.read_csv(data_path)

    raw_data["date"] = pd.to_datetime(
        raw_data["date"]
    )

    raw_data = (
        raw_data
        .sort_values("date")
        .reset_index(drop=True)
    )

  
    # Generate engineered features
  

    engineered_data = create_forecast_features(
        raw_data
    )

  
    # Select latest feature row
   

    latest_features = engineered_data.iloc[[-1]]

    # Ensure exact feature order
    latest_features = latest_features[
        model_features
    ]

   
    # Scale features
   

    latest_features_scaled = scaler.transform(
        latest_features
    )

    
    # Generate forecast
    

    prediction = model.predict(
        latest_features_scaled
    )[0]

    
    # Get latest information
   

    latest_timestamp = raw_data["date"].max()

    forecast_timestamp = (
        latest_timestamp +
        pd.Timedelta(hours=1)
    )

    latest_consumption = (
        raw_data
        .sort_values("date")["Appliances"]
        .iloc[-1]
    )

    change = (
        prediction -
        latest_consumption
    )

    
    # Determine expected trend
    

    if change > 0:
        trend = "Increase"

    elif change < 0:
        trend = "Decrease"

    else:
        trend = "Stable"

    
    # Return forecast
    

    return {
        "latest_time": latest_timestamp,
        "forecast_time": forecast_timestamp,
        "current_consumption": latest_consumption,
        "predicted_consumption": prediction,
        "expected_change": change,
        "expected_trend": trend
    }



# Run Directly


if __name__ == "__main__":

    forecast = forecast_energy()

    print("\nEnergy Consumption Forecast")
    print("---------------------------")

    print(
        "Latest available time :",
        forecast["latest_time"]
    )

    print(
        "Forecast time          :",
        forecast["forecast_time"]
    )

    print(
        "Forecast horizon       : 1 hour"
    )

    print(
        "Current consumption    :",
        round(
            forecast["current_consumption"],
            2
        ),
        "Wh"
    )

    print(
        "Predicted consumption  :",
        round(
            forecast["predicted_consumption"],
            2
        ),
        "Wh"
    )

    print(
        "Expected change        :",
        round(
            forecast["expected_change"],
            2
        ),
        "Wh"
    )

    print(
        "Expected trend         :",
        forecast["expected_trend"]
    )