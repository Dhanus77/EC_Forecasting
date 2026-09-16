import pandas as pd
import numpy as np


def create_forecast_features(raw_data):
    """
    Create the 49 features required by the final
    Ridge Regression energy forecasting model.

    The function uses historical energy consumption,
    temporal features, indoor environmental features,
    and weather features.
    """

    data = raw_data.copy()

  
    # Date handling


    data["date"] = pd.to_datetime(data["date"])

    data = (
        data
        .sort_values("date")
        .reset_index(drop=True)
    )


    # Remove redundant random variables
 

    data = data.drop(
        columns=["rv1", "rv2"],
        errors="ignore"
    )


    # Temporal Features
  

    data["hour"] = data["date"].dt.hour
    data["day"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    data["week_of_year"] = (
        data["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    # Cyclical hour encoding
    data["hour_sin"] = np.sin(
        2 * np.pi * data["hour"] / 24
    )

    data["hour_cos"] = np.cos(
        2 * np.pi * data["hour"] / 24
    )

      # Historical Energy Lag Features


    data["lag_10min"] = (
        data["Appliances"].shift(1)
    )

    data["lag_20min"] = (
        data["Appliances"].shift(2)
    )

    data["lag_30min"] = (
        data["Appliances"].shift(3)
    )

    data["lag_1hour"] = (
        data["Appliances"].shift(6)
    )

    data["lag_2hours"] = (
        data["Appliances"].shift(12)
    )

    data["lag_1day"] = (
        data["Appliances"].shift(144)
    )

    data["lag_1week"] = (
        data["Appliances"].shift(1008)
    )


    # Rolling Consumption Features


    # Shift first to ensure only past
    # consumption is used.
    past_consumption = (
        data["Appliances"].shift(1)
    )

    data["rolling_mean_30min"] = (
        past_consumption
        .rolling(3)
        .mean()
    )

    data["rolling_std_30min"] = (
        past_consumption
        .rolling(3)
        .std()
    )

    data["rolling_mean_1hour"] = (
        past_consumption
        .rolling(6)
        .mean()
    )

    data["rolling_std_1hour"] = (
        past_consumption
        .rolling(6)
        .std()
    )

    data["rolling_mean_2hours"] = (
        past_consumption
        .rolling(12)
        .mean()
    )

    data["rolling_std_2hours"] = (
        past_consumption
        .rolling(12)
        .std()
    )


    # Weather Change Features
  

    data["T_out_change_1h"] = (
        data["T_out"] -
        data["T_out"].shift(6)
    )

    data["RH_out_change_1h"] = (
        data["RH_out"] -
        data["RH_out"].shift(6)
    )

      # Remove rows without enough history
  

    data = (
        data
        .dropna()
        .reset_index(drop=True)
    )

    return data