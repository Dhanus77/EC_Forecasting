import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import joblib

try:
    import requests
except ImportError:
    requests = None



# PATHS / IMPORTS


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from feature_engineering import create_forecast_features


DATA_PATH = os.path.join(BASE_DIR, "data", "energydata_complete.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_ridge_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "final_scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "model_features.pkl")


# PAGE


st.set_page_config(
    page_title="Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)



# STYLE


st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        letter-spacing: -0.6px;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        color: #8fa0b5;
        font-size: 0.92rem;
        margin-bottom: 1.3rem;
    }

    .section-title {
        font-size: 1.28rem;
        font-weight: 700;
        margin-top: 1.35rem;
        margin-bottom: 0.8rem;
    }

    .metric-card {
        background: #151922;
        border: 1px solid #2b3442;
        border-radius: 11px;
        padding: 17px 19px;
        min-height: 108px;
    }

    .metric-label {
        color: #8fa0b5;
        font-size: 0.72rem;
        font-weight: 650;
        letter-spacing: 0.35px;
        margin-bottom: 7px;
    }

    .metric-value {
        color: #f5f7fa;
        font-size: 1.45rem;
        font-weight: 720;
    }

    .metric-caption {
        color: #718096;
        font-size: 0.70rem;
        margin-top: 6px;
    }

    .status {
        display: inline-block;
        padding: 5px 11px;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 650;
    }

    .status-live {
        color: #4ade80;
        background: rgba(34,197,94,.10);
        border: 1px solid rgba(34,197,94,.30);
    }

    .status-sim {
        color: #60a5fa;
        background: rgba(59,130,246,.10);
        border: 1px solid rgba(59,130,246,.30);
    }

    .status-manual {
        color: #fbbf24;
        background: rgba(251,191,36,.09);
        border: 1px solid rgba(251,191,36,.25);
    }

    .status-history {
        color: #94a3b8;
        background: rgba(148,163,184,.08);
        border: 1px solid rgba(148,163,184,.22);
    }

    .info-box {
        background: #142b43;
        border: 1px solid #24517d;
        border-radius: 9px;
        padding: 13px 16px;
        color: #b8dcff;
        font-size: 0.84rem;
        line-height: 1.5;
    }

    .waiting-box {
        background: #171b22;
        border: 1px dashed #465160;
        border-radius: 11px;
        padding: 24px;
        color: #c6ced9;
        line-height: 1.55;
    }

    .small-muted {
        color: #718096;
        font-size: 0.74rem;
    }

    hr {
        border-top: 1px solid #2a303b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# LOAD MODEL / DATA


@st.cache_resource
def load_model_files():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    model_features = joblib.load(FEATURES_PATH)
    return model, scaler, model_features


@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values("date").reset_index(drop=True)


model, scaler, model_features = load_model_files()
historical_data = load_dataset()

# SESSION STATE

if "simulation_history" not in st.session_state:
    seed = historical_data.tail(1500).copy()
    now = pd.Timestamp.now().floor("10min")
    seed["date"] = seed["date"] + (now - seed["date"].iloc[-1])
    st.session_state.simulation_history = seed.reset_index(drop=True)

if "manual_history" not in st.session_state:
    st.session_state.manual_history = historical_data.tail(1500).copy().reset_index(drop=True)

if "manual_result" not in st.session_state:
    st.session_state.manual_result = None



# HELPERS


def make_forecast(data):
    engineered = create_forecast_features(data)

    if engineered.empty:
        raise ValueError("Not enough historical readings to create forecasting features.")

    latest = engineered.iloc[[-1]][model_features]
    scaled = scaler.transform(latest)
    prediction = float(model.predict(scaled)[0])

    current = float(data["Appliances"].iloc[-1])
    latest_time = pd.Timestamp(data["date"].iloc[-1])
    forecast_time = latest_time + pd.Timedelta(hours=1)
    change = prediction - current

    if change > 2:
        trend = "Increase"
    elif change < -2:
        trend = "Decrease"
    else:
        trend = "Stable"

    return {
        "current": current,
        "prediction": prediction,
        "change": change,
        "trend": trend,
        "latest_time": latest_time,
        "forecast_time": forecast_time,
    }


def add_weather_to_row(row, weather):
    row["T_out"] = weather["temperature"]
    row["RH_out"] = weather["humidity"]
    row["Windspeed"] = weather["wind_speed"]
    row["Visibility"] = weather["visibility"]
    row["Press_mm_hg"] = weather["pressure"]
    row["Tdewpoint"] = weather["dewpoint"]
    return row


def generate_simulated_reading(history):
    last = history.iloc[-1].copy()

    next_time = pd.Timestamp(last["date"]) + pd.Timedelta(minutes=10)
    hour = next_time.hour + next_time.minute / 60.0

    morning_peak = 100 * np.exp(-((hour - 8) / 2.0) ** 2)
    evening_peak = 150 * np.exp(-((hour - 19) / 3.0) ** 2)

    base = 75 + morning_peak + evening_peak
    recent_mean = history["Appliances"].tail(6).mean()

    energy = (
        0.55 * base
        + 0.45 * recent_mean
        + np.random.normal(0, 12)
    )

    energy = float(np.clip(energy, 10, 600))

    temperature = (
        27
        + 4 * np.sin(2 * np.pi * (hour - 14) / 24)
        + np.random.normal(0, 0.25)
    )

    humidity = (
        62
        - 12 * np.sin(2 * np.pi * (hour - 14) / 24)
        + np.random.normal(0, 1.2)
    )

    wind = max(
        0.5,
        float(last["Windspeed"]) + np.random.normal(0, 0.25)
    )

    visibility = max(
        5,
        float(last["Visibility"]) + np.random.normal(0, 0.4)
    )

    pressure = (
        float(last["Press_mm_hg"]) + np.random.normal(0, 0.4)
    )

    dewpoint = temperature - ((100 - humidity) / 5.0)

    new_row = last.copy()
    new_row["date"] = next_time
    new_row["Appliances"] = energy
    new_row["T_out"] = temperature
    new_row["RH_out"] = np.clip(humidity, 20, 95)
    new_row["Windspeed"] = wind
    new_row["Visibility"] = visibility
    new_row["Press_mm_hg"] = pressure
    new_row["Tdewpoint"] = dewpoint

    for col in ["T1","T2","T3","T4","T5","T6","T7","T8","T9"]:
        new_row[col] = float(last[col]) + np.random.normal(0, 0.08)

    for col in ["RH_1","RH_2","RH_3","RH_4","RH_5","RH_6","RH_7","RH_8","RH_9"]:
        new_row[col] = np.clip(
            float(last[col]) + np.random.normal(0, 0.5),
            20,
            90,
        )

    new_row["lights"] = max(
        0,
        float(last["lights"]) + np.random.normal(0, 3),
    )

    return pd.DataFrame([new_row])


@st.cache_data(ttl=300)
def get_live_weather(location):
    """
    Gets current weather from Open-Meteo.
    This is used only for the LIVE mode's weather display.
    """
    if requests is None:
        return None, "The requests package is not installed."

    try:
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {
            "name": location,
            "count": 1,
            "language": "en",
            "format": "json",
        }

        geo_response = requests.get(
            geo_url,
            params=geo_params,
            timeout=8,
        )
        geo_response.raise_for_status()

        geo = geo_response.json()

        if not geo.get("results"):
            return None, f"Location '{location}' was not found."

        place = geo["results"][0]

        latitude = place["latitude"]
        longitude = place["longitude"]

        weather_url = "https://api.open-meteo.com/v1/forecast"

        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "wind_speed_10m,visibility,surface_pressure,"
                "dew_point_2m"
            ),
            "timezone": "auto",
        }

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=8,
        )
        weather_response.raise_for_status()

        current = weather_response.json()["current"]

        weather = {
            "temperature": float(current["temperature_2m"]),
            "humidity": float(current["relative_humidity_2m"]),
            "wind_speed": float(current["wind_speed_10m"]) / 3.6,
            "visibility": float(current["visibility"]) / 1000,
            "pressure": float(current["surface_pressure"]) * 0.75006156,
            "dewpoint": float(current["dew_point_2m"]),
            "time": current["time"],
            "location": place.get("name", location),
        }

        return weather, None

    except Exception as exc:
        return None, f"Live weather unavailable: {exc}"


def weather_cards(row):
    a, b, c = st.columns(3)

    with a:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">OUTDOOR TEMPERATURE</div>
                <div class="metric-value">{row["T_out"]:.1f} °C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">OUTDOOR HUMIDITY</div>
                <div class="metric-value">{row["RH_out"]:.1f} %</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">WIND SPEED</div>
                <div class="metric-value">{row["Windspeed"]:.1f} m/s</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">VISIBILITY</div>
                <div class="metric-value">{row["Visibility"]:.1f} km</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">PRESSURE</div>
                <div class="metric-value">{row["Press_mm_hg"]:.1f} mmHg</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">DEW POINT</div>
                <div class="metric-value">{row["Tdewpoint"]:.1f} °C</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_forecast(result):
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">CURRENT CONSUMPTION</div>
                <div class="metric-value">{result["current"]:.2f} Wh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">PREDICTED NEXT HOUR</div>
                <div class="metric-value">{result["prediction"]:.2f} Wh</div>
                <div class="metric-caption">1-hour-ahead forecast</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">EXPECTED CHANGE</div>
                <div class="metric-value">{result["change"]:+.2f} Wh</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">EXPECTED TREND</div>
                <div class="metric-value">{result["trend"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def show_timing(result):
    a, b, c = st.columns(3)

    with a:
        st.markdown("**Latest Reading Time**")
        st.code(
            result["latest_time"].strftime("%Y-%m-%d %H:%M:%S"),
            language=None,
        )

    with b:
        st.markdown("**Forecast Time**")
        st.code(
            result["forecast_time"].strftime("%Y-%m-%d %H:%M:%S"),
            language=None,
        )

    with c:
        st.markdown("**Forecast Horizon**")
        st.write("1 Hour Ahead")



# SIDEBAR


st.sidebar.markdown("### Forecast Control")

mode = st.sidebar.radio(
    "Data Source",
    [
        "LIVE",
        "LIVE SIMULATION",
        "MANUAL PREDICTION",
        "HISTORICAL DATASET",
    ],
    index=0,
)

st.sidebar.markdown("---")

st.sidebar.markdown("### Forecast Settings")
st.sidebar.write("Forecast Horizon")
st.sidebar.info("1 Hour Ahead")

st.sidebar.write("Model")
st.sidebar.info("Ridge Regression")

if mode == "LIVE":
    st.sidebar.markdown("---")
    weather_location = st.sidebar.text_input(
        "Weather Location",
        value="Chennai",
        help="Used only to retrieve current weather for LIVE mode.",
    )



# HEADER


st.markdown(
    '<div class="main-title">Energy Consumption Forecasting Using Time Series and Weather Features</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">One-hour-ahead energy consumption prediction</div>',
    unsafe_allow_html=True,
)



# LIVE MODE


if mode == "LIVE":

    st.markdown(
        '<span class="status status-live">● LIVE</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Live Energy Source</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="waiting-box">
        <strong>Waiting for live energy data</strong><br><br>
        Connect a smart meter, smart plug, IoT energy sensor,
        Home Assistant integration, or supported energy API to
        provide live consumption readings.<br><br>
        The forecasting pipeline is designed to receive the energy
        history from that source and pass it through the existing
        feature-engineering and Ridge Regression model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Live Weather</div>',
        unsafe_allow_html=True,
    )

    live_weather, weather_error = get_live_weather(weather_location)

    if live_weather is not None:

        st.markdown(
            f'<div class="small-muted">Current weather for {live_weather["location"]} · '
            f'{live_weather["time"]}</div>',
            unsafe_allow_html=True,
        )

        live_weather_row = {
            "T_out": live_weather["temperature"],
            "RH_out": live_weather["humidity"],
            "Windspeed": live_weather["wind_speed"],
            "Visibility": live_weather["visibility"],
            "Press_mm_hg": live_weather["pressure"],
            "Tdewpoint": live_weather["dewpoint"],
        }

        weather_cards(live_weather_row)

    else:
        st.warning(weather_error)

    st.markdown(
        '<div class="section-title">Forecast Status</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Forecasting will begin automatically after a compatible live "
        "energy source supplies sufficient historical readings for the "
        "required lag and rolling features."
    )



# LIVE SIMULATION


elif mode == "LIVE SIMULATION":

    @st.fragment(run_every="5s")
    def live_simulation_fragment():

        history = st.session_state.simulation_history

        new_reading = generate_simulated_reading(history)

        history = pd.concat(
            [history, new_reading],
            ignore_index=True,
        ).tail(1500).reset_index(drop=True)

        st.session_state.simulation_history = history

        result = make_forecast(history)

        st.markdown(
            '<span class="status status-sim">● LIVE SIMULATION</span>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="small-muted">Simulated energy readings · automatically refreshed every 5 seconds</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Forecast Overview</div>',
            unsafe_allow_html=True,
        )

        show_forecast(result)

        st.markdown(
            '<div class="section-title">Latest Weather & Environment</div>',
            unsafe_allow_html=True,
        )

        weather_cards(history.iloc[-1])

        st.markdown(
            '<div class="section-title">Energy Consumption Monitoring</div>',
            unsafe_allow_html=True,
        )

        chart_data = history.tail(36)[["date", "Appliances"]].copy()
        chart_data = chart_data.set_index("date")

        st.line_chart(
            chart_data,
            y="Appliances",
            height=350,
        )

        st.markdown(
            f'<div class="small-muted">Latest simulated reading: '
            f'{result["latest_time"].strftime("%Y-%m-%d %H:%M:%S")}</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Forecast Timing</div>',
            unsafe_allow_html=True,
        )

        show_timing(result)

    live_simulation_fragment()



# MANUAL PREDICTION


elif mode == "MANUAL PREDICTION":

    st.markdown(
        '<span class="status status-manual">● MANUAL PREDICTION</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Prediction Input</div>',
        unsafe_allow_html=True,
    )

    date_col, time_col = st.columns(2)

    default_date = pd.Timestamp.now().date()
    default_time = pd.Timestamp.now().floor("10min").time()

    with date_col:
        manual_date = st.date_input(
            "Prediction Date",
            value=default_date,
        )

    with time_col:
        manual_time = st.time_input(
            "Prediction Time",
            value=default_time,
        )

    st.markdown("**Energy Consumption**")

    energy_input = st.number_input(
        "Current Consumption (Wh)",
        min_value=1.0,
        max_value=2000.0,
        value=430.0,
        step=5.0,
    )

    st.markdown("**Weather & Environment**")

    c1, c2, c3 = st.columns(3)

    with c1:
        temperature_input = st.number_input(
            "Outdoor Temperature (°C)",
            min_value=-30.0,
            max_value=60.0,
            value=28.0,
            step=0.1,
        )

    with c2:
        humidity_input = st.number_input(
            "Outdoor Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0,
        )

    with c3:
        wind_input = st.number_input(
            "Wind Speed (m/s)",
            min_value=0.0,
            max_value=50.0,
            value=4.0,
            step=0.1,
        )

    c1, c2, c3 = st.columns(3)

    with c1:
        visibility_input = st.number_input(
            "Visibility (km)",
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=0.5,
        )

    with c2:
        pressure_input = st.number_input(
            "Pressure (mmHg)",
            min_value=600.0,
            max_value=850.0,
            value=755.0,
            step=0.1,
        )

    with c3:
        dewpoint_input = st.number_input(
            "Dew Point (°C)",
            min_value=-40.0,
            max_value=50.0,
            value=19.0,
            step=0.1,
        )

    generate = st.button(
        "Generate Forecast",
        type="primary",
        use_container_width=True,
    )

    if generate:

        prediction_time = pd.Timestamp(
            pd.Timestamp(manual_date).date()
        ) + pd.Timedelta(
            hours=manual_time.hour,
            minutes=manual_time.minute,
        )

        history = st.session_state.manual_history.copy()

        # Use the previous historical row as a structural template.
        row = history.iloc[-1].copy()

        row["date"] = prediction_time
        row["Appliances"] = energy_input
        row["T_out"] = temperature_input
        row["RH_out"] = humidity_input
        row["Windspeed"] = wind_input
        row["Visibility"] = visibility_input
        row["Press_mm_hg"] = pressure_input
        row["Tdewpoint"] = dewpoint_input

        manual_row = pd.DataFrame([row])

        history = pd.concat(
            [history, manual_row],
            ignore_index=True,
        )

        # Ensure chronological order after inserting today's/custom reading.
        history = history.sort_values("date").reset_index(drop=True)

        try:
            result = make_forecast(history)
            st.session_state.manual_result = result
            st.session_state.manual_display_row = row

        except Exception as exc:
            st.session_state.manual_result = None
            st.error(f"Unable to generate forecast: {exc}")

    if st.session_state.manual_result is not None:

        result = st.session_state.manual_result
        row = st.session_state.manual_display_row

        st.markdown(
            '<div class="section-title">Forecast Overview</div>',
            unsafe_allow_html=True,
        )

        show_forecast(result)

        st.markdown(
            '<div class="section-title">Weather & Environment</div>',
            unsafe_allow_html=True,
        )

        weather_cards(row)

        st.markdown(
            '<div class="section-title">Forecast Interpretation</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="info-box">
                Based on the supplied conditions, energy consumption is
                expected to <strong>{result["trend"].lower()}</strong>
                over the next hour, from
                <strong>{result["current"]:.2f} Wh</strong> to approximately
                <strong>{result["prediction"]:.2f} Wh</strong>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Forecast Timing</div>',
            unsafe_allow_html=True,
        )

        show_timing(result)

    else:

        st.info(
            "Enter today's or a custom date/time and the current energy "
            "and weather conditions, then select Generate Forecast."
        )



# HISTORICAL DATASET


else:

    result = make_forecast(historical_data)
    latest_row = historical_data.iloc[-1]

    st.markdown(
        '<span class="status status-history">● HISTORICAL DATASET</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Forecast Overview</div>',
        unsafe_allow_html=True,
    )

    show_forecast(result)

    st.markdown(
        '<div class="section-title">Latest Weather & Environment</div>',
        unsafe_allow_html=True,
    )

    weather_cards(latest_row)

    st.markdown(
        '<div class="section-title">Recent Energy Consumption</div>',
        unsafe_allow_html=True,
    )

    chart_data = historical_data.tail(36)[
        ["date", "Appliances"]
    ].copy()

    chart_data = chart_data.set_index("date")

    st.line_chart(
        chart_data,
        y="Appliances",
        height=350,
    )

    st.markdown(
        '<div class="section-title">Energy Consumption Insights</div>',
        unsafe_allow_html=True,
    )

    i1, i2, i3, i4 = st.columns(4)

    with i1:
        st.metric(
            "Average Consumption",
            f'{historical_data["Appliances"].mean():.2f} Wh',
        )

    with i2:
        st.metric(
            "Maximum Recorded",
            f'{historical_data["Appliances"].max():.0f} Wh',
        )

    with i3:
        st.metric(
            "Minimum Recorded",
            f'{historical_data["Appliances"].min():.0f} Wh',
        )

    with i4:
        st.metric(
            "Recent 6-Hour Average",
            f'{historical_data.tail(36)["Appliances"].mean():.2f} Wh',
        )

    st.markdown(
        '<div class="section-title">Forecast Timing</div>',
        unsafe_allow_html=True,
    )

    show_timing(result)

    st.markdown(
        '<div class="section-title">Model Information</div>',
        unsafe_allow_html=True,
    )

    m1, m2 = st.columns(2)

    with m1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">MODEL CONFIGURATION</div>
                <p><strong>Model:</strong> Ridge Regression</p>
                <p><strong>Regularization:</strong> α = 4000</p>
                <p><strong>Forecast Horizon:</strong> 1 hour</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">TEST PERFORMANCE</div>
                <p><strong>MAE:</strong> 39.63 Wh</p>
                <p><strong>RMSE:</strong> 80.08 Wh</p>
                <p><strong>R²:</strong> 0.233</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
