import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import text
import os
from database import get_engine

load_dotenv()

ROOT = Path(__file__).parent.parent.parent

# --- Load models once at startup ---
with open(ROOT / "models/xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

iso_forest = joblib.load(ROOT / "models/isolation_forest_model.pkl")
shap_explainer = joblib.load(ROOT / "models/shap_explainer.pkl")
zone_encoder = joblib.load(ROOT / "models/zone_encoder.pkl")

# --- Constants ---
TIMESLOT_TO_HOUR = {
    "morning": 8,
    "afternoon": 12,
    "evening": 16,
    "night": 20
}

SEASON_MAP = {
    "spring": 0,
    "summer": 1,
    "fall": 2,
    "winter": 3
}

FEATURE_COLS = [
    "hour_bucket", "day_of_week", "month", "season", "is_weekend", "is_holiday",
    "zone", "zone_population_density", "zone_area_km2", "zone_historical_crime_rate",
    "rolling_7day_crime_rate", "rolling_30day_crime_rate",
    "temperature", "precipitation", "humidity", "wind_speed"
]


def get_risk_label(score: float) -> str:
    if score < 0.33:
        return "Low"
    elif score < 0.66:
        return "Medium"
    return "High"


def get_season(month: int) -> int:
    if month in [3, 4, 5]:
        return 0   # spring
    elif month in [6, 7, 8]:
        return 1   # summer
    elif month in [9, 10, 11]:
        return 2   # fall
    return 3       # winter


def predict_risk(request) -> dict:
    engine = get_engine()
    date = pd.to_datetime(request.date)
    zone_encoded = zone_encoder.transform([request.zone])[0]

    # Fetch zone static features from DB
    query = text("""
        SELECT zone_population_density, zone_area_km2, zone_historical_crime_rate,
               rolling_7day_crime_rate, rolling_30day_crime_rate
        FROM crimes
        WHERE zone = :zone
        ORDER BY date DESC
        LIMIT 1
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"zone": zone_encoded}).fetchone()

    if result is None:
        raise ValueError(f"No data found for zone: {request.zone}")

    # Fetch weather for requested date
    weather_query = text("""
        SELECT temperature, precipitation, humidity, wind_speed
        FROM crimes
        WHERE date = :date
        LIMIT 1
    """)

    with engine.connect() as conn:
        weather = conn.execute(weather_query, {"date": request.date}).fetchone()

    # Fallback to zone average if date not found
    if weather is None:
        weather_query_fallback = text("""
            SELECT AVG(temperature), AVG(precipitation), AVG(humidity), AVG(wind_speed)
            FROM crimes