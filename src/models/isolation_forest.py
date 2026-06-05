import pandas as pd
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine
import joblib
from dotenv import load_dotenv
from pathlib import Path
import os
import mlflow
import mlflow.sklearn
from evaluate import setup_mlflow

ROOT = Path(__file__).parent.parent.parent

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

FEATURES = [
    'hour_bucket', 'day_of_week', 'month', 'is_weekend',
    'is_holiday', 'zone_population_density', 'zone_area_km2',
    'zone_historical_crime_rate', 'rolling_7day_crime_rate',
    'rolling_30day_crime_rate', 'temperature', 'precipitation',
    'humidity', 'wind_speed'
]

def train_isolation_forest():
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM crimes", engine)
    print(f"Data loaded! Shape: {df.shape}")

    X = df[FEATURES]

    with mlflow.start_run(run_name="isolation_forest"):
        params = {
            "n_estimators": 100,
            "contamination": 0.02,
            "random_state": 42
        }

        model = IsolationForest(**params)
        model.fit(X)
        print("Model trained successfully!")

        mlflow.log_params(params)
        mlflow.sklearn.log_model(model, "isolation_forest_model")

        joblib.dump(model, ROOT / 'models/isolation_forest_model.pkl')
        print("Model saved!")

    return model, df

def detect_anomaly(model, df, zone, timeslot, date):
    zone_data = df[(df['zone'] == zone) & (df['timeslot'] == timeslot)]
    historical_avg = zone_data['rolling_7day_crime_rate'].mean()
    current_data = zone_data[zone_data['date'] == date]

    if len(current_data) == 0:
        return {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "message": "No data available for this zone and date"
        }

    X_current = current_data[FEATURES]
    prediction = model.predict(X_current)[0]
    score = model.decision_function(X_current)[0]

    anomaly_score = round(1 - (score + 0.5), 2)
    anomaly_score = max(0.0, min(1.0, anomaly_score))

    current_rate = current_data['rolling_7day_crime_rate'].values[0]
    if historical_avg > 0:
        multiplier = round(current_rate / historical_avg, 1)
    else:
        multiplier = 1.0

    is_anomaly = prediction == -1

    if is_anomaly:
        day_name = pd.to_datetime(date).strftime('%A')
        message = f"Crime rate is {multiplier}x higher than usual {day_name} {timeslot} average for this zone"
    else:
        message = "No unusual activity detected for this zone at this time"

    return {
        "is_anomaly": is_anomaly,
        "anomaly_score": anomaly_score,
        "message": message
    }


if __name__ == "__main__":
    setup_mlflow()
    model, df = train_isolation_forest()
    result = detect_anomaly(model, df, "Austin", "night", "2023-01-06")
    print(result)