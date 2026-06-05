import pandas as pd
import numpy as np 
from sklearn.ensemble import IsolationForest
from sqlalchemy import create_engine
import joblib
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
# Features that the model will use for training 

FEATURES = [
    'hour_bucket', 'day_of_week', 'month', 'is_weekend',
    'is_holiday', 'zone_population_density', 'zone_area_km2',
    'zone_historical_crime_rate', 'rolling_7day_crime_rate',
    'rolling_30day_crime_rate', 'temperature', 'precipitation',
    'humidity', 'wind_speed'
]
def train_isolation_forest():
    # Load data from database
    engine = create_engine(DATABASE_URL)
    df = pd.read_sql("SELECT * FROM crimes", engine)
    print(f"Data loaded! Shape: {df.shape}")

    # Select only freature columns
    X = df[FEATURES]

    # Create the Isolation Forest model
    model = IsolationForest(
        n_estimators=100,
        contamination=0.02,
        random_state=42
    )

    # Train the model
    model.fit(X)
    print("Model trained successfully!")

    # Save the model to a file
    joblib.dump(model, 'src/models/isolation_forest_model.pkl')
    print("Model saved!")
    return model, df

def detect_anomaly(model, df, zone, timeslot, date):
    # Filter data for the specific zone and timeslot
    zone_data = df[(df['zone'] == zone) & (df['timeslot'] == timeslot)]

    # Calculate historical average crime rate for this zone and timeslot
    historical_avg = zone_data['rolling_7day_crime_rate'].mean()

    # Get current crime rate
    current_data = zone_data[zone_data['date'] == date]

    if len(current_data) == 0:
        return {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "message": "No data available for this zone and date"
        }

    # Select features for prediction
    X_current = current_data[FEATURES]

    # Get anomaly prediction (-1 = anomaly, 1 = normal)
    prediction = model.predict(X_current)[0]
    score = model.decision_function(X_current)[0]

    # Normalize score to 0-1 range
    anomaly_score = round(1 - (score + 0.5), 2)
    anomaly_score = max(0.0, min(1.0, anomaly_score))

    # Calculate multiplier
    current_rate = current_data['rolling_7day_crime_rate'].values[0]
    if historical_avg > 0:
        multiplier = round(current_rate / historical_avg, 1)
    else:
        multiplier = 1.0

    # Check if anomaly
    is_anomaly = prediction == -1

    # Create message
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
    model, df = train_isolation_forest()
    
    # Test with a sample
    result = detect_anomaly(model, df, "Austin", "night", "2023-01-06")
    print(result)