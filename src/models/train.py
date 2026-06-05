import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from evaluate import setup_mlflow

ROOT = Path(__file__).parent.parent.parent

# Load features
df = pd.read_parquet(ROOT / "data/processed/features.parquet")

# Encode season
season_map = {"spring": 0, "summer": 1, "fall": 2, "winter": 3}
df["season"] = df["season"].map(season_map)

# Encode zone
le = LabelEncoder()
df["zone"] = le.fit_transform(df["zone"])

# Time-based split
train = df[df["year"] <= 2023]
val = df[df["year"] == 2024]
test = df[df["year"] == 2025]

print(f"Train: {train.shape}")
print(f"Val:   {val.shape}")
print(f"Test:  {test.shape}")

# Features & target
feature_cols = [
    "hour_bucket", "day_of_week", "month", "season", "is_weekend", "is_holiday",
    "zone", "zone_population_density", "zone_area_km2", "zone_historical_crime_rate",
    "rolling_7day_crime_rate", "rolling_30day_crime_rate",
    "temperature", "precipitation", "humidity", "wind_speed"
]

X_train = train[feature_cols]
y_train = train["risk_score"]
X_val = val[feature_cols]
y_val = val["risk_score"]
X_test = test[feature_cols]
y_test = test["risk_score"]

# Setup MLflow
setup_mlflow()

# --- Baseline: Linear Regression ---
with mlflow.start_run(run_name="baseline_linear_regression"):

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    val_preds = lr.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    mae = mean_absolute_error(y_val, val_preds)
    r2 = r2_score(y_val, val_preds)

    # Log metrics
    mlflow.log_metric("val_rmse", round(rmse, 4))
    mlflow.log_metric("val_mae", round(mae, 4))
    mlflow.log_metric("val_r2", round(r2, 4))

    # Log model
    mlflow.sklearn.log_model(lr, "linear_regression_model")

    print("── Baseline Linear Regression ──")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

# --- XGBoost ---
with mlflow.start_run(run_name="xgboost"):

    params = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 20
    }

    xgb = XGBRegressor(**params)
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    val_preds_xgb = xgb.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, val_preds_xgb))
    mae = mean_absolute_error(y_val, val_preds_xgb)
    r2 = r2_score(y_val, val_preds_xgb)

    # Log params
    mlflow.log_params(params)

    # Log metrics
    mlflow.log_metric("val_rmse", round(rmse, 4))
    mlflow.log_metric("val_mae", round(mae, 4))
    mlflow.log_metric("val_r2", round(r2, 4))

    # Log model
    mlflow.xgboost.log_model(xgb, "xgboost_model")

    # Save model locally too
    with open(ROOT / "models/xgb_model.pkl", "wb") as f:
        pickle.dump(xgb, f)

    print("── XGBoost ──")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print("Model saved")