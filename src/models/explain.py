import shap
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

# Load trained XGBoost model
with open(ROOT / "models/xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

# Load features
df = pd.read_parquet(ROOT / "data/processed/features.parquet")

# Encode season and zone same way as training
season_map = {"spring": 0, "summer": 1, "fall": 2, "winter": 3}
df["season"] = df["season"].map(season_map)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df["zone"] = le.fit_transform(df["zone"])

feature_cols = [
    "hour_bucket", "day_of_week", "month", "season", "is_weekend", "is_holiday",
    "zone", "zone_population_density", "zone_area_km2", "zone_historical_crime_rate",
    "rolling_7day_crime_rate", "rolling_30day_crime_rate",
    "temperature", "precipitation", "humidity", "wind_speed"
]

X = df[feature_cols]

# Create SHAP explainer
print("Creating SHAP explainer...")
explainer = shap.TreeExplainer(xgb_model)

# Save explainer for FastAPI to use
joblib.dump(explainer, ROOT / "models/shap_explainer.pkl")
print("SHAP explainer saved!")


def get_shap_factors(input_row: pd.DataFrame) -> list[dict]:
    """
    Given a single row of features, return top 5 SHAP factors.
    input_row: DataFrame with exactly one row, same columns as feature_cols
    """
    shap_values = explainer.shap_values(input_row)

    # Pair feature names with their SHAP values
    shap_pairs = list(zip(feature_cols, shap_values[0]))

    # Sort by absolute value — biggest impact first
    shap_pairs_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)

    # Return top 5
    top_5 = [
        {"feature": name, "value": round(float(val), 4)}
        for name, val in shap_pairs_sorted[:5]
    ]

    return top_5


if __name__ == "__main__":
    # Test with one sample row
    sample = X.iloc[[0]]
    factors = get_shap_factors(sample)

    print("\nTop 5 SHAP factors for sample row:")
    for i, f in enumerate(factors, 1):
        print(f"  {i}. {f['feature']}: {f['value']}")