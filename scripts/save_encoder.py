import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

ROOT = Path(".")

df = pd.read_parquet(ROOT / "data/processed/features.parquet")
le = LabelEncoder()
le.fit(df["zone"])
joblib.dump(le, ROOT / "models/zone_encoder.pkl")
print("Zone encoder saved!")
print(f"Classes: {le.classes_[:5]}...")