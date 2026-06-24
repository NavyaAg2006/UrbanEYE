from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import Literal
from services import predict_risk, get_history, get_heatmap

class PredictRequest(BaseModel):
    zone: str
    date: str
    timeslot: Literal["morning", "afternoon", "evening", "night"]

class ShapFactor(BaseModel):
    feature: str
    value: float

class Anomaly(BaseModel):
    flag: bool
    score: float
    message: str

class PredictResponse(BaseModel):
    risk_score: float
    risk_label: str
    shap_factors: list[ShapFactor]
    anomaly: Anomaly

class HistoryResponse(BaseModel):
    dates: list[str]
    counts: list[float]
    granularity: Literal["daily","weekly","monthly"]

class ZoneRisk(BaseModel):
    name: str
    risk_score: float

app = FastAPI(title="UrbanEye API", version="1.0.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict",response_model=PredictResponse)
def predict(request:PredictRequest):
    try:
        result=predict_risk(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{zone}",response_model=HistoryResponse)
def history(zone:str):
    try:
        result=get_history(zone)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/heatmap", response_model=list[ZoneRisk])
def heatmap():
    try:
        result = get_heatmap()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from sqlalchemy import text
from database import get_engine

def get_history(zone: str):
    engine = get_engine()
    with engine.connect() as conn:
        # Get total crimes per day for this zone
        query = text("""
            SELECT date, SUM(crime_count) as count
            FROM crimes
            WHERE zone = :zone
            GROUP BY date
            ORDER BY date
        """)
        df = pd.read_sql(query, conn, params={"zone": zone})

    df["date"] = pd.to_datetime(df["date"])
    total = df["count"].sum()

    if total < 50:
        # Monthly
        df = df.resample("ME", on="date")["count"].sum().reset_index()
        df = df.tail(6)
        granularity = "monthly"
    elif total <= 300:
        # Weekly
        df = df.resample("W", on="date")["count"].sum().reset_index()
        df = df.tail(16)
        granularity = "weekly"
    else:
        # Daily
        df = df.tail(60)
        granularity = "daily"

    return HistoryResponse(
        dates=df["date"].dt.strftime("%Y-%m-%d").tolist(),
        counts=df["count"].tolist(),
        granularity=granularity
    )


def get_heatmap():
    engine = get_engine()
    with engine.connect() as conn:
        # Latest available date's risk scores per zone
        query = text("""
            SELECT zone, AVG(risk_score) as risk_score
            FROM crimes
            WHERE date = (SELECT MAX(date) FROM crimes)
            GROUP BY zone
        """)
        df = pd.read_sql(query, conn)

    return [
        ZoneRisk(name=str(row["zone"]), risk_score=round(row["risk_score"], 4))
        for _, row in df.iterrows()
    ]