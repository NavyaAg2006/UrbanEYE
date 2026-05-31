from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from typing import Literal

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