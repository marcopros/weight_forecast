"""FastAPI serving endpoint for cumulative weight predictions.

Start with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager
from datetime import date

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# ── Globals filled at startup ──────────────────────────────────────────────
MODEL = None
RM_STATS: pd.DataFrame | None = None
MASTER_DF: pd.DataFrame | None = None
CFG: dict = {}


def load_config(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and reference data once at startup."""
    global MODEL, RM_STATS, MASTER_DF, CFG
    cfg_path = os.environ.get("CONFIG_PATH", "configs/params.yaml")
    CFG = load_config(cfg_path)
    MODEL = joblib.load(CFG["data"]["processed"]["model"])
    MASTER_DF = pd.read_csv(
        CFG["data"]["processed"]["master"], parse_dates=["date"]
    )
    # Reconstruct rm_stats
    RM_STATS = (
        MASTER_DF.groupby("rm_id")["net_weight"]
        .agg([("rm_mean", "mean"), ("rm_std", "std"), ("rm_median", "median")])
        .reset_index()
    )
    yield


app = FastAPI(
    title="Cumulative Weight Forecast API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    rm_id: int
    forecast_end_date: date


class PredictionResponse(BaseModel):
    rm_id: int
    forecast_end_date: str
    cumulative_weight: float


class BatchRequest(BaseModel):
    items: list[PredictionRequest]


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


# ── Feature builder ────────────────────────────────────────────────────────
def build_features_for_request(rm_id: int, end_date: date) -> pd.DataFrame:
    """Build a single feature row for the given rm_id and date."""
    dt = pd.Timestamp(end_date)
    year_start = pd.Timestamp(year=dt.year, month=1, day=1)

    row = {
        "month": dt.month,
        "day": dt.day,
        "dayofweek": dt.dayofweek,
        "dayofyear": dt.dayofyear,
        "weekofyear": dt.isocalendar().week,
        "quarter": dt.quarter,
        "days_since_year_start": (dt - year_start).days + 1,
    }

    # Compute lags from historical data
    rm_hist = MASTER_DF[MASTER_DF["rm_id"] == rm_id].sort_values("date")
    if rm_hist.empty:
        for lag in CFG["features"]["lag_days"]:
            row[f"lag_{lag}d"] = 0.0
        for w in CFG["features"]["rolling_windows"]:
            row[f"rolling_mean_{w}d"] = 0.0
            row[f"rolling_sum_{w}d"] = 0.0
    else:
        for lag in CFG["features"]["lag_days"]:
            target_date = dt - pd.Timedelta(days=lag)
            match = rm_hist[rm_hist["date"] == target_date]
            row[f"lag_{lag}d"] = float(match["net_weight"].iloc[0]) if not match.empty else 0.0

        for w in CFG["features"]["rolling_windows"]:
            window_end = dt - pd.Timedelta(days=1)
            window_start = window_end - pd.Timedelta(days=w - 1)
            window_data = rm_hist[
                (rm_hist["date"] >= window_start) & (rm_hist["date"] <= window_end)
            ]["net_weight"]
            row[f"rolling_mean_{w}d"] = float(window_data.mean()) if len(window_data) > 0 else 0.0
            row[f"rolling_sum_{w}d"] = float(window_data.sum()) if len(window_data) > 0 else 0.0

    # rm_stats
    rm_row = RM_STATS[RM_STATS["rm_id"] == rm_id]
    if not rm_row.empty:
        row["rm_mean"] = float(rm_row["rm_mean"].iloc[0])
        row["rm_std"] = float(rm_row["rm_std"].iloc[0])
        row["rm_median"] = float(rm_row["rm_median"].iloc[0])
    else:
        row["rm_mean"] = 0.0
        row["rm_std"] = 0.0
        row["rm_median"] = 0.0

    return pd.DataFrame([row])[CFG["features"]["feature_cols"]]


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    features = build_features_for_request(req.rm_id, req.forecast_end_date)
    pred = float(np.clip(MODEL.predict(features)[0], 0, None))
    return PredictionResponse(
        rm_id=req.rm_id,
        forecast_end_date=str(req.forecast_end_date),
        cumulative_weight=round(pred, 2),
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    results = []
    for item in req.items:
        features = build_features_for_request(item.rm_id, item.forecast_end_date)
        pred = float(np.clip(MODEL.predict(features)[0], 0, None))
        results.append(
            PredictionResponse(
                rm_id=item.rm_id,
                forecast_end_date=str(item.forecast_end_date),
                cumulative_weight=round(pred, 2),
            )
        )
    return BatchResponse(predictions=results)
