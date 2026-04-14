"""FastAPI serving endpoint for cumulative weight predictions.

Features: input validation, prediction logging, Prometheus metrics,
feedback loop endpoint, and optimised feature building.
Start with: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import csv
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

from src.data.feature_builder import build_single_row

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("forecast-api")

# ── Globals filled at startup ──────────────────────────────────────────────
MODEL = None
RM_STATS: pd.DataFrame | None = None
MASTER_DF: pd.DataFrame | None = None
MATERIAL_INFO: pd.DataFrame | None = None
CFG: dict = {}
KNOWN_RM_IDS: set = set()
# Pre-indexed history per rm_id for O(1) lookup
RM_HISTORY_INDEX: dict[int, pd.DataFrame] = {}
MODEL_HASH: str = "unknown"

# ── Prometheus-style metrics (in-memory counters) ──────────────────────────
METRICS = {
    "requests_total": 0,
    "requests_errors": 0,
    "latency_sum_ms": 0.0,
    "latency_count": 0,
    "latency_max_ms": 0.0,
}

# Prediction log file path
PREDICTION_LOG_PATH = "data/processed/prediction_log.csv"


def load_config(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _init_prediction_log():
    """Ensure prediction log CSV exists with headers."""
    if not os.path.exists(PREDICTION_LOG_PATH):
        os.makedirs(os.path.dirname(PREDICTION_LOG_PATH), exist_ok=True)
        with open(PREDICTION_LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            cols = ["timestamp", "rm_id", "forecast_end_date",
                    "cumulative_weight", "latency_ms"]
            writer.writerow(cols)


def _log_prediction(
    rm_id: int, forecast_end_date: str,
    cumulative_weight: float, latency_ms: float,
):
    """Append prediction to the prediction log (append-only)."""
    try:
        with open(PREDICTION_LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            row = [datetime.now(UTC).isoformat(), rm_id,
                   forecast_end_date, cumulative_weight, latency_ms]
            writer.writerow(row)
    except OSError:
        logger.warning("Failed to write prediction log")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and reference data once at startup."""
    global MODEL, RM_STATS, MASTER_DF, MATERIAL_INFO, CFG
    global KNOWN_RM_IDS, RM_HISTORY_INDEX, MODEL_HASH
    cfg_path = os.environ.get("CONFIG_PATH", "configs/params.yaml")
    CFG = load_config(cfg_path)

    logger.info("Loading model...")
    MODEL = joblib.load(CFG["data"]["processed"]["model"])

    logger.info("Loading master data...")
    MASTER_DF = pd.read_csv(
        CFG["data"]["processed"]["master"], parse_dates=["date"]
    )

    # Build pre-indexed history per rm_id for fast lookup
    for rm_id, group in MASTER_DF.groupby("rm_id"):
        RM_HISTORY_INDEX[int(rm_id)] = group.sort_values("date").copy()

    KNOWN_RM_IDS = set(MASTER_DF["rm_id"].unique())

    # Reconstruct rm_stats
    RM_STATS = (
        MASTER_DF.groupby("rm_id")["net_weight"]
        .agg([("rm_mean", "mean"), ("rm_std", "std"), ("rm_median", "median")])
        .reset_index()
    )

    # Load material info if available
    if CFG["features"].get("use_material_features", False):
        materials_path = CFG["data"]["raw"].get("materials")
        if materials_path and os.path.exists(materials_path):
            from src.data.clean import load_materials
            MATERIAL_INFO = load_materials(materials_path)
            logger.info(f"Loaded material features for {len(MATERIAL_INFO)} rm_ids")

    # Compute model hash for versioning
    try:
        import hashlib
        sha = hashlib.sha256()
        with open(CFG["data"]["processed"]["model"], "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        MODEL_HASH = sha.hexdigest()[:12]
    except OSError:
        MODEL_HASH = "unknown"

    _init_prediction_log()
    logger.info(f"Startup complete. Model hash: {MODEL_HASH}, {len(KNOWN_RM_IDS)} rm_ids loaded")
    yield


app = FastAPI(
    title="Cumulative Weight Forecast API",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Schemas ────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    rm_id: int
    forecast_end_date: date = Field(
        validation_alias=AliasChoices("forecast_end_date", "date"),
        serialization_alias="forecast_end_date",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_date_alias(cls, data):
        if isinstance(data, dict) and "forecast_end_date" not in data and "date" in data:
            data = data.copy()
            data["forecast_end_date"] = data["date"]
        return data

    @field_validator("forecast_end_date")
    @classmethod
    def validate_date_range(cls, v):
        if v.year < 2020 or v.year > 2030:
            raise ValueError("forecast_end_date must be between 2020 and 2030")
        return v


class PredictionResponse(BaseModel):
    rm_id: int
    forecast_end_date: str
    cumulative_weight: float
    model_hash: str


class BatchRequest(BaseModel):
    items: list[PredictionRequest]

    @field_validator("items")
    @classmethod
    def validate_batch_size(cls, v):
        if len(v) > 1000:
            raise ValueError("Batch size must not exceed 1000 items")
        if len(v) == 0:
            raise ValueError("Batch must contain at least 1 item")
        return v


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class ActualsRequest(BaseModel):
    """Feedback loop: submit actual cumulative weights for past predictions."""
    rm_id: int
    date: date
    actual_cumulative_weight: float


class ActualsBatchRequest(BaseModel):
    items: list[ActualsRequest]


# ── Feature builder (uses unified feature_builder) ─────────────────────────
def _build_features(rm_id: int, end_date: date) -> pd.DataFrame:
    """Build features using the pre-indexed history for O(1) rm_id lookup."""
    rm_hist = RM_HISTORY_INDEX.get(rm_id, pd.DataFrame())
    return build_single_row(
        rm_id=rm_id,
        target_date=pd.Timestamp(end_date),
        rm_history=rm_hist,
        rm_stats=RM_STATS,
        cfg=CFG,
        material_info=MATERIAL_INFO,
    )


def _do_predict(rm_id: int, forecast_end_date: date) -> tuple[float, float]:
    """Run prediction and return (cumulative_weight, latency_ms)."""
    start = time.perf_counter()
    features = _build_features(rm_id, forecast_end_date)
    pred = float(np.clip(MODEL.predict(features)[0], 0, None))
    latency_ms = (time.perf_counter() - start) * 1000
    return round(pred, 2), latency_ms


# ── Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_hash": MODEL_HASH,
        "known_rm_ids": len(KNOWN_RM_IDS),
    }


@app.get("/metrics")
def metrics():
    """Prometheus-style metrics endpoint for observability."""
    avg_latency = (
        METRICS["latency_sum_ms"] / METRICS["latency_count"]
        if METRICS["latency_count"] > 0
        else 0
    )
    return {
        "requests_total": METRICS["requests_total"],
        "requests_errors": METRICS["requests_errors"],
        "avg_latency_ms": round(avg_latency, 2),
        "max_latency_ms": round(METRICS["latency_max_ms"], 2),
        "model_hash": MODEL_HASH,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    METRICS["requests_total"] += 1

    if req.rm_id not in KNOWN_RM_IDS:
        METRICS["requests_errors"] += 1
        raise HTTPException(
            status_code=400,
            detail=f"Unknown rm_id={req.rm_id}. Known rm_ids: {len(KNOWN_RM_IDS)} total.",
        )

    pred, latency_ms = _do_predict(req.rm_id, req.forecast_end_date)

    METRICS["latency_sum_ms"] += latency_ms
    METRICS["latency_count"] += 1
    METRICS["latency_max_ms"] = max(METRICS["latency_max_ms"], latency_ms)

    _log_prediction(req.rm_id, str(req.forecast_end_date), pred, latency_ms)

    return PredictionResponse(
        rm_id=req.rm_id,
        forecast_end_date=str(req.forecast_end_date),
        cumulative_weight=pred,
        model_hash=MODEL_HASH,
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    METRICS["requests_total"] += 1
    results = []

    for item in req.items:
        if item.rm_id not in KNOWN_RM_IDS:
            METRICS["requests_errors"] += 1
            raise HTTPException(
                status_code=400,
                detail=f"Unknown rm_id={item.rm_id} in batch.",
            )

        pred, latency_ms = _do_predict(item.rm_id, item.forecast_end_date)

        METRICS["latency_sum_ms"] += latency_ms
        METRICS["latency_count"] += 1
        METRICS["latency_max_ms"] = max(METRICS["latency_max_ms"], latency_ms)

        _log_prediction(item.rm_id, str(item.forecast_end_date), pred, latency_ms)

        results.append(
            PredictionResponse(
                rm_id=item.rm_id,
                forecast_end_date=str(item.forecast_end_date),
                cumulative_weight=pred,
                model_hash=MODEL_HASH,
            )
        )
    return BatchResponse(predictions=results)


@app.post("/actuals")
def submit_actuals(req: ActualsBatchRequest):
    """Feedback loop: submit actual weights for past predictions.

    This enables real production monitoring by comparing what was predicted
    against what actually happened.
    """
    actuals_path = "data/processed/actuals_log.csv"
    write_header = not os.path.exists(actuals_path)

    try:
        with open(actuals_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "rm_id", "date", "actual_cumulative_weight"])
            for item in req.items:
                writer.writerow([
                    datetime.now(UTC).isoformat(),
                    item.rm_id,
                    str(item.date),
                    item.actual_cumulative_weight,
                ])
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to log actuals: {e}")

    return {"status": "ok", "items_logged": len(req.items)}
