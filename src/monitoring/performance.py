"""Performance monitoring: track predictions vs actuals over time.

Supports both offline evaluation (validation data) and online monitoring
(comparing logged predictions against actual data when available).
Can be run standalone: python -m src.monitoring.performance
"""

import json
import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def quantile_loss(y_true, y_pred, tau: float = 0.2) -> float:
    errors = y_true - y_pred
    return float(np.where(errors >= 0, tau * errors, -(1 - tau) * errors).mean())


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    tau: float = 0.2,
) -> dict:
    """Compare predictions against actuals. Returns performance metrics."""
    merged = predictions_df.merge(
        actuals_df,
        on=["rm_id", "date"],
        suffixes=("_pred", "_actual"),
    )

    if merged.empty:
        return {"error": "No matching rows between predictions and actuals"}

    y_true = merged.iloc[:, -1].values  # actual
    y_pred = merged.iloc[:, -2].values  # predicted

    errors = y_true - y_pred
    metrics = {
        "n_samples": int(len(merged)),
        "quantile_loss": quantile_loss(y_true, y_pred, tau),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "mean_prediction": float(y_pred.mean()),
        "mean_actual": float(y_true.mean()),
        "overestimation_rate": float((y_pred > y_true).mean()),
    }
    return metrics


def check_retrain_needed(metrics: dict, threshold: float) -> bool:
    """Returns True if quantile loss exceeds threshold, suggesting retraining."""
    return metrics.get("quantile_loss", 0) > threshold


def analyze_prediction_distribution(pred_log_path: str) -> dict:
    """Analyze logged predictions for distribution shifts over time."""
    if not os.path.exists(pred_log_path):
        return {"status": "no_prediction_log"}

    log_df = pd.read_csv(pred_log_path, parse_dates=["timestamp"])
    if log_df.empty:
        return {"status": "empty_log"}

    log_df["date"] = log_df["timestamp"].dt.date

    # Compare recent vs older predictions
    daily_stats = log_df.groupby("date").agg(
        n_requests=("cumulative_weight", "size"),
        mean_prediction=("cumulative_weight", "mean"),
        std_prediction=("cumulative_weight", "std"),
        p95_latency_ms=("latency_ms", lambda x: x.quantile(0.95)),
        p50_latency_ms=("latency_ms", lambda x: x.quantile(0.50)),
    ).reset_index()

    recent = daily_stats.tail(7)
    summary = {
        "status": "ok",
        "total_predictions": int(len(log_df)),
        "unique_rm_ids_served": int(log_df["rm_id"].nunique()),
        "recent_daily_stats": recent.to_dict(orient="records"),
        "overall_mean_prediction": float(log_df["cumulative_weight"].mean()),
        "overall_mean_latency_ms": float(log_df["latency_ms"].mean()),
        "p95_latency_ms": float(log_df["latency_ms"].quantile(0.95)),
    }
    return summary


def run(config_path: str = "configs/params.yaml"):
    cfg = load_config(config_path)
    tau = cfg["train"]["quantile_tau"]
    threshold = cfg["monitoring"]["performance_threshold"]

    pred_path = cfg["data"]["processed"]["predictions"]
    print(f"Loading predictions from {pred_path}...")
    preds = pd.read_csv(pred_path, parse_dates=["date"])

    # In a real setup, actuals would come from new data.
    # For now, use validation data as a demonstration.
    master_path = cfg["data"]["processed"]["master"]
    master = pd.read_csv(master_path, parse_dates=["date"])
    actuals = master[master["date"] >= cfg["train"]["val_cutoff"]][
        ["rm_id", "date", "cumulative_weight"]
    ].copy()

    if "predicted_cumulative_weight" in preds.columns:
        preds = preds.rename(columns={"predicted_cumulative_weight": "cumulative_weight_pred"})

    metrics = evaluate_predictions(preds, actuals, tau)
    print(f"Performance metrics: {json.dumps(metrics, indent=2)}")

    retrain = check_retrain_needed(metrics, threshold)
    if retrain:
        ql = metrics.get('quantile_loss', 'N/A')
        print(f"⚠ Quantile loss ({ql}) > threshold ({threshold}). Retrain recommended.")
    else:
        print("Model performance is within acceptable range.")

    # Analyze prediction log if available
    pred_log_path = cfg["monitoring"].get("prediction_log", "data/processed/prediction_log.csv")
    pred_log_analysis = analyze_prediction_distribution(pred_log_path)
    if pred_log_analysis.get("status") == "ok":
        print(f"Prediction log: {pred_log_analysis['total_predictions']} predictions logged")
        print(f"  p95 latency: {pred_log_analysis['p95_latency_ms']:.1f}ms")
    else:
        print(f"Prediction log: {pred_log_analysis.get('status', 'unavailable')}")

    report = {
        "metrics": metrics,
        "retrain_recommended": retrain,
        "prediction_log_analysis": pred_log_analysis,
        "evaluated_at": datetime.now(UTC).isoformat(),
    }

    out_path = "data/processed/performance_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Performance report saved → {out_path}")

    return metrics


if __name__ == "__main__":
    run()
