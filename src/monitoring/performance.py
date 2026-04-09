"""Performance monitoring: track predictions vs actuals over time.

Can be run standalone: python -m src.monitoring.performance
"""

import json

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
    """Compare predictions against actuals. Returns performance metrics.

    Both DataFrames must have columns: rm_id, date (or forecast_end_date), and value column.
    """
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

    out_path = "data/processed/performance_report.json"
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics, "retrain_recommended": retrain}, f, indent=2)
    print(f"Performance report saved → {out_path}")

    return metrics


if __name__ == "__main__":
    run()
