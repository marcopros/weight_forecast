"""Model training with MLflow experiment tracking.

Trains a LightGBM regressor on cumulative weight data.
Includes baseline comparison, per-rm_id error analysis, and model versioning.
Can be run standalone: python -m src.models.train
"""

import hashlib
import json
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

try:
    import mlflow
    import mlflow.lightgbm

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def load_config(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def quantile_loss(y_true, y_pred, tau: float = 0.2) -> float:
    errors = y_true - y_pred
    return float(np.where(errors >= 0, tau * errors, -(1 - tau) * errors).mean())


def prepare_splits(
    master_df: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = cfg["features"]["feature_cols"]
    target_col = cfg["features"]["target_col"]
    val_cutoff = cfg["train"]["val_cutoff"]

    train_df = master_df[master_df["date"] < val_cutoff].copy()
    val_df = master_df[master_df["date"] >= val_cutoff].copy()

    train_df = train_df.dropna(subset=feature_cols + [target_col])
    val_df = val_df.dropna(subset=feature_cols + [target_col])

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    print(f"Training set:   {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    return X_train, y_train, X_val, y_val


# ── Baseline models ────────────────────────────────────────────────────────
def run_baselines(
    master_df: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.Series, cfg: dict
) -> dict:
    """Run naive baseline models and return their metrics for comparison."""
    tau = cfg["train"]["quantile_tau"]
    val_cutoff = cfg["train"]["val_cutoff"]
    val_df = master_df[master_df["date"] >= val_cutoff].copy()
    results = {}

    # Baseline 1: Global mean prediction
    train_mean = master_df[master_df["date"] < val_cutoff]["cumulative_weight"].mean()
    preds_mean = np.full(len(y_val), train_mean)
    results["global_mean"] = {
        "quantile_loss": quantile_loss(y_val.values, preds_mean, tau),
        "mae": float(np.abs(y_val.values - preds_mean).mean()),
    }

    # Baseline 2: Per-rm_id mean (last year same period)
    rm_means = (
        master_df[master_df["date"] < val_cutoff]
        .groupby("rm_id")["cumulative_weight"].mean()
    )
    preds_rm_mean = val_df["rm_id"].map(rm_means).fillna(train_mean).values
    results["per_rm_mean"] = {
        "quantile_loss": quantile_loss(y_val.values, preds_rm_mean, tau),
        "mae": float(np.abs(y_val.values - preds_rm_mean).mean()),
    }

    # Baseline 3: Last year carry-over (same day-of-year, same rm_id)
    train_df = master_df[master_df["date"] < val_cutoff].copy()
    train_df["dayofyear"] = train_df["date"].dt.dayofyear
    val_df_b = val_df.copy()
    val_df_b["dayofyear"] = val_df_b["date"].dt.dayofyear
    last_year = (
        train_df.sort_values("date")
        .groupby(["rm_id", "dayofyear"])
        .last()["cumulative_weight"]
    )
    preds_ly = val_df_b.set_index(["rm_id", "dayofyear"]).index.map(
        lambda x: last_year.get(x, train_mean)
    )
    preds_ly = np.array([float(x) for x in preds_ly])
    results["last_year_carryover"] = {
        "quantile_loss": quantile_loss(y_val.values, preds_ly, tau),
        "mae": float(np.abs(y_val.values - preds_ly).mean()),
    }

    # Baseline 4: Zero prediction (conservative)
    preds_zero = np.zeros(len(y_val))
    results["zero_prediction"] = {
        "quantile_loss": quantile_loss(y_val.values, preds_zero, tau),
        "mae": float(np.abs(y_val.values - preds_zero).mean()),
    }

    print("\n── Baseline Comparison ──")
    for name, m in results.items():
        print(f"  {name:25s}  QL={m['quantile_loss']:>12.2f}  MAE={m['mae']:>12.2f}")

    return results


# ── Per-rm_id error analysis ───────────────────────────────────────────────
def analyze_errors_per_rm(
    model, X_val: pd.DataFrame, y_val: pd.Series, val_df: pd.DataFrame, tau: float
) -> dict:
    """Break down model errors by rm_id to find problematic materials."""
    preds = np.clip(model.predict(X_val), 0, None)
    analysis_df = val_df[["rm_id", "date"]].copy()
    analysis_df = analysis_df.iloc[: len(preds)]
    analysis_df["actual"] = y_val.values
    analysis_df["predicted"] = preds
    analysis_df["error"] = analysis_df["actual"] - analysis_df["predicted"]
    analysis_df["abs_error"] = analysis_df["error"].abs()

    per_rm = analysis_df.groupby("rm_id").agg(
        n_samples=("error", "size"),
        mae=("abs_error", "mean"),
        mean_error=("error", "mean"),
        mean_actual=("actual", "mean"),
        mean_predicted=("predicted", "mean"),
    ).reset_index()

    per_rm["ql"] = per_rm.apply(
        lambda row: quantile_loss(
            analysis_df[analysis_df["rm_id"] == row["rm_id"]]["actual"].values,
            analysis_df[analysis_df["rm_id"] == row["rm_id"]]["predicted"].values,
            tau,
        ),
        axis=1,
    )
    per_rm["bias"] = np.where(per_rm["mean_error"] > 0, "underestimates", "overestimates")
    per_rm = per_rm.sort_values("ql", ascending=False)

    top_10_worst = per_rm.head(10).to_dict(orient="records")
    top_10_best = per_rm.tail(10).to_dict(orient="records")

    summary = {
        "total_rm_ids": int(per_rm["rm_id"].nunique()),
        "top_10_worst": top_10_worst,
        "top_10_best": top_10_best,
        "overestimating_rm_ids": int((per_rm["mean_error"] < 0).sum()),
        "underestimating_rm_ids": int((per_rm["mean_error"] > 0).sum()),
    }

    print("\n── Per-rm_id Error Analysis ──")
    print(f"  Total rm_ids evaluated: {summary['total_rm_ids']}")
    print(f"  Overestimating: {summary['overestimating_rm_ids']}")
    print(f"  Underestimating: {summary['underestimating_rm_ids']}")
    print(f"  Worst 3 rm_ids by QL: {[r['rm_id'] for r in top_10_worst[:3]]}")

    return summary


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cfg: dict,
) -> lgb.LGBMRegressor:
    model_params = {
        k: v
        for k, v in cfg["model"].items()
        if k not in ("early_stopping_rounds",)
    }
    early_stopping_rounds = cfg["model"]["early_stopping_rounds"]

    model = lgb.LGBMRegressor(**model_params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model


def evaluate(model, X_val, y_val, tau: float = 0.2) -> dict:
    preds = np.clip(model.predict(X_val), 0, None)
    errors = y_val.values - preds
    metrics = {
        "quantile_loss": quantile_loss(y_val.values, preds, tau),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
    }
    return metrics


def compute_model_hash(model_path: str) -> str:
    """Compute SHA-256 hash of the serialized model for versioning."""
    sha = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()[:12]


def get_feature_importance(model, feature_cols: list[str]) -> dict:
    """Extract and rank feature importances."""
    importances = model.feature_importances_
    pairs = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    return {name: int(imp) for name, imp in pairs}


def run(config_path: str = "configs/params.yaml"):
    cfg = load_config(config_path)
    master_path = cfg["data"]["processed"]["master"]
    model_path = cfg["data"]["processed"]["model"]
    tau = cfg["train"]["quantile_tau"]

    print(f"Loading featured master table from {master_path}...")
    master_df = pd.read_csv(master_path, parse_dates=["date"])

    X_train, y_train, X_val, y_val = prepare_splits(master_df, cfg)

    # ── Baselines ──────────────────────────────────────────────
    baseline_results = {}
    if cfg["train"].get("run_baselines", True):
        baseline_results = run_baselines(master_df, X_val, y_val, cfg)

    # ── MLflow tracking ────────────────────────────────────────
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        mlflow.start_run()
        mlflow.log_params(cfg["model"])

    print("Training LightGBM...")
    train_start = time.time()
    model = train_model(X_train, y_train, X_val, y_val, cfg)
    train_duration = time.time() - train_start
    print(f"Best iteration: {model.best_iteration_} (trained in {train_duration:.1f}s)")

    metrics = evaluate(model, X_val, y_val, tau)
    print(f"Validation Quantile Loss (τ={tau}): {metrics['quantile_loss']:.4f}")
    print(f"Validation MAE: {metrics['mae']:.2f}")
    print(f"Validation RMSE: {metrics['rmse']:.2f}")

    # Feature importance
    feature_importance = get_feature_importance(model, cfg["features"]["feature_cols"])
    print("\n── Feature Importance (top 5) ──")
    for name, imp in list(feature_importance.items())[:5]:
        print(f"  {name:25s}  {imp}")

    # Per-rm_id error analysis
    val_df = master_df[master_df["date"] >= cfg["train"]["val_cutoff"]].dropna(
        subset=cfg["features"]["feature_cols"] + [cfg["features"]["target_col"]]
    )
    error_analysis = analyze_errors_per_rm(model, X_val, y_val, val_df, tau)

    # ── Model versioning ───────────────────────────────────────
    joblib.dump(model, model_path)
    model_hash = compute_model_hash(model_path)
    print(f"\nModel saved → {model_path}  (hash: {model_hash})")

    # Save comprehensive metrics
    full_metrics = {
        "quantile_loss": metrics["quantile_loss"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "model_hash": model_hash,
        "best_iteration": model.best_iteration_,
        "train_duration_seconds": round(train_duration, 1),
        "n_features": len(cfg["features"]["feature_cols"]),
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "feature_importance": feature_importance,
        "baselines": baseline_results,
        "error_analysis": error_analysis,
    }

    if MLFLOW_AVAILABLE:
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(
            {f"baseline_{k}_ql": v["quantile_loss"] for k, v in baseline_results.items()}
        )
        mlflow.log_dict(feature_importance, "feature_importance.json")
        mlflow.log_dict(error_analysis, "error_analysis.json")
        mlflow.set_tag("model_hash", model_hash)
        mlflow.lightgbm.log_model(
            model,
            "model",
            registered_model_name=cfg["mlflow"]["registered_model_name"],
        )
        mlflow.end_run()

    with open("metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    print("Metrics saved → metrics.json")

    # Save error analysis separately
    error_path = cfg["data"]["processed"].get(
        "error_analysis", "data/processed/error_analysis.json"
    )
    with open(error_path, "w") as f:
        json.dump(error_analysis, f, indent=2, default=str)
    print(f"Error analysis saved → {error_path}")

    return model, full_metrics


if __name__ == "__main__":
    run()
