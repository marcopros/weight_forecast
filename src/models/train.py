"""Model training with MLflow experiment tracking.

Trains a LightGBM regressor on cumulative weight data.
Can be run standalone: python -m src.models.train
"""

import json

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
        if k != "early_stopping_rounds"
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


def run(config_path: str = "configs/params.yaml"):
    cfg = load_config(config_path)
    master_path = cfg["data"]["processed"]["master"]
    model_path = cfg["data"]["processed"]["model"]
    tau = cfg["train"]["quantile_tau"]

    print(f"Loading featured master table from {master_path}...")
    master_df = pd.read_csv(master_path, parse_dates=["date"])

    X_train, y_train, X_val, y_val = prepare_splits(master_df, cfg)

    # ── MLflow tracking ────────────────────────────────────────
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        mlflow.start_run()
        mlflow.log_params(cfg["model"])

    print("Training LightGBM...")
    model = train_model(X_train, y_train, X_val, y_val, cfg)
    print(f"Best iteration: {model.best_iteration_}")

    metrics = evaluate(model, X_val, y_val, tau)
    print(f"Validation Quantile Loss (τ={tau}): {metrics['quantile_loss']:.4f}")
    print(f"Validation MAE: {metrics['mae']:.2f}")
    print(f"Validation RMSE: {metrics['rmse']:.2f}")

    if MLFLOW_AVAILABLE:
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(
            model,
            "model",
            registered_model_name=cfg["mlflow"]["registered_model_name"],
        )
        mlflow.end_run()

    # Always save with joblib as well (used by API and DVC)
    joblib.dump(model, model_path)
    print(f"Model saved → {model_path}")

    # Save metrics for DVC
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved → metrics.json")

    return model, metrics


if __name__ == "__main__":
    run()
