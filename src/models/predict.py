"""Generate 2025 cumulative weight predictions and build the submission file.

Can be run standalone: python -m src.models.predict
"""

import joblib
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from src.data.clean import (
    load_prediction_mapping,
)
from src.data.features import _compute_features_for_rm, add_rm_stats, add_temporal_features


def load_config(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_test_features(
    master_df: pd.DataFrame,
    rm_stats: pd.DataFrame,
    unique_rm_ids,
    cfg: dict,
) -> pd.DataFrame:
    """Create the 2025 test grid, merge historical data, and compute features."""
    feat_cfg = cfg["features"]
    test_start = cfg["train"]["test_start"]
    test_end = cfg["train"]["test_end"]

    test_range = pd.date_range(start=test_start, end=test_end, freq="D")
    test_idx = pd.MultiIndex.from_product(
        [unique_rm_ids, test_range], names=["rm_id", "date"]
    )
    test_df = pd.DataFrame(index=test_idx).reset_index()
    test_df["net_weight"] = np.nan
    test_df["cumulative_weight"] = np.nan

    # Combine with historical data to compute lags
    combined = pd.concat([master_df, test_df], ignore_index=True)
    combined = combined.sort_values(["rm_id", "date"])
    combined["net_weight"] = combined["net_weight"].fillna(0)

    # Temporal features
    combined = add_temporal_features(combined)

    # Lag & rolling features
    print("Computing lag & rolling features for test set...")
    results = Parallel(n_jobs=-1, verbose=1)(
        delayed(_compute_features_for_rm)(
            group, feat_cfg["lag_days"], feat_cfg["rolling_windows"]
        )
        for _, group in combined.groupby("rm_id")
    )
    combined = pd.concat(results, ignore_index=True).sort_values(["rm_id", "date"])

    # Merge rm_stats (drop existing to avoid duplicates)
    stats_cols = ["rm_mean", "rm_std", "rm_median"]
    existing = [c for c in stats_cols if c in combined.columns]
    if existing:
        combined = combined.drop(columns=existing)
    combined["rm_id"] = combined["rm_id"].astype(rm_stats["rm_id"].dtype)
    combined = pd.merge(combined, rm_stats, on="rm_id", how="left")

    # Extract 2025 only
    final_test_df = combined[combined["year"] == 2025].copy()

    # Fill NaN in lag/rolling columns
    for col in feat_cfg["feature_cols"]:
        if col in final_test_df.columns and final_test_df[col].isna().any():
            final_test_df[col] = final_test_df[col].fillna(0)

    return final_test_df


def predict_and_submit(
    model,
    test_df: pd.DataFrame,
    prediction_mapping_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    feature_cols = cfg["features"]["feature_cols"]

    X_test = test_df[feature_cols].copy()
    preds = np.clip(model.predict(X_test), 0, None)

    test_df["predicted_cumulative_weight"] = preds
    test_df["rm_id"] = test_df["rm_id"].astype(int)

    # Monotonic constraint: cumulative must never decrease per rm_id
    test_df = test_df.sort_values(["rm_id", "date"])
    test_df["predicted_cumulative_weight"] = test_df.groupby("rm_id")[
        "predicted_cumulative_weight"
    ].cummax()

    # Build submission via prediction_mapping
    submission_df = prediction_mapping_df.copy()
    submission_df = pd.merge(
        submission_df,
        test_df[["rm_id", "date", "predicted_cumulative_weight"]],
        left_on=["rm_id", "forecast_end_date"],
        right_on=["rm_id", "date"],
        how="left",
    )
    submission_df["predicted_cumulative_weight"] = submission_df[
        "predicted_cumulative_weight"
    ].fillna(0)

    final = submission_df[["ID", "predicted_cumulative_weight"]].copy()
    final.columns = ["ID", "predicted_weight"]
    return final


def run(config_path: str = "configs/params.yaml"):
    cfg = load_config(config_path)
    model_path = cfg["data"]["processed"]["model"]
    master_path = cfg["data"]["processed"]["master"]
    pred_out = cfg["data"]["processed"]["predictions"]
    sub_out = cfg["data"]["processed"]["submission"]

    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    print(f"Loading master table from {master_path}...")
    master_df = pd.read_csv(master_path, parse_dates=["date"])

    # Reconstruct rm_stats from master
    _, rm_stats = add_rm_stats(
        master_df[["rm_id", "net_weight"]].copy()
    )
    # Need to drop rm_mean etc. from master_df if present
    stats_cols = ["rm_mean", "rm_std", "rm_median"]
    existing = [c for c in stats_cols if c in master_df.columns]
    if existing:
        master_df_clean = master_df.drop(columns=existing)
    else:
        master_df_clean = master_df

    unique_rm_ids = master_df["rm_id"].unique()

    print("Building test features...")
    test_df = build_test_features(master_df_clean, rm_stats, unique_rm_ids, cfg)

    print("Loading prediction mapping...")
    pm = load_prediction_mapping(cfg["data"]["raw"]["prediction_mapping"])

    print("Generating predictions...")
    submission = predict_and_submit(model, test_df, pm, cfg)

    submission.to_csv(sub_out, index=False)
    print(f"Submission saved → {sub_out}  ({len(submission)} rows)")

    # Save detailed predictions
    test_df[["rm_id", "date", "predicted_cumulative_weight"]].to_csv(
        pred_out, index=False
    )
    print(f"Detailed predictions saved → {pred_out}")

    return submission


if __name__ == "__main__":
    run()
