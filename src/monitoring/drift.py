"""Data drift detection using Evidently AI.

Compares production feature distributions against the training reference.
Can be run standalone: python -m src.monitoring.drift
"""

import json
import pandas as pd
import yaml

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False


def load_config(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Run Evidently data drift detection on the given feature columns.

    Returns a dict with 'dataset_drift' (bool) and per-column results.
    """
    if not EVIDENTLY_AVAILABLE:
        print("WARNING: evidently not installed. Falling back to basic stats comparison.")
        return _basic_drift_check(reference_df, production_df, feature_cols)

    ref = reference_df[feature_cols].copy()
    prod = production_df[feature_cols].copy()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=prod)
    result = report.as_dict()

    drift_info = result["metrics"][0]["result"]
    return {
        "dataset_drift": drift_info.get("dataset_drift", False),
        "share_of_drifted_columns": drift_info.get("share_of_drifted_columns", 0),
        "number_of_drifted_columns": drift_info.get("number_of_drifted_columns", 0),
        "details": drift_info.get("drift_by_columns", {}),
    }


def _basic_drift_check(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Fallback drift check based on mean/std comparison (no evidently needed)."""
    drifted = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in production_df.columns:
            continue
        ref_mean = reference_df[col].mean()
        ref_std = reference_df[col].std()
        prod_mean = production_df[col].mean()
        # Flag drift if production mean is > 3 std from reference mean
        if ref_std > 0 and abs(prod_mean - ref_mean) > 3 * ref_std:
            drifted.append(col)

    return {
        "dataset_drift": len(drifted) > 0,
        "share_of_drifted_columns": len(drifted) / max(len(feature_cols), 1),
        "number_of_drifted_columns": len(drifted),
        "drifted_columns": drifted,
    }


def run(config_path: str = "configs/params.yaml"):
    cfg = load_config(config_path)
    feature_cols = cfg["features"]["feature_cols"]
    master_path = cfg["data"]["processed"]["master"]
    pred_path = cfg["data"]["processed"]["predictions"]

    print("Loading reference (training) data...")
    reference = pd.read_csv(master_path, parse_dates=["date"])
    reference = reference[reference["date"] < cfg["train"]["val_cutoff"]]

    print("Loading production (prediction) data...")
    production = pd.read_csv(master_path, parse_dates=["date"])
    production = production[production["date"] >= cfg["train"]["val_cutoff"]]

    # Only keep common feature cols that exist
    available = [c for c in feature_cols if c in reference.columns and c in production.columns]

    print(f"Checking drift on {len(available)} features...")
    result = check_drift(reference, production, available)

    print(f"Dataset drift: {result['dataset_drift']}")
    print(f"Drifted columns: {result['number_of_drifted_columns']}")

    out_path = "data/processed/drift_report.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Drift report saved → {out_path}")

    return result


if __name__ == "__main__":
    run()
