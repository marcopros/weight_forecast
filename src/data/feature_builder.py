"""Unified feature builder for training and serving.

Single source of truth for feature construction — used by:
- src/data/features.py (batch training)
- src/models/predict.py (batch prediction)
- src/api/main.py (online serving)
"""

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_temporal_features(dt: pd.Timestamp) -> dict:
    """Build temporal features for a single timestamp."""
    year_start = pd.Timestamp(year=dt.year, month=1, day=1)
    return {
        "month": dt.month,
        "day": dt.day,
        "dayofweek": dt.dayofweek,
        "dayofyear": dt.dayofyear,
        "weekofyear": int(dt.isocalendar().week),
        "quarter": dt.quarter,
        "days_since_year_start": (dt - year_start).days + 1,
    }


def build_lag_features(
    rm_history: pd.DataFrame,
    target_date: pd.Timestamp,
    lag_days: list[int],
) -> dict:
    """Compute lag features from historical data for a single date."""
    feats = {}
    for lag in lag_days:
        lookup_date = target_date - pd.Timedelta(days=lag)
        match = rm_history[rm_history["date"] == lookup_date]
        feats[f"lag_{lag}d"] = float(match["net_weight"].iloc[0]) if not match.empty else 0.0
    return feats


def build_rolling_features(
    rm_history: pd.DataFrame,
    target_date: pd.Timestamp,
    rolling_windows: list[int],
) -> dict:
    """Compute rolling window features from historical data for a single date."""
    feats = {}
    for w in rolling_windows:
        window_end = target_date - pd.Timedelta(days=1)  # shift by 1 to avoid leakage
        window_start = window_end - pd.Timedelta(days=w - 1)
        window_data = rm_history[
            (rm_history["date"] >= window_start) & (rm_history["date"] <= window_end)
        ]["net_weight"]
        feats[f"rolling_mean_{w}d"] = float(window_data.mean()) if len(window_data) > 0 else 0.0
        feats[f"rolling_sum_{w}d"] = float(window_data.sum()) if len(window_data) > 0 else 0.0
    return feats


def build_rm_stat_features(rm_stats: pd.DataFrame, rm_id: int) -> dict:
    """Look up pre-computed material-level aggregate stats."""
    rm_row = rm_stats[rm_stats["rm_id"] == rm_id]
    if not rm_row.empty:
        return {
            "rm_mean": float(rm_row["rm_mean"].iloc[0]),
            "rm_std": float(rm_row["rm_std"].iloc[0]),
            "rm_median": float(rm_row["rm_median"].iloc[0]),
        }
    return {"rm_mean": 0.0, "rm_std": 0.0, "rm_median": 0.0}


def build_material_features(material_info: pd.DataFrame | None, rm_id: int) -> dict:
    """Look up material-level features from extended data (alloy, format type)."""
    if material_info is None or material_info.empty:
        return {"alloy_encoded": 0.0, "format_type": 0.0}
    row = material_info[material_info["rm_id"] == rm_id]
    if row.empty:
        return {"alloy_encoded": 0.0, "format_type": 0.0}
    return {
        "alloy_encoded": float(row["alloy_encoded"].iloc[0]),
        "format_type": float(row["format_type"].iloc[0]),
    }


def build_single_row(
    rm_id: int,
    target_date: pd.Timestamp,
    rm_history: pd.DataFrame,
    rm_stats: pd.DataFrame,
    cfg: dict,
    material_info: pd.DataFrame | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Build a single feature row for one (rm_id, date) pair.

    This is the canonical feature builder used by both batch and online inference.
    """
    feat_cfg = cfg["features"]
    if feature_cols is None:
        feature_cols = feat_cfg["feature_cols"]

    row = build_temporal_features(pd.Timestamp(target_date))
    row.update(build_lag_features(rm_history, pd.Timestamp(target_date), feat_cfg["lag_days"]))
    row.update(build_rolling_features(rm_history, pd.Timestamp(target_date), feat_cfg["rolling_windows"]))
    row.update(build_rm_stat_features(rm_stats, rm_id))

    if cfg["features"].get("use_material_features", False):
        row.update(build_material_features(material_info, rm_id))

    # Ensure all expected columns exist
    for col in feature_cols:
        if col not in row:
            row[col] = 0.0

    return pd.DataFrame([row])[feature_cols]
