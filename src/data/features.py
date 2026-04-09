"""Feature engineering for cumulative weight forecasting.

Adds temporal, lag, rolling, and material-level aggregate features.
Can be run standalone: python -m src.data.features
"""

import pandas as pd
import yaml
from joblib import Parallel, delayed


def load_config(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Temporal features ───────────────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["days_since_year_start"] = (
        df["date"] - pd.to_datetime(df["year"].astype(str) + "-01-01")
    ).dt.days + 1
    return df


# ── Lag & rolling features (per rm_id) ─────────────────────────────────────
def _compute_features_for_rm(
    rm_data: pd.DataFrame,
    lag_days: list[int],
    rolling_windows: list[int],
) -> pd.DataFrame:
    rm_data = rm_data.sort_values("date").copy()

    # Lag on daily net_weight
    for lag in lag_days:
        rm_data[f"lag_{lag}d"] = rm_data["net_weight"].shift(lag)

    # Rolling on daily net_weight (shifted by 1 to avoid leakage)
    for window in rolling_windows:
        rolling_obj = rm_data["net_weight"].shift(1).rolling(window, min_periods=1)
        rm_data[f"rolling_mean_{window}d"] = rolling_obj.mean()
        rm_data[f"rolling_sum_{window}d"] = rolling_obj.sum()

    return rm_data


def add_lag_rolling_features(
    df: pd.DataFrame,
    lag_days: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    if lag_days is None:
        lag_days = [7, 28, 56]
    if rolling_windows is None:
        rolling_windows = [7, 28]

    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_compute_features_for_rm)(group, lag_days, rolling_windows)
        for _, group in df.groupby("rm_id")
    )
    return pd.concat(results, ignore_index=True).sort_values(["rm_id", "date"])


# ── Material-level aggregate stats ─────────────────────────────────────────
def add_rm_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rm_stats = (
        df.groupby("rm_id")["net_weight"]
        .agg([("rm_mean", "mean"), ("rm_std", "std"), ("rm_median", "median")])
        .reset_index()
    )
    df = pd.merge(df, rm_stats, on="rm_id", how="left")
    return df, rm_stats


# ── Full pipeline ──────────────────────────────────────────────────────────
def engineer_features(
    master_df: pd.DataFrame,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full feature engineering. Returns (featured_df, rm_stats)."""
    if cfg is None:
        cfg = load_config()

    feat_cfg = cfg["features"]

    print("Adding temporal features...")
    master_df = add_temporal_features(master_df)

    print("Adding lag & rolling features...")
    master_df = add_lag_rolling_features(
        master_df,
        lag_days=feat_cfg["lag_days"],
        rolling_windows=feat_cfg["rolling_windows"],
    )

    print("Adding rm_id aggregate stats...")
    master_df, rm_stats = add_rm_stats(master_df)

    return master_df, rm_stats


def run(config_path: str = "configs/params.yaml"):
    cfg = load_config(config_path)
    master_path = cfg["data"]["processed"]["master"]

    print(f"Loading master table from {master_path}...")
    master_df = pd.read_csv(master_path, parse_dates=["date"])

    master_df, rm_stats = engineer_features(master_df, cfg)

    master_df.to_csv(master_path, index=False)
    print(f"Featured master table saved → {master_path}  shape={master_df.shape}")
    return master_df, rm_stats


if __name__ == "__main__":
    run()
