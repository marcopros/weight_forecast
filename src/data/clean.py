"""Data loading and cleaning pipeline.

Reads raw CSVs, normalises types/dates, validates data quality,
and writes cleaned versions to data/processed/.
Can be run standalone: python -m src.data.clean
"""

import os

import pandas as pd
import yaml

from src.data.validate import validate_master, validate_receivals


def load_config(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_receivals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date_arrival"] = pd.to_datetime(df["date_arrival"], errors="coerce", utc=True)
    df = df.dropna(subset=["date_arrival", "rm_id", "net_weight"])
    df["rm_id"] = df["rm_id"].astype(int)
    df["date"] = df["date_arrival"].dt.tz_localize(None)
    # Filter out negative weights
    df = df[df["net_weight"] >= 0]
    validate_receivals(df)
    return df


def load_prediction_mapping(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["forecast_end_date"] = pd.to_datetime(df["forecast_end_date"])
    if "forecast_start_date" in df.columns:
        df["forecast_start_date"] = pd.to_datetime(df["forecast_start_date"])
    return df


def load_materials(path: str) -> pd.DataFrame:
    """Load and encode material features (alloy type, format type)."""
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["rm_id"])
    df["rm_id"] = df["rm_id"].astype(int)

    # Encode alloy as numeric (label encoding)
    if "raw_material_alloy" in df.columns:
        alloy_map = {v: i for i, v in enumerate(df["raw_material_alloy"].dropna().unique())}
        df["alloy_encoded"] = df["raw_material_alloy"].map(alloy_map).fillna(-1).astype(float)
    else:
        df["alloy_encoded"] = 0.0

    # Format type (already numeric or needs encoding)
    if "raw_material_format_type" in df.columns:
        df["format_type"] = pd.to_numeric(df["raw_material_format_type"], errors="coerce").fillna(0.0)
    else:
        df["format_type"] = 0.0

    # Keep one row per rm_id (take first)
    df = df.groupby("rm_id").first().reset_index()
    return df[["rm_id", "alloy_encoded", "format_type"]]


def aggregate_daily(receivals_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate net_weight per (rm_id, date) and compute annual cumulative weight."""
    daily = (
        receivals_df.groupby(["rm_id", receivals_df["date"].dt.date])["net_weight"]
        .sum()
        .reset_index()
    )
    daily.columns = ["rm_id", "date", "net_weight"]
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily = daily.sort_values(["rm_id", "date"])
    daily["cumulative_weight"] = daily.groupby(["rm_id", "year"])["net_weight"].cumsum()
    return daily


def build_master_table(
    daily_receivals: pd.DataFrame,
    unique_rm_ids,
    end_date: str = "2024-12-31",
) -> pd.DataFrame:
    """Create a full (rm_id × date) grid and merge with daily receivals."""
    start_date = daily_receivals["date"].min()
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    multi_index = pd.MultiIndex.from_product(
        [unique_rm_ids, date_range], names=["rm_id", "date"]
    )
    master = pd.DataFrame(index=multi_index).reset_index()

    master = pd.merge(
        master,
        daily_receivals[["rm_id", "date", "net_weight", "cumulative_weight"]],
        on=["rm_id", "date"],
        how="left",
    )
    master["net_weight"] = master["net_weight"].fillna(0)

    # Recompute cumulative for all days (including zero-delivery days)
    master["year"] = master["date"].dt.year
    master = master.sort_values(["rm_id", "date"])
    master["cumulative_weight"] = master.groupby(["rm_id", "year"])[
        "net_weight"
    ].cumsum()

    return master


def run(config_path: str = "configs/params.yaml"):
    cfg = load_config(config_path)
    os.makedirs(cfg["data"]["processed"]["dir"], exist_ok=True)

    print("Loading receivals...")
    receivals_df = load_receivals(cfg["data"]["raw"]["receivals"])

    print("Aggregating daily...")
    daily = aggregate_daily(receivals_df)
    unique_rm_ids = receivals_df["rm_id"].unique()

    print("Building master table...")
    master = build_master_table(daily, unique_rm_ids)

    # Merge material features if available
    if cfg["features"].get("use_material_features", False):
        materials_path = cfg["data"]["raw"].get("materials")
        if materials_path and os.path.exists(materials_path):
            print("Loading material features...")
            materials = load_materials(materials_path)
            master = pd.merge(master, materials, on="rm_id", how="left")
            master["alloy_encoded"] = master["alloy_encoded"].fillna(0.0)
            master["format_type"] = master["format_type"].fillna(0.0)
            print(f"  Merged {len(materials)} material records")

    # Validate output
    validate_master(master)
    print("✓ Master table passed validation")

    out_path = cfg["data"]["processed"]["master"]
    master.to_csv(out_path, index=False)
    print(f"Master table saved → {out_path}  shape={master.shape}")
    return master


if __name__ == "__main__":
    run()
