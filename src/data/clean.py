"""Data loading and cleaning pipeline.

Reads raw CSVs, normalises types/dates, and writes cleaned versions to data/processed/.
Can be run standalone: python -m src.data.clean
"""

import os
import pandas as pd
import yaml


def load_config(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_receivals(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date_arrival"] = pd.to_datetime(df["date_arrival"], errors="coerce", utc=True)
    df = df.dropna(subset=["date_arrival"])
    df["date"] = df["date_arrival"].dt.tz_localize(None)
    return df


def load_prediction_mapping(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["forecast_end_date"] = pd.to_datetime(df["forecast_end_date"])
    if "forecast_start_date" in df.columns:
        df["forecast_start_date"] = pd.to_datetime(df["forecast_start_date"])
    return df


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

    out_path = cfg["data"]["processed"]["master"]
    master.to_csv(out_path, index=False)
    print(f"Master table saved → {out_path}  shape={master.shape}")
    return master


if __name__ == "__main__":
    run()
