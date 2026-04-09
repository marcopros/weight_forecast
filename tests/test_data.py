"""Tests for the data cleaning and feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import aggregate_daily, build_master_table
from src.data.features import _compute_features_for_rm, add_temporal_features


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_receivals():
    """Minimal receivals-like DataFrame."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    rm_ids = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
    return pd.DataFrame(
        {
            "rm_id": rm_ids,
            "date": dates,
            "net_weight": np.random.uniform(100, 1000, size=10),
            "date_arrival": dates,
        }
    )


@pytest.fixture
def sample_daily(sample_receivals):
    return aggregate_daily(sample_receivals)


# ── Tests ───────────────────────────────────────────────────────────────────
class TestAggregate:
    def test_output_columns(self, sample_daily):
        assert "rm_id" in sample_daily.columns
        assert "date" in sample_daily.columns
        assert "net_weight" in sample_daily.columns
        assert "cumulative_weight" in sample_daily.columns

    def test_cumulative_monotonic(self, sample_daily):
        """Cumulative weight must be non-decreasing within each (rm_id, year)."""
        for _, group in sample_daily.groupby(["rm_id", "year"]):
            diffs = group["cumulative_weight"].diff().dropna()
            assert (diffs >= 0).all()

    def test_no_negative_weights(self, sample_daily):
        assert (sample_daily["net_weight"] >= 0).all()


class TestMasterTable:
    def test_full_date_grid(self, sample_daily):
        rm_ids = sample_daily["rm_id"].unique()
        master = build_master_table(sample_daily, rm_ids, end_date="2023-01-15")
        # Every rm_id should have the same number of dates
        counts = master.groupby("rm_id").size()
        assert counts.nunique() == 1

    def test_no_nan_net_weight(self, sample_daily):
        rm_ids = sample_daily["rm_id"].unique()
        master = build_master_table(sample_daily, rm_ids, end_date="2023-01-15")
        assert master["net_weight"].isna().sum() == 0


class TestFeatures:
    def test_temporal_features_added(self, sample_daily):
        rm_ids = sample_daily["rm_id"].unique()
        master = build_master_table(sample_daily, rm_ids, end_date="2023-01-15")
        featured = add_temporal_features(master)
        for col in ["month", "day", "dayofweek", "dayofyear", "quarter", "days_since_year_start"]:
            assert col in featured.columns

    def test_lag_features(self):
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        df = pd.DataFrame({"date": dates, "net_weight": range(30), "cumulative_weight": range(30)})
        result = _compute_features_for_rm(df, lag_days=[7], rolling_windows=[7])
        assert "lag_7d" in result.columns
        assert "rolling_mean_7d" in result.columns
        # First 7 values of lag should be NaN
        assert result["lag_7d"].isna().sum() == 7

    def test_rolling_no_leakage(self):
        """Rolling features must be shifted by 1 to avoid data leakage."""
        dates = pd.date_range("2023-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "date": dates, "net_weight": [0] * 9 + [100], "cumulative_weight": range(10),
        })
        result = _compute_features_for_rm(df, lag_days=[], rolling_windows=[7])
        # The last row's rolling stats should NOT include day 10's net_weight=100
        # (because of shift(1))
        assert result["rolling_sum_7d"].iloc[-1] == 0.0
