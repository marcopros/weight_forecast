"""Integration tests for the full ML pipeline.

Tests end-to-end pipeline execution on a small fixture dataset,
data validation, and feature builder consistency.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.clean import aggregate_daily, build_master_table
from src.data.feature_builder import (
    build_lag_features,
    build_rm_stat_features,
    build_rolling_features,
    build_single_row,
    build_temporal_features,
)
from src.data.features import add_temporal_features, engineer_features
from src.models.train import quantile_loss


# ── Fixtures ────────────────────────────────────────────────────────────────
@pytest.fixture
def mini_receivals():
    """Small receivals dataset (2 rm_ids, 30 days)."""
    dates_1 = pd.date_range("2023-06-01", periods=30, freq="D")
    dates_2 = pd.date_range("2023-06-01", periods=30, freq="D")
    return pd.DataFrame({
        "rm_id": [1] * 30 + [2] * 30,
        "date": list(dates_1) + list(dates_2),
        "net_weight": list(np.random.uniform(100, 500, 30)) + list(np.random.uniform(50, 200, 30)),
        "date_arrival": list(dates_1) + list(dates_2),
    })


@pytest.fixture
def mini_master(mini_receivals):
    daily = aggregate_daily(mini_receivals)
    rm_ids = mini_receivals["rm_id"].unique()
    return build_master_table(daily, rm_ids, end_date="2023-07-15")


@pytest.fixture
def mini_cfg():
    return {
        "features": {
            "lag_days": [7, 28],
            "rolling_windows": [7],
            "target_col": "cumulative_weight",
            "use_material_features": False,
            "feature_cols": [
                "month", "day", "dayofweek", "dayofyear", "weekofyear",
                "quarter", "days_since_year_start",
                "lag_7d", "lag_28d",
                "rolling_mean_7d", "rolling_sum_7d",
                "rm_mean", "rm_std", "rm_median",
            ],
        },
        "train": {"val_cutoff": "2023-07-01", "quantile_tau": 0.2},
    }


# ── Integration: Pipeline end-to-end ───────────────────────────────────────
class TestPipelineIntegration:
    def test_clean_to_features(self, mini_master, mini_cfg):
        """Test that clean → features pipeline produces valid output."""
        featured, rm_stats = engineer_features(mini_master.copy(), mini_cfg)

        # All feature columns should exist
        for col in mini_cfg["features"]["feature_cols"]:
            assert col in featured.columns, f"Missing feature: {col}"

        # No infinite values
        numeric_cols = featured.select_dtypes(include=[np.number]).columns
        assert not featured[numeric_cols].isin([np.inf, -np.inf]).any().any()

    def test_cumulative_weight_monotonic(self, mini_master):
        """Cumulative weight must be non-decreasing within (rm_id, year)."""
        for _, grp in mini_master.groupby(["rm_id", "year"]):
            diffs = grp["cumulative_weight"].diff().dropna()
            assert (diffs >= -1e-6).all(), "Cumulative weight decreased!"

    def test_no_data_leakage_in_features(self, mini_master, mini_cfg):
        """Ensure lag/rolling features don't leak future information."""
        featured, _ = engineer_features(mini_master.copy(), mini_cfg)
        # For each rm_id, check that lag_7d at row i equals net_weight at row (i-7)
        for rm_id, grp in featured.groupby("rm_id"):
            grp = grp.sort_values("date").reset_index(drop=True)
            for i in range(7, len(grp)):
                lag_val = grp.loc[i, "lag_7d"]
                actual_val = grp.loc[i - 7, "net_weight"]
                if not np.isnan(lag_val):
                    assert abs(lag_val - actual_val) < 1e-6, \
                        f"Leakage at rm_id={rm_id}, row={i}: lag={lag_val} != actual={actual_val}"


# ── Feature builder consistency ─────────────────────────────────────────────
class TestFeatureBuilderConsistency:
    def test_temporal_features_match(self):
        """Unified builder should produce same temporal features as batch."""
        dt = pd.Timestamp("2024-06-15")
        single = build_temporal_features(dt)

        df = pd.DataFrame({"date": [dt]})
        df = add_temporal_features(df)
        batch = df.iloc[0]

        assert single["month"] == batch["month"]
        assert single["day"] == batch["day"]
        assert single["dayofweek"] == batch["dayofweek"]
        assert single["quarter"] == batch["quarter"]

    def test_lag_features_empty_history(self):
        """Lag features should default to 0 with empty history."""
        empty_hist = pd.DataFrame(columns=["date", "net_weight"])
        feats = build_lag_features(empty_hist, pd.Timestamp("2025-01-15"), [7, 28])
        assert feats["lag_7d"] == 0.0
        assert feats["lag_28d"] == 0.0

    def test_rolling_features_empty_history(self):
        """Rolling features should default to 0 with empty history."""
        empty_hist = pd.DataFrame(columns=["date", "net_weight"])
        feats = build_rolling_features(empty_hist, pd.Timestamp("2025-01-15"), [7])
        assert feats["rolling_mean_7d"] == 0.0
        assert feats["rolling_sum_7d"] == 0.0

    def test_rm_stats_unknown_rm(self):
        """Unknown rm_id should return zero stats."""
        rm_stats = pd.DataFrame(
            {"rm_id": [1], "rm_mean": [100.0],
             "rm_std": [10.0], "rm_median": [95.0]}
        )
        feats = build_rm_stat_features(rm_stats, 999)
        assert feats["rm_mean"] == 0.0

    def test_single_row_output_shape(self, mini_master, mini_cfg):
        """build_single_row should return exactly 1 row with all feature columns."""
        rm_stats = (
            mini_master.groupby("rm_id")["net_weight"]
            .agg([("rm_mean", "mean"), ("rm_std", "std"), ("rm_median", "median")])
            .reset_index()
        )
        rm_hist = mini_master[mini_master["rm_id"] == 1].sort_values("date")
        row = build_single_row(
            rm_id=1,
            target_date=pd.Timestamp("2023-07-10"),
            rm_history=rm_hist,
            rm_stats=rm_stats,
            cfg=mini_cfg,
        )
        assert row.shape[0] == 1
        assert list(row.columns) == mini_cfg["features"]["feature_cols"]


# ── Data validation tests ──────────────────────────────────────────────────
class TestDataValidation:
    def test_master_no_null_weights(self, mini_master):
        """Master table should have no null net_weight after cleaning."""
        assert mini_master["net_weight"].isna().sum() == 0

    def test_master_all_rm_ids_present(self, mini_receivals, mini_master):
        """All rm_ids from receivals should appear in master."""
        expected = set(mini_receivals["rm_id"].unique())
        actual = set(mini_master["rm_id"].unique())
        assert expected == actual

    def test_no_negative_cumulative(self, mini_master):
        """Cumulative weight should never be negative."""
        assert (mini_master["cumulative_weight"] >= -1e-6).all()


# ── Quantile loss edge cases ──────────────────────────────────────────────
class TestQuantileLossEdgeCases:
    def test_all_zeros(self):
        y = np.zeros(10)
        assert quantile_loss(y, y, 0.2) == 0.0

    def test_large_values(self):
        y_true = np.array([1e9])
        y_pred = np.array([1e9])
        assert quantile_loss(y_true, y_pred, 0.2) == pytest.approx(0.0)

    def test_single_element(self):
        loss = quantile_loss(np.array([100.0]), np.array([90.0]), 0.2)
        assert loss > 0
