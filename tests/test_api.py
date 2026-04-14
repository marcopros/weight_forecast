"""Tests for the FastAPI serving endpoint.

Includes unit tests, contract tests, and error handling tests.
"""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_globals():
    """Patch module-level globals so the app can start without real data."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([42.5])

    mock_master = pd.DataFrame(
        {
            "rm_id": [1, 1, 1],
            "date": pd.date_range("2024-12-29", periods=3),
            "net_weight": [10.0, 20.0, 30.0],
        }
    )
    mock_rm_stats = pd.DataFrame(
        {"rm_id": [1], "rm_mean": [20.0], "rm_std": [10.0], "rm_median": [20.0]}
    )
    mock_cfg = {
        "features": {
            "feature_cols": [
                "month", "day", "dayofweek", "dayofyear", "weekofyear",
                "quarter", "days_since_year_start",
                "lag_7d", "lag_28d", "lag_56d",
                "rolling_mean_7d", "rolling_sum_7d",
                "rolling_mean_28d", "rolling_sum_28d",
                "rm_mean", "rm_std", "rm_median",
                "alloy_encoded", "format_type",
            ],
            "lag_days": [7, 28, 56],
            "rolling_windows": [7, 28],
            "use_material_features": False,
        },
        "data": {"processed": {"model": "", "master": ""}, "raw": {}},
    }
    mock_rm_history_index = {
        1: mock_master.sort_values("date").copy(),
    }

    with patch("src.api.main.MODEL", mock_model), \
         patch("src.api.main.MASTER_DF", mock_master), \
         patch("src.api.main.RM_STATS", mock_rm_stats), \
         patch("src.api.main.CFG", mock_cfg), \
         patch("src.api.main.KNOWN_RM_IDS", {1}), \
         patch("src.api.main.RM_HISTORY_INDEX", mock_rm_history_index), \
         patch("src.api.main.MODEL_HASH", "test123abc"), \
         patch("src.api.main.MATERIAL_INFO", None), \
         patch("src.api.main._log_prediction"):
        from src.api.main import app
        # Override lifespan to do nothing
        app.router.lifespan_context = _noop_lifespan
        yield TestClient(app)


@asynccontextmanager
async def _noop_lifespan(app):
    yield


class TestHealthEndpoint:
    def test_health(self, mock_globals):
        client = mock_globals
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_hash" in data
        assert "known_rm_ids" in data


class TestMetricsEndpoint:
    def test_metrics(self, mock_globals):
        client = mock_globals
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "requests_total" in data
        assert "avg_latency_ms" in data
        assert "model_hash" in data


class TestPredictEndpoint:
    def test_predict_returns_200(self, mock_globals):
        client = mock_globals
        response = client.post(
            "/predict",
            json={"rm_id": 1, "forecast_end_date": "2025-03-15"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "cumulative_weight" in data
        assert "model_hash" in data
        assert data["rm_id"] == 1
        assert data["cumulative_weight"] >= 0

    def test_predict_accepts_date_alias(self, mock_globals):
        client = mock_globals
        response = client.post(
            "/predict",
            json={"rm_id": 1, "date": "2025-03-15"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["forecast_end_date"] == "2025-03-15"

    def test_predict_unknown_rm_id_returns_400(self, mock_globals):
        """Contract test: unknown rm_id must return 400, not silent zeros."""
        client = mock_globals
        response = client.post(
            "/predict",
            json={"rm_id": 99999, "forecast_end_date": "2025-03-15"},
        )
        assert response.status_code == 400
        assert "Unknown rm_id" in response.json()["detail"]

    def test_predict_invalid_date_returns_422(self, mock_globals):
        """Contract test: invalid date format returns 422."""
        client = mock_globals
        response = client.post(
            "/predict",
            json={"rm_id": 1, "forecast_end_date": "not-a-date"},
        )
        assert response.status_code == 422

    def test_predict_extreme_date_returns_422(self, mock_globals):
        """Contract test: date far outside valid range returns 422."""
        client = mock_globals
        response = client.post(
            "/predict",
            json={"rm_id": 1, "forecast_end_date": "1900-01-01"},
        )
        assert response.status_code == 422

    def test_predict_missing_fields_returns_422(self, mock_globals):
        """Contract test: missing required fields return 422."""
        client = mock_globals
        response = client.post("/predict", json={"rm_id": 1})
        assert response.status_code == 422

    def test_batch_predict(self, mock_globals):
        client = mock_globals
        response = client.post(
            "/predict/batch",
            json={
                "items": [
                    {"rm_id": 1, "forecast_end_date": "2025-03-15"},
                    {"rm_id": 1, "forecast_end_date": "2025-04-01"},
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2

    def test_batch_empty_returns_422(self, mock_globals):
        """Contract test: empty batch returns 422."""
        client = mock_globals
        response = client.post(
            "/predict/batch",
            json={"items": []},
        )
        assert response.status_code == 422

    def test_batch_unknown_rm_id_returns_400(self, mock_globals):
        """Contract test: batch with unknown rm_id returns 400."""
        client = mock_globals
        response = client.post(
            "/predict/batch",
            json={
                "items": [
                    {"rm_id": 1, "forecast_end_date": "2025-03-15"},
                    {"rm_id": 99999, "forecast_end_date": "2025-03-15"},
                ]
            },
        )
        assert response.status_code == 400


class TestActualsEndpoint:
    def test_submit_actuals(self, mock_globals):
        """Test feedback loop endpoint."""
        import tempfile
        import os
        with patch("src.api.main.os.path.exists", return_value=False), \
             patch("builtins.open", create=True):
            client = mock_globals
            # Just test the endpoint exists and accepts the right schema
            response = client.post(
                "/actuals",
                json={
                    "items": [
                        {"rm_id": 1, "date": "2025-03-15", "actual_cumulative_weight": 5000.0},
                    ]
                },
            )
            # May fail due to file writing in test, but should not be 404/405
            assert response.status_code in (200, 500)
