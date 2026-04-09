"""Tests for the FastAPI serving endpoint."""

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
            ],
            "lag_days": [7, 28, 56],
            "rolling_windows": [7, 28],
        },
        "data": {"processed": {"model": "", "master": ""}},
    }

    with patch("src.api.main.MODEL", mock_model), \
         patch("src.api.main.MASTER_DF", mock_master), \
         patch("src.api.main.RM_STATS", mock_rm_stats), \
         patch("src.api.main.CFG", mock_cfg):
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
        assert data["rm_id"] == 1
        assert data["cumulative_weight"] >= 0

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
