# Cumulative Weight Forecast

End-to-end MLOps system for forecasting cumulative raw-material delivery weights at Hydro ASA aluminium smelters. Built with **LightGBM**, **FastAPI**, **DVC**, **Evidently AI**, and **Docker** — from data ingestion to production serving and continuous monitoring.

> **Competition metric**: Asymmetric Quantile Loss (τ = 0.2) — overestimation is penalised 4× more than underestimation, reflecting real logistics costs where excess inventory is more expensive than a shortfall.

---

## Results at a Glance

| Metric | Value |
|---|---|
| **Quantile Loss (τ=0.2)** | 29,389 |
| **MAE** | 128,673 |
| **RMSE** | 703,108 |
| **vs. best baseline** | **−33.3 %** (zero-prediction: 44,082) |
| **Model** | LightGBM · 19 features · 85 boosting rounds |
| **Serving latency** | < 5 ms / request (single row) |

### Baseline Comparison

| Model | Quantile Loss | MAE |
|---|---|---|
| **LightGBM (ours)** | **29,389** | **128,673** |
| Zero prediction | 44,082 | 220,412 |
| Last-year carryover | 62,188 | 126,610 |
| Per-rm_id mean | 144,291 | 304,979 |
| Global mean | 171,132 | 358,660 |

### Top-5 Features

| Feature | Importance |
|---|---|
| `rm_mean` | 533 |
| `alloy_encoded` | 447 |
| `rm_std` | 360 |
| `dayofyear` | 359 |
| `format_type` | 273 |

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                           Data Pipeline (DVC)                         │
│  receivals.csv ──► clean ──► features ──► validate ──► train ──► predict │
│                    │                        │            │            │
│                    ▼                        ▼            ▼            ▼
│              master_for_modeling.csv    pandera      model.joblib  submission.csv
└───────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────┐
│                    Production Serving                    │
│  FastAPI :8000           Streamlit Dashboard :8501       │
│  /predict  /predict/batch    Model Performance           │
│  /metrics  /actuals          Data Drift                  │
│  /health   /docs             Baseline Comparison         │
│                              Feature Importance           │
│                              Per-rm_id Error Analysis     │
└─────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────┐
│                   CI/CD (GitHub Actions)                 │
│  lint ──► test ──► validate data ──► train ──►          │
│  metric gate ──► drift check ──► security scan ──►      │
│  Docker build & push (GHCR, tagged by model hash)       │
└─────────────────────────────────────────────────────────┘
```

### Source Layout

```
src/
├── data/
│   ├── clean.py            Data loading, cleaning, aggregation
│   ├── features.py         Batch feature engineering (lag, rolling, stats)
│   ├── feature_builder.py  Unified builder (batch + online — no skew)
│   └── validate.py         Pandera schemas for data quality gates
├── models/
│   ├── train.py            Training, baselines, error analysis, MLflow
│   └── predict.py          Batch inference & competition submission
├── api/
│   └── main.py             FastAPI: predict, batch, metrics, actuals
└── monitoring/
    ├── drift.py            Evidently AI feature-distribution drift
    ├── performance.py      Quantile-loss tracking & retrain alerts
    └── dashboard.py        Streamlit UI (5 monitoring sections)
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerised deployment)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
make all    # clean → features → validate → train → predict
```

Or step by step:

```bash
python -m src.data.clean       # Load & clean receivals, build master table
python -m src.data.features    # Temporal, lag, rolling, material features
python -m src.data.validate    # Pandera schema checks
python -m src.models.train     # Train LightGBM + baselines + error analysis
python -m src.models.predict   # Generate 2025 predictions & submission
```

### 3. Start the API

```bash
make serve   # → http://localhost:8000
```

### 4. Make a Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rm_id": 342, "date": "2025-03-15"}'
```

```json
{
  "rm_id": 342,
  "forecast_end_date": "2025-03-15",
  "cumulative_weight": 0.0,
  "model_hash": "c6c0b5046041"
}
```

Batch predictions, interactive docs, and more at [`/docs`](http://localhost:8000/docs).

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Readiness check, model hash, rm_id count |
| `/metrics` | GET | Prometheus-style counters (requests, latency, errors) |
| `/predict` | POST | Single prediction (accepts `date` or `forecast_end_date`) |
| `/predict/batch` | POST | Batch predictions (up to 1,000 items) |
| `/actuals` | POST | Feedback loop — submit actual weights for monitoring |
| `/docs` | GET | Interactive Swagger UI |

### Request Schema

```jsonc
// Single
{"rm_id": 342, "date": "2025-03-15"}

// Batch
{"items": [
  {"rm_id": 342, "date": "2025-03-15"},
  {"rm_id": 3865, "date": "2025-06-01"}
]}
```

### Error Handling

| Status | Condition |
|---|---|
| 400 | Unknown `rm_id` |
| 422 | Invalid date format, date outside 2020–2030, missing fields, empty batch |

---

## Pipeline (DVC)

```bash
dvc repro    # Runs only stages with changed dependencies
```

**Stages**: `clean → features → validate → train → predict → monitor_drift → monitor_performance`

DVC tracks data lineage and ensures reproducibility. The pipeline config is in `dvc.yaml`; all hyperparameters live in `configs/params.yaml`.

---

## Model

| | |
|---|---|
| **Algorithm** | LightGBM (gradient-boosted trees) |
| **Objective** | `quantile` with α = 0.2 |
| **Target** | Cumulative weight from Jan 1 per (rm_id, year) |
| **Features (19)** | Temporal (7) · Lag (3) · Rolling (4) · rm_id stats (3) · Material (2) |
| **Validation** | Temporal split — train < 2024-01-01, val ≥ 2024-01-01 |
| **Post-processing** | `max(prediction, 0)` — clip negatives |
| **Experiment tracking** | MLflow (local `mlruns/` directory) |

### Feature Groups

| Group | Features | Description |
|---|---|---|
| Temporal | `month`, `day`, `dayofweek`, `dayofyear`, `weekofyear`, `quarter`, `days_since_year_start` | Calendar decomposition |
| Lag | `lag_7d`, `lag_28d`, `lag_56d` | Past cumulative weight at fixed offsets |
| Rolling | `rolling_mean_7d`, `rolling_sum_7d`, `rolling_mean_28d`, `rolling_sum_28d` | Smoothed recent history |
| rm_id Stats | `rm_mean`, `rm_std`, `rm_median` | Per-material historical aggregates |
| Material | `alloy_encoded`, `format_type` | Alloy composition & physical format |

### Training–Serving Consistency

A single `feature_builder.py` module is used by both batch training (`features.py`) and online serving (`main.py`), eliminating training–serving skew.

---

## Data Validation

[Pandera](https://pandera.readthedocs.io/) schemas enforce data quality at every stage:

| Schema | Checks |
|---|---|
| `receivals_schema` | `rm_id` non-null int, `net_weight` ≥ 0, `date_arrival` present |
| `master_schema` | All core columns present and typed |
| `featured_schema` | All 19 feature columns present |
| `prediction_schema` | Predictions non-negative, valid rm_ids |

Run standalone:

```bash
make validate
```

---

## Monitoring

### Data Drift

```bash
python -m src.monitoring.drift
```

Uses **Evidently AI** to compare training-set and recent feature distributions column by column. Outputs `data/processed/drift_report.json`.

### Model Performance

```bash
python -m src.monitoring.performance
```

Computes quantile loss against ground truth, analyses API prediction logs (latency p95, volume), and triggers retrain alerts when loss exceeds the configured threshold.

### Dashboard

```bash
make dashboard    # → http://localhost:8501
```

Five monitoring sections:

1. **Model Performance** — QL, MAE, RMSE, retrain alerts, API latency stats
2. **Data Drift** — Column-level drift detection
3. **Predictions Overview** — Daily aggregates, per-rm_id time series
4. **Training Metrics** — Baseline comparison chart, feature importance, model hash & iteration
5. **Error Analysis** — Worst/best rm_ids by quantile loss, over/underestimation breakdown

---

## Docker

### Quick Start (API + Dashboard)

```bash
cd docker && docker-compose up --build -d
```

| Service | Port | Description |
|---|---|---|
| `api` | 8000 | FastAPI prediction service |
| `dashboard` | 8501 | Streamlit monitoring UI |

The dashboard depends on the API health check (30 s interval). Both services mount the host `data/processed/` directory so pipeline outputs are visible in real time.

### Standalone API

```bash
docker build -f docker/Dockerfile.api -t forecast-api .
docker run -p 8000:8000 -v $(pwd)/data/processed:/app/data/processed forecast-api
```

### Resource Limits

| Service | Memory | CPUs |
|---|---|---|
| API | 2 GB | 2 |
| Dashboard | 1 GB | 1 |

---

## CI/CD

GitHub Actions workflow (`.github/workflows/ml_pipeline.yml`) runs on every push to `main` and on a weekly schedule (Monday 06:00 UTC).

### Pipeline

```
lint-and-test ──► train-and-evaluate ──► build-and-push
                         │
                  security-scan
```

| Job | Steps |
|---|---|
| **lint-and-test** | `ruff check` → `pytest` (40 tests) |
| **train-and-evaluate** | Full pipeline → monitoring → metric gate (QL < 200K) → drift check → upload artifacts |
| **security-scan** | `pip-audit` for dependency vulnerabilities |
| **build-and-push** | Docker image → GHCR, tagged by `latest`, `git-sha`, and `model-hash` |

The **metric gate** automatically blocks deployment if the quantile loss exceeds the safety threshold, preventing model regressions from reaching production.

---

## Testing

```bash
make test              # All tests (40 tests)
make test-integration  # Integration & consistency tests only
```

| Suite | Tests | Coverage |
|---|---|---|
| `test_data.py` | 8 | Data cleaning, feature engineering, loss function |
| `test_model.py` | 5 | Model loading, prediction shape, non-negativity |
| `test_api.py` | 12 | All endpoints, error handling, batch, alias support |
| `test_integration.py` | 13 | E2E pipeline, feature builder consistency, data validation |

---

## Configuration

All hyperparameters and paths are centralised in `configs/params.yaml`:

```yaml
# Key settings
train:
  val_cutoff: "2024-01-01"
  quantile_tau: 0.2
  run_baselines: true

model:
  objective: quantile
  alpha: 0.2
  n_estimators: 1000
  learning_rate: 0.05
  num_leaves: 31
  early_stopping_rounds: 100

monitoring:
  performance_threshold: 0.5
  latency_p95_threshold_ms: 100
```

---

## Project Structure

```
.
├── configs/params.yaml              All hyperparameters, paths, thresholds
├── data/
│   ├── kernel/                      Raw data (receivals, purchase_orders)
│   ├── extended/                    Supplementary data (materials, transportation)
│   └── processed/                   Pipeline outputs (model, predictions, reports)
├── docker/
│   ├── Dockerfile.api               Multi-stage build for API + dashboard
│   └── docker-compose.yml           API + dashboard with health checks
├── docs/                            Assignment spec, dataset documentation
├── notebooks/                       Exploration & analysis notebooks
├── src/                             Application source code (see layout above)
├── tests/                           Unit, contract, and integration tests
├── .github/workflows/               CI/CD pipeline with metric gates
├── dvc.yaml                         DVC pipeline definition (7 stages)
├── Makefile                         Convenience targets
├── metrics.json                     Latest training metrics (DVC-tracked)
├── requirements.txt                 Python dependencies
└── pyproject.toml                   Ruff linter configuration
```

---

## Production Deployment Guide

### Use Case: Logistics Planning for Smelting Operations

The system predicts cumulative raw-material weights per material (`rm_id`) over time. The logistics team uses forecasts to:
- **Book transport** in advance (reduce spot-market costs)
- **Optimise safety stock** (avoid warehouse overflow)
- **Plan production shifts** based on expected material availability

### Recommended Workflow

1. **Daily predictions** — ERP system calls `/predict` each morning for active materials
2. **Weekly retrain** — CI/CD runs `make all` every Monday, deploys if metric gate passes
3. **Feedback loop** — Actual delivery weights are submitted via `/actuals` for monitoring
4. **Alerting** — Dashboard drift and performance sections trigger retrain recommendations

### Scaling

- For higher throughput, scale the API horizontally behind a load balancer (stateless design)
- For real-time streaming, the `/predict` endpoint supports sub-5 ms latency per request
- For batch scoring, use `/predict/batch` (up to 1,000 items per call)

---

## Configuration

All parameters in [`configs/params.yaml`](configs/params.yaml): model hyperparameters, feature definitions, data paths, MLflow settings, monitoring thresholds.

## Tech Stack

LightGBM · FastAPI · MLflow · DVC · Evidently · Streamlit · Docker · GitHub Actions
