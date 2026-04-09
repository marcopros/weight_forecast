# Cumulative Weight Forecast

Forecasting cumulative raw material delivery weights for Hydro ASA using LightGBM with an asymmetric quantile loss (τ=0.2) that penalizes overestimation 4× more than underestimation.

## Architecture

```
src/
├── data/         clean.py · features.py       Data pipeline & feature engineering
├── models/       train.py · predict.py        Training (MLflow) & inference
├── api/          main.py                      FastAPI serving endpoint
└── monitoring/   drift.py · performance.py    Evidently drift + perf tracking
                  dashboard.py                 Streamlit monitoring UI
```

## Quick Start

```bash
pip install -r requirements.txt

# Run full pipeline
make all          # clean → features → train → predict

# Or step by step
python -m src.data.clean
python -m src.data.features
python -m src.models.train
python -m src.models.predict
```

## Serving

```bash
make serve        # → http://localhost:8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"rm_id": 365, "forecast_end_date": "2025-03-15"}'
# → {"rm_id": 365, "forecast_end_date": "2025-03-15", "cumulative_weight": 56431.42}
```

Interactive API docs at `/docs`. Batch predictions at `POST /predict/batch`.

## Pipeline (DVC)

```bash
dvc repro         # Runs only stages with changed inputs
```

Stages: `clean → features → train → predict → monitor_drift → monitor_performance`

## Monitoring

```bash
make monitor      # Data drift + model performance checks
make dashboard    # Streamlit UI on :8501
```

- **Drift detection**: Evidently AI compares feature distributions (train vs production)
- **Performance tracking**: Quantile loss vs configurable threshold → retrain alerts

## Docker

```bash
docker build -f docker/Dockerfile.api -t forecast-api .
docker run -p 8000:8000 forecast-api

# Or with monitoring dashboard
cd docker && docker-compose up
```

## Testing

```bash
make test         # 16 tests covering data pipeline, loss function, and API
```

## Project Structure

```
configs/params.yaml          Hyperparameters, paths, thresholds
data/kernel/                 Raw data (receivals, purchase_orders)
data/extended/               Supplementary data (materials, transportation)
data/processed/              Pipeline outputs (model, predictions)
notebooks/                   Original exploration notebooks
docs/                        Assignment spec, dataset documentation
.github/workflows/           CI/CD: lint → test → train → docker push
```

## Model

| | |
|---|---|
| **Algorithm** | LightGBM Regressor |
| **Target** | Cumulative weight from Jan 1 per (rm_id, year) |
| **Features** | Temporal (7) · Lag (3) · Rolling (4) · Material stats (3) |
| **Loss** | Quantile τ=0.2 (conservative: penalizes overestimation) |
| **Post-processing** | Clip negatives · Monotonic constraint (cummax per rm_id) |

## Configuration

All parameters in [`configs/params.yaml`](configs/params.yaml): model hyperparameters, feature definitions, data paths, MLflow settings, monitoring thresholds.

## Tech Stack

LightGBM · FastAPI · MLflow · DVC · Evidently · Streamlit · Docker · GitHub Actions
