"""Streamlit dashboard for monitoring model performance and data drift.

Run with: streamlit run src/monitoring/dashboard.py
"""

import json

import pandas as pd
import streamlit as st
import yaml


def load_config(path: str = "configs/params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    st.set_page_config(page_title="ML Monitor", layout="wide")
    st.title("Cumulative Weight Forecast — Monitoring Dashboard")

    cfg = load_config()

    # ── Sidebar ─────────────────────────────────────────────
    st.sidebar.header("Settings")
    st.sidebar.number_input("Quantile τ", value=cfg["train"]["quantile_tau"], step=0.05)
    perf_threshold = cfg["monitoring"]["performance_threshold"]

    # ── Performance Report ──────────────────────────────────
    st.header("1. Model Performance")
    try:
        with open("data/processed/performance_report.json") as f:
            perf = json.load(f)
        metrics = perf.get("metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Quantile Loss", f"{metrics.get('quantile_loss', 0):.4f}")
        col2.metric("MAE", f"{metrics.get('mae', 0):.2f}")
        col3.metric("RMSE", f"{metrics.get('rmse', 0):.2f}")
        col4.metric(
            "Overestimation Rate",
            f"{metrics.get('overestimation_rate', 0):.1%}",
        )
        if perf.get("retrain_recommended"):
            st.error(
                f"Retrain recommended: quantile loss ({metrics.get('quantile_loss', 0):.4f}) "
                f"exceeds threshold ({perf_threshold})"
            )
        else:
            st.success("Model performance is within acceptable range.")
    except FileNotFoundError:
        st.warning("No performance report found. Run `python -m src.monitoring.performance` first.")

    # ── Drift Report ────────────────────────────────────────
    st.header("2. Data Drift")
    try:
        with open("data/processed/drift_report.json") as f:
            drift = json.load(f)
        col1, col2 = st.columns(2)
        col1.metric("Dataset Drift", "Yes" if drift.get("dataset_drift") else "No")
        col2.metric("Drifted Columns", drift.get("number_of_drifted_columns", 0))
        if drift.get("dataset_drift"):
            st.warning("Data drift detected! Consider retraining the model.")
        else:
            st.success("No significant data drift detected.")
    except FileNotFoundError:
        st.warning("No drift report found. Run `python -m src.monitoring.drift` first.")

    # ── Predictions Overview ────────────────────────────────
    st.header("3. Predictions Overview")
    try:
        pred_path = cfg["data"]["processed"]["predictions"]
        preds = pd.read_csv(pred_path, parse_dates=["date"])
        if "predicted_cumulative_weight" in preds.columns:
            value_col = "predicted_cumulative_weight"
        else:
            value_col = preds.columns[-1]

        # Aggregate by date
        daily_agg = preds.groupby("date")[value_col].agg(["mean", "sum", "count"]).reset_index()
        st.line_chart(daily_agg.set_index("date")[["mean"]], use_container_width=True)

        # Per rm_id selector
        rm_ids = sorted(preds["rm_id"].unique())
        selected_rm = st.selectbox("Select rm_id", rm_ids)
        rm_preds = preds[preds["rm_id"] == selected_rm].sort_values("date")
        st.line_chart(rm_preds.set_index("date")[[value_col]], use_container_width=True)
    except FileNotFoundError:
        st.warning("No predictions found. Run `python -m src.models.predict` first.")

    # ── Training Metrics (from MLflow or metrics.json) ──────
    st.header("4. Training Metrics")
    try:
        with open("metrics.json") as f:
            train_metrics = json.load(f)
        col1, col2, col3 = st.columns(3)
        col1.metric("Train QL", f"{train_metrics.get('quantile_loss', 0):.4f}")
        col2.metric("Train MAE", f"{train_metrics.get('mae', 0):.2f}")
        col3.metric("Train RMSE", f"{train_metrics.get('rmse', 0):.2f}")
    except FileNotFoundError:
        st.info("No training metrics found. Run `python -m src.models.train` first.")


if __name__ == "__main__":
    main()
