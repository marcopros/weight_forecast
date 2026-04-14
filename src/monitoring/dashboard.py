"""Streamlit dashboard for monitoring model performance and data drift.

Includes: performance metrics, data drift, predictions overview,
feature importance, per-rm_id error analysis, baseline comparison,
API latency monitoring, and model versioning info.
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

        # Show prediction log analysis if available
        log_analysis = perf.get("prediction_log_analysis", {})
        if log_analysis.get("status") == "ok":
            st.subheader("API Latency & Usage")
            lcol1, lcol2, lcol3 = st.columns(3)
            lcol1.metric("Total Predictions Served", log_analysis.get("total_predictions", 0))
            lcol2.metric("p95 Latency", f"{log_analysis.get('p95_latency_ms', 0):.1f} ms")
            lcol3.metric("Unique rm_ids Served", log_analysis.get("unique_rm_ids_served", 0))

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

    # ── Training Metrics + Model Info ───────────────────────
    st.header("4. Training Metrics & Model Info")
    try:
        with open("metrics.json") as f:
            train_metrics = json.load(f)

        col1, col2, col3 = st.columns(3)
        col1.metric("Quantile Loss", f"{train_metrics.get('quantile_loss', 0):.4f}")
        col2.metric("MAE", f"{train_metrics.get('mae', 0):.2f}")
        col3.metric("RMSE", f"{train_metrics.get('rmse', 0):.2f}")

        # Model version info
        mcol1, mcol2, mcol3 = st.columns(3)
        mcol1.metric("Model Hash", train_metrics.get("model_hash", "N/A"))
        mcol2.metric("Best Iteration", train_metrics.get("best_iteration", "N/A"))
        mcol3.metric("Train Duration", f"{train_metrics.get('train_duration_seconds', 0):.1f}s")

        # ── Baseline Comparison ─────────────────────────────
        baselines = train_metrics.get("baselines", {})
        if baselines:
            st.subheader("Baseline Comparison")
            baseline_data = []
            for name, bm in baselines.items():
                baseline_data.append({
                    "Model": name,
                    "Quantile Loss": bm.get("quantile_loss", 0),
                    "MAE": bm.get("mae", 0),
                })
            baseline_data.append({
                "Model": "LightGBM (ours)",
                "Quantile Loss": train_metrics.get("quantile_loss", 0),
                "MAE": train_metrics.get("mae", 0),
            })
            baseline_df = pd.DataFrame(baseline_data).sort_values("Quantile Loss")
            st.dataframe(baseline_df, use_container_width=True, hide_index=True)

            # Bar chart comparison
            st.bar_chart(baseline_df.set_index("Model")["Quantile Loss"])

        # ── Feature Importance ──────────────────────────────
        feat_imp = train_metrics.get("feature_importance", {})
        if feat_imp:
            st.subheader("Feature Importance")
            imp_df = pd.DataFrame(
                [{"Feature": k, "Importance": v} for k, v in feat_imp.items()]
            ).sort_values("Importance", ascending=True)
            st.bar_chart(imp_df.set_index("Feature"))

    except FileNotFoundError:
        st.info("No training metrics found. Run `python -m src.models.train` first.")

    # ── Error Analysis ──────────────────────────────────────
    st.header("5. Per-rm_id Error Analysis")
    try:
        with open("data/processed/error_analysis.json") as f:
            error_data = json.load(f)

        ecol1, ecol2 = st.columns(2)
        ecol1.metric("Total rm_ids Evaluated", error_data.get("total_rm_ids", 0))
        ecol2.metric(
            "Overestimating / Underestimating",
            f"{error_data.get('overestimating_rm_ids', 0)}"
            f" / {error_data.get('underestimating_rm_ids', 0)}"
        )

        st.subheader("Worst 10 rm_ids (by Quantile Loss)")
        worst = error_data.get("top_10_worst", [])
        if worst:
            worst_df = pd.DataFrame(worst)
            st.dataframe(worst_df, use_container_width=True, hide_index=True)

        st.subheader("Best 10 rm_ids (by Quantile Loss)")
        best = error_data.get("top_10_best", [])
        if best:
            best_df = pd.DataFrame(best)
            st.dataframe(best_df, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.info("No error analysis found. Run `python -m src.models.train` first.")


if __name__ == "__main__":
    main()
