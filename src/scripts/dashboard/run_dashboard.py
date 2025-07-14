# src/scripts/dashboard/run_dashboard.py

import streamlit as st
import pandas as pd
import json
from pathlib import Path

from src.scripts.utils.config import load_config

st.set_page_config(layout="wide", page_title="PuntingPro Dashboard")


# --- Load Data ---
@st.cache_data
def load_data(paths):
    model_meta_path = Path(paths["model"]).with_suffix(".json")
    with open(model_meta_path, "r") as f:
        model_meta = json.load(f)

    # --- MODIFIED: Ensure feature importances are loaded correctly ---
    feature_importance_df = pd.DataFrame(
        {
            "feature": model_meta.get("features", []),
            "importance": model_meta.get("feature_importances", []),
        }
    ).sort_values(by="importance", ascending=False)

    backtest_summary = pd.read_csv(paths["tournament_summary"])
    return feature_importance_df, backtest_summary, model_meta


# --- Main App ---
def main():
    st.title("🎾 PuntingPro: Tennis Value Betting Dashboard")

    try:
        config = load_config("config.yaml")
        paths = config["data_paths"]

        feature_importance_df, backtest_summary, model_meta = load_data(paths)

        # --- Key Metrics ---
        st.header("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Type", model_meta.get("model_type", "N/A"))
        col2.metric("Cross-Validated AUC", f"{model_meta.get('cross_val_auc', 0):.4f}")
        col3.metric("Number of Features", len(model_meta.get("features", [])))

        st.markdown("---")

        # --- Visualizations ---
        col1, col2 = st.columns(2)

        with col1:
            st.header("Top 25 Feature Importances")
            if not feature_importance_df.empty:
                chart_data = feature_importance_df.head(25).set_index("feature")
                st.bar_chart(chart_data)
            else:
                st.warning("Feature importance data is not available.")

        with col2:
            st.header("Backtest ROI by Tournament")
            st.dataframe(backtest_summary.sort_values(by="roi", ascending=False))

    except FileNotFoundError as e:
        st.error(
            f"Error loading data: {e}. Please ensure the `consolidate`, `build`, and `model` commands have been run successfully."
        )
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
