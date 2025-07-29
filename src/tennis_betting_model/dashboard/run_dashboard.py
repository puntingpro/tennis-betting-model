import pandas as pd
import streamlit as st
from pathlib import Path

from tennis_betting_model.utils.logger import setup_logging, log_error
from tennis_betting_model.utils.config import load_config


def load_backtest_data(paths: dict) -> pd.DataFrame:
    """Loads and prepares the backtest results data."""
    try:
        results_path = Path(paths["backtest_results"])
        df = pd.read_csv(results_path)
        df["tourney_date"] = pd.to_datetime(df["tourney_date"])
        # Ensure 'pnl' column exists from the backtest script
        if "pnl" not in df.columns:
            df["pnl"] = df.apply(
                lambda row: (row["odds"] - 1) * 0.95 if row["winner"] == 1 else -1,
                axis=1,
            )
        return df.sort_values("tourney_date")
    except FileNotFoundError:
        log_error(
            f"Backtest results not found at {paths['backtest_results']}. Please run a backtest first."
        )
        return pd.DataFrame()


def run() -> None:
    """Main function to run the enhanced Streamlit dashboard."""
    st.set_page_config(layout="wide", page_title="PuntingPro Performance Dashboard")
    setup_logging()

    st.title("🎾 PuntingPro Performance Dashboard")
    st.markdown(
        "An interactive dashboard to analyze backtesting results and betting strategies."
    )

    try:
        config = load_config("config.yaml")
        paths = config["data_paths"]
        df_full = load_backtest_data(paths)
    except Exception as e:
        st.error(f"Failed to load configuration or data. Error: {e}")
        return

    if df_full.empty:
        st.warning("No backtest data available to display.")
        return

    # --- Sidebar Filters ---
    st.sidebar.header("Strategy Filters")

    min_date = df_full["tourney_date"].min().date()
    max_date = df_full["tourney_date"].max().date()

    date_range = st.sidebar.date_input(
        "Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
    )

    odds_range = st.sidebar.slider(
        "Odds Range",
        min_value=1.0,
        max_value=float(df_full["odds"].max()),
        value=(1.0, 10.0),
        step=0.1,
    )

    ev_range = st.sidebar.slider(
        "Expected Value (EV) Range",
        min_value=float(df_full["expected_value"].min()),
        max_value=float(df_full["expected_value"].max()),
        value=(0.1, float(df_full["expected_value"].max())),
        step=0.01,
    )

    # Apply filters
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df = df_full[
        (df_full["tourney_date"] >= start_date)
        & (df_full["tourney_date"] <= end_date)
        & (df_full["odds"] >= odds_range[0])
        & (df_full["odds"] <= odds_range[1])
        & (df_full["expected_value"] >= ev_range[0])
        & (df_full["expected_value"] <= ev_range[1])
    ].copy()

    # --- Main Content ---
    if df.empty:
        st.info("No bets match the current filter criteria.")
        return

    # --- Key Performance Indicators (KPIs) ---
    st.header("Performance Overview")
    total_bets = len(df)
    total_pnl = df["pnl"].sum()
    roi = (total_pnl / total_bets) * 100 if total_bets > 0 else 0
    win_rate = (df["winner"].sum() / total_bets) * 100 if total_bets > 0 else 0
    avg_odds = df["odds"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Bets", f"{total_bets:,}")
    col2.metric("Total P/L (Units)", f"{total_pnl:,.2f}")
    col3.metric("ROI", f"{roi:.2f}%")
    col4.metric("Win Rate", f"{win_rate:.2f}%")
    col5.metric("Avg. Odds", f"{avg_odds:.2f}")

    st.divider()

    # --- Cumulative Profit Chart ---
    st.header("Cumulative Profit/Loss Over Time")
    df["cumulative_pnl"] = df["pnl"].cumsum()
    st.line_chart(df.set_index("tourney_date")["cumulative_pnl"])

    st.divider()

    # --- Data View ---
    st.header("Filtered Bet History")
    st.dataframe(
        df[
            [
                "tourney_date",
                "match_id",
                "odds",
                "predicted_prob",
                "expected_value",
                "kelly_fraction",
                "winner",
                "pnl",
            ]
        ].tail(100)
    )


if __name__ == "__main__":
    run()
