import pandas as pd
import streamlit as st
from pathlib import Path
from typing import cast, Tuple, List, Any
import datetime

from tennis_betting_model.utils.logger import setup_logging, log_error
from tennis_betting_model.utils.config import load_config
from tennis_betting_model.pipeline.simulate_bankroll_growth import (
    simulate_bankroll_growth,
    calculate_max_drawdown,
)


@st.cache_data
def load_backtest_data(paths: dict) -> pd.DataFrame:
    """Loads and prepares the backtest results data."""
    try:
        results_path = Path(paths["backtest_results"])
        df = pd.read_csv(results_path)
        df["tourney_date"] = pd.to_datetime(df["tourney_date"])

        if "pnl" not in df.columns:
            df["pnl"] = df.apply(
                lambda row: (row["odds"] - 1) * 0.95 if row["winner"] == 1 else -1,
                axis=1,
            )

        features_path = Path(paths["consolidated_features"])
        if features_path.exists():
            df_features = pd.read_csv(features_path, usecols=["market_id", "rank_diff"])
            df["market_id"] = df["market_id"].astype(str)
            df_features["market_id"] = df_features["market_id"].astype(str)
            df = pd.merge(df, df_features, on="market_id", how="left")
        else:
            df["rank_diff"] = 0

        return df.sort_values("tourney_date")
    except FileNotFoundError:
        log_error(
            f"Backtest results not found at {paths['backtest_results']}. Please run a backtest first."
        )
        return cast(pd.DataFrame, pd.DataFrame())
    except Exception as e:
        st.error(f"An error occurred loading backtest data: {e}")
        return cast(pd.DataFrame, pd.DataFrame())


def create_summary_table(
    df: pd.DataFrame, column: str, bins: List[Any], title: str
) -> None:
    """Creates and displays a summary table for a given column, bucketing the data."""
    st.subheader(title)
    if column not in df.columns or df[column].isnull().all():
        st.warning(f"Data for '{column}' is not available to generate summary.")
        return

    # For EV, we need to dynamically add the max value to the bins
    if column == "expected_value" and not df.empty:
        bins.append(df["expected_value"].max())

    df[f"{column}_bucket"] = pd.cut(df[column], bins=bins, right=False)
    summary = (
        df.groupby(f"{column}_bucket", observed=True)
        .agg(bets=("market_id", "count"), pnl=("pnl", "sum"))
        .reset_index()
    )
    summary["roi"] = (summary["pnl"] / summary["bets"]) * 100
    st.dataframe(
        summary.style.format({"pnl": "{:.2f}", "roi": "{:.2f}%"}),
        use_container_width=True,
    )


def run() -> None:
    """Main function to run the enhanced Streamlit dashboard."""
    st.set_page_config(layout="wide", page_title="Performance Dashboard")
    setup_logging()

    st.title("疾 PuntingPro Performance Dashboard")
    st.markdown(
        "An interactive dashboard to analyze backtesting results and betting strategies."
    )

    try:
        config = load_config("config.yaml")
        paths = config["data_paths"]
        analysis_params = config.get("analysis_params", {})
        df_full = load_backtest_data(paths)
    except Exception as e:
        st.error(f"Failed to load configuration or data. Error: {e}")
        return

    if df_full.empty:
        st.warning("No backtest data available to display.")
        return

    # --- Sidebar Filters ---
    st.sidebar.header("Master Strategy Filters")

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
        value=(0.0, float(df_full["expected_value"].max())),
        step=0.01,
    )

    # --- Filter Data based on sidebar selections ---
    date_range_tuple = cast(Tuple[datetime.date, datetime.date], date_range)
    odds_range_tuple = cast(Tuple[float, float], odds_range)
    ev_range_tuple = cast(Tuple[float, float], ev_range)

    start_date, end_date = pd.to_datetime(date_range_tuple[0]), pd.to_datetime(
        date_range_tuple[1]
    )
    df = df_full[
        (df_full["tourney_date"] >= start_date)
        & (df_full["tourney_date"] <= end_date)
        & (df_full["odds"] >= odds_range_tuple[0])
        & (df_full["odds"] <= odds_range_tuple[1])
        & (df_full["expected_value"] >= ev_range_tuple[0])
        & (df_full["expected_value"] <= ev_range_tuple[1])
    ].copy()

    if df.empty:
        st.info("No bets match the current filter criteria.")
        return

    # --- Main Page Layout ---
    st.header("嶋 Performance Overview")
    total_bets = len(df)
    total_pnl = df["pnl"].sum()
    roi = (total_pnl / total_bets) * 100 if total_bets > 0 else 0
    avg_odds = df["odds"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bets", f"{total_bets:,}")
    col2.metric("Total P/L (Units)", f"{total_pnl:,.2f}")
    col3.metric("ROI", f"{roi:.2f}%")
    col4.metric("Avg. Odds", f"{avg_odds:.2f}")

    st.divider()

    # --- Bankroll Simulation Section ---
    st.header("腸 Bankroll Growth Simulation")
    sim_col1, sim_col2 = st.columns([1, 3])

    with sim_col1:
        st.subheader("Simulation Settings")
        initial_bankroll = st.number_input("Initial Bankroll", value=1000.0, step=100.0)
        staking_strategy = st.selectbox(
            "Staking Strategy", ["kelly", "flat", "percent"]
        )
        kelly_fraction = (
            st.slider("Kelly Fraction", 0.01, 1.0, 0.1, 0.01)
            if staking_strategy == "kelly"
            else 0.5
        )
        stake_unit = (
            st.number_input("Stake Unit / Percent", value=10.0, step=1.0)
            if staking_strategy != "kelly"
            else 10.0
        )

    simulated_df = simulate_bankroll_growth(
        df.copy(),
        initial_bankroll=initial_bankroll,
        strategy=staking_strategy,
        stake_unit=stake_unit,
        kelly_fraction=kelly_fraction,
    )

    with sim_col2:
        if not simulated_df.empty:
            peak_bankroll, max_dd = calculate_max_drawdown(simulated_df["bankroll"])
            final_bankroll = simulated_df["bankroll"].iloc[-1]

            st.subheader("Simulation Results")
            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
            kpi_col1.metric("Final Bankroll", f"${final_bankroll:,.2f}")
            kpi_col2.metric("Peak Bankroll", f"${peak_bankroll:,.2f}")
            kpi_col3.metric("Max Drawdown", f"{max_dd:.2%}")

            st.line_chart(simulated_df.set_index("tourney_date")["bankroll"])

    st.divider()

    # --- Performance Breakdown Section ---
    st.header("投 Performance Breakdown")
    breakdown_col1, breakdown_col2 = st.columns(2)

    with breakdown_col1:
        create_summary_table(
            df.copy(),
            "odds",
            analysis_params.get("odds_bins", []),
            "By Odds Bucket",
        )

    with breakdown_col2:
        create_summary_table(
            df.copy(),
            "expected_value",
            analysis_params.get("ev_bins", []),
            "By Expected Value (EV)",
        )

    create_summary_table(
        df.copy(),
        "rank_diff",
        analysis_params.get("rank_bins", []),
        "By Player Rank Difference",
    )

    st.divider()

    # --- Filtered Bet History ---
    st.header("搭 Filtered Bet History")
    st.dataframe(
        df[
            [
                "tourney_date",
                "tourney_name",
                "odds",
                "predicted_prob",
                "expected_value",
                "kelly_fraction",
                "winner",
                "pnl",
            ]
        ].tail(200)
    )


if __name__ == "__main__":
    run()
