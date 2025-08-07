import pandas as pd
import streamlit as st
import numpy as np
from typing import cast, Tuple, List, Any
import datetime

from tennis_betting_model.utils.logger import setup_logging
from tennis_betting_model.utils.config import load_config, Config
from tennis_betting_model.utils.data_loader import DataLoader
from tennis_betting_model.utils.config_schema import DataPaths
from tennis_betting_model.pipeline.simulate_bankroll_growth import (
    simulate_bankroll_growth,
    calculate_max_drawdown,
)


@st.cache_data
def load_data(_paths: DataPaths) -> pd.DataFrame:
    """Wrapper function to cache the data loading."""
    data_loader = DataLoader(_paths)
    return cast(pd.DataFrame, data_loader.load_backtest_data_for_dashboard())


def create_summary_table(
    df: pd.DataFrame, column: str, bins: List[Any], title: str
) -> None:
    """Creates and displays a summary table for a given column, bucketing the data."""
    st.subheader(title)
    if column not in df.columns or df[column].isnull().all() or df.empty:
        st.warning(f"Data for '{column}' is not available to generate summary.")
        return

    # Use a robust binning method with infinite boundaries to capture all data
    bin_edges = sorted(list(set([-np.inf] + bins + [np.inf])))

    df[f"{column}_bucket"] = pd.cut(df[column], bins=bin_edges)
    summary = (
        df.groupby(f"{column}_bucket", observed=False)
        .agg(bets=("market_id", "count"), pnl=("pnl", "sum"))
        .reset_index()
    )
    # Remove empty bins before calculating ROI
    summary = summary[summary["bets"] > 0].copy()
    if summary.empty:
        st.warning(f"No data falls into the defined bins for '{column}'.")
        return

    summary["roi"] = (summary["pnl"] / summary["bets"]) * 100
    st.dataframe(
        summary.style.format({"pnl": "{:.2f}", "roi": "{:.2f}%"}),
        use_container_width=True,
    )


def run() -> None:
    """Main function to run the enhanced Streamlit dashboard."""
    st.set_page_config(layout="wide", page_title="Performance Dashboard")
    setup_logging()

    st.title("Performance Dashboard")
    st.markdown(
        "An interactive dashboard to analyze backtesting results and betting strategies."
    )

    try:
        config_dict = load_config("config.yaml")
        config = Config(**config_dict)
        analysis_params = config.analysis_params.dict()
        df_full = load_data(config.data_paths)
    except Exception as e:
        st.error(f"Failed to load configuration or data. Error: {e}")
        return

    if df_full.empty:
        st.warning("No backtest data available to display.")
        return

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

    st.header("Performance Overview")
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

    st.header("Bankroll Growth Simulation")
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
        max_stake_cap = st.slider("Max Stake Cap (% of Bankroll)", 1, 100, 5, 1)

    simulated_df = simulate_bankroll_growth(
        df.copy(),
        config.simulation_params.dict(),
        initial_bankroll=initial_bankroll,
        strategy=staking_strategy,
        stake_unit=stake_unit,
        kelly_fraction=kelly_fraction,
        max_stake_cap=(max_stake_cap / 100.0),
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

    st.header("Performance Breakdown")
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

    st.header("Filtered Bet History")
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
