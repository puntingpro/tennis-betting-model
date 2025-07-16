# src/scripts/dashboard/run_dashboard.py

import streamlit as st
import pandas as pd
import json
from pathlib import Path
from typing import Dict

from src.scripts.utils.config import load_config
from src.scripts.pipeline.simulate_bankroll_growth import (
    simulate_bankroll_growth,
    calculate_max_drawdown,
)

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    page_title="PuntingPro Dashboard",
    page_icon="🎾"
)

# --- Helper Functions ---
def get_tournament_category(tourney_name: str) -> str:
    """Categorizes a tournament name into a broader category for better analysis."""
    tourney_name = str(tourney_name).lower()
    category_map: Dict[str, str] = {
        'grand slam': 'Grand Slam', 'australian open': 'Grand Slam', 'roland garros': 'Grand Slam',
        'french open': 'Grand Slam', 'wimbledon': 'Grand Slam', 'us open': 'Grand Slam',
        'masters': 'Masters 1000', 'tour finals': 'Tour Finals', 'next gen finals': 'Tour Finals',
        'atp cup': 'Team Event', 'davis cup': 'Team Event', 'laver cup': 'Team Event',
        'olympics': 'Olympics', 'challenger': 'Challenger', 'itf': 'ITF / Futures'
    }
    for keyword, category in category_map.items():
        if keyword in tourney_name:
            return category
    return 'ATP / WTA Tour'

# --- Data Caching ---
@st.cache_data
def load_data(paths):
    """Loads all necessary data for the dashboard, caching the results."""
    model_meta_path = Path(paths['model']).with_suffix(".json")
    if not model_meta_path.exists():
        st.error(f"Model metadata not found at {model_meta_path}. Please run the 'model' command.")
        return None, None
    with open(model_meta_path, 'r') as f:
        model_meta = json.load(f)
    
    feature_importance_df = pd.DataFrame({
        'feature': model_meta.get('features', []),
        'importance': model_meta.get('feature_importances', [])
    }).sort_values(by='importance', ascending=False)
    
    backtest_path = Path(paths['backtest_results'])
    if not backtest_path.exists():
        st.error(f"Backtest results not found at {backtest_path}. Please run the 'backtest' command.")
        return None, None
        
    backtest_df = pd.read_csv(backtest_path, parse_dates=['tourney_date'])
    backtest_df['tourney_category'] = backtest_df['tourney_name'].apply(get_tournament_category)
    
    return feature_importance_df, backtest_df

# --- Main App ---
def main():
    st.title("🎾 PuntingPro: Strategy & Performance Dashboard")
    st.markdown("An interactive dashboard to analyze model performance and simulate betting strategies.")

    try:
        config = load_config("config.yaml") #
        paths = config['data_paths'] #
        
        feature_importance_df, backtest_df = load_data(paths)

        if feature_importance_df is None or backtest_df is None:
            st.warning("Required data files are missing. Please run the build, model, and backtest commands.")
            return

        # --- Sidebar ---
        st.sidebar.header("Simulation Controls")
        strategy = st.sidebar.selectbox("Select Staking Strategy", ["kelly", "flat", "percent"], index=0)
        initial_bankroll = st.sidebar.number_input("Initial Bankroll ($)", min_value=100.0, value=1000.0, step=100.0)

        if strategy == "kelly":
            stake_unit = 0.5
            kelly_fraction = st.sidebar.slider("Kelly Fraction", 0.1, 1.0, 0.5, 0.05)
        elif strategy == "flat":
            stake_unit = st.sidebar.number_input("Flat Stake Unit ($)", min_value=1.0, value=10.0, step=1.0)
            kelly_fraction = 0.0
        else: # percent
            stake_unit = st.sidebar.slider("Stake Percentage (%)", 0.5, 10.0, 1.0, 0.5)
            kelly_fraction = 0.0

        st.sidebar.header("Data Filters")
        
        # --- ADDED: Data Filtering Controls ---
        min_date = backtest_df['tourney_date'].min().date()
        max_date = backtest_df['tourney_date'].max().date()
        date_range = st.sidebar.date_input("Date Range", [min_date, max_date])
        
        all_categories = sorted(backtest_df['tourney_category'].unique())
        selected_categories = st.sidebar.multiselect("Tournament Categories", all_categories, default=all_categories)
        
        min_odds = float(backtest_df['odds'].min())
        max_odds = float(backtest_df['odds'].max())
        odds_range = st.sidebar.slider("Odds Range", min_value=min_odds, max_value=max_odds, value=(min_odds, max_odds))

        # --- Apply Filters ---
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        filtered_df = backtest_df[
            (backtest_df['tourney_date'] >= start_date) &
            (backtest_df['tourney_date'] <= end_date) &
            (backtest_df['tourney_category'].isin(selected_categories)) &
            (backtest_df['odds'] >= odds_range[0]) &
            (backtest_df['odds'] <= odds_range[1])
        ]
        
        st.sidebar.info(f"Displaying {len(filtered_df)} of {len(backtest_df)} total backtested bets.")

        # --- Run Simulation on Filtered Data ---
        simulation_df = simulate_bankroll_growth(
            filtered_df.copy(),
            initial_bankroll=initial_bankroll,
            strategy=strategy,
            stake_unit=stake_unit,
            kelly_fraction=kelly_fraction
        ).dropna(subset=['bankroll'])

        # --- Main Content Area ---
        st.header("Bankroll Growth Simulation")

        if not simulation_df.empty:
            final_bankroll = simulation_df['bankroll'].iloc[-1]
            total_profit = final_bankroll - initial_bankroll
            total_wagered = simulation_df['stake'].sum()
            roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
            win_rate = (simulation_df['profit'] > 0).mean() * 100
            peak_bankroll, max_drawdown = calculate_max_drawdown(simulation_df['bankroll'])

            kpi_cols = st.columns(4)
            kpi_cols[0].metric("Final Bankroll", f"${final_bankroll:,.2f}", f"Profit: ${total_profit:,.2f}")
            kpi_cols[1].metric("Return on Investment", f"{roi:.2f}%")
            kpi_cols[2].metric("Max Drawdown", f"{abs(max_drawdown):.2%}")
            kpi_cols[3].metric("Win Rate", f"{win_rate:.2f}%")
            
            # --- MODIFIED: Added Cumulative Profit chart ---
            simulation_df['cumulative_profit'] = simulation_df['profit'].cumsum()
            
            chart_cols = st.columns(2)
            chart_cols[0].subheader("Bankroll Growth")
            chart_cols[0].line_chart(simulation_df.set_index('tourney_date')['bankroll'])
            chart_cols[1].subheader("Cumulative Profit")
            chart_cols[1].line_chart(simulation_df.set_index('tourney_date')['cumulative_profit'])
            
            st.markdown("---")
            st.header("Simulation Data")
            # --- ADDED: Detailed data table for analysis ---
            if st.checkbox("Show Detailed Bet-by-Bet Simulation Data"):
                st.dataframe(simulation_df)

        else:
            st.warning("No data matches the selected filters. Simulation could not be run.")

        st.markdown("---")
        
        st.header("Model Feature Importances")
        st.bar_chart(feature_importance_df.head(25).set_index('feature'))

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.exception(e)

if __name__ == "__main__":
    main()