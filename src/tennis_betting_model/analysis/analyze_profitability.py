# src/tennis_betting_model/analysis/analyze_profitability.py

import pandas as pd
from pathlib import Path
from tennis_betting_model.utils.config_schema import Config
from tennis_betting_model.utils.logger import setup_logging, log_info, log_error
from tennis_betting_model.utils.betting_math import calculate_pnl


def print_report(df: pd.DataFrame, title: str):
    """Helper function to print a standardized performance report."""
    if df.empty:
        log_info(f"\n{title}\n" + "-" * 50 + "\nNo bets in this category.\n" + "-" * 50)
        return

    df_report = calculate_pnl(df.copy())

    total_bets = len(df_report)
    total_pnl = df_report["pnl"].sum()

    roi = (total_pnl / total_bets) * 100 if total_bets > 0 else 0
    win_rate = (df_report["winner"].sum() / total_bets) * 100 if total_bets > 0 else 0
    avg_odds = df_report["odds"].mean()

    print(f"\n{title}")
    print("-" * 50)
    print(f"{'Total Bets Placed:':<25} {total_bets}")
    print(f"{'Win Rate:':<25} {win_rate:.2f}%")
    print(f"{'Average Odds:':<25} {avg_odds:.2f}")
    print(f"{'Total Profit/Loss:':<25} {total_pnl:.2f} units")
    print(f"{'Return on Investment (ROI):':<25} {roi:.2f}%")
    print("-" * 50)


def main_cli(config: Config):
    """Main CLI entrypoint for analyzing backtest profitability from config."""
    setup_logging()
    paths = config.data_paths
    strategies = config.analysis_strategies

    log_info("--- Running Definitive Strategy Analysis Report ---")

    try:
        backtest_results_path = Path(paths.backtest_results)
        log_info(f"Loading backtest data from {backtest_results_path}...")
        df = pd.read_csv(backtest_results_path)
    except FileNotFoundError:
        log_error(f"Error: The file {backtest_results_path} was not found.")
        log_error("Please run the 'backtest' command first to generate this file.")
        return

    print_report(df, "Overall Performance (All Value Bets)")

    if not strategies:
        log_error("No strategies found in config under 'analysis_strategies'.")
        return

    # CHANGED: Iterate over the dictionary's values
    for strategy_model in strategies.values():
        strategy = strategy_model.dict()
        name = strategy.get("name", "Unnamed Strategy")
        min_odds = strategy.get("min_odds", 0.0)
        max_odds = strategy.get("max_odds", 1000.0)
        min_ev = strategy.get("min_ev", 0.0)

        strategy_df = df[
            (df["odds"] >= min_odds)
            & (df["odds"] <= max_odds)
            & (df["expected_value"] >= min_ev)
        ]
        print_report(strategy_df, f"Strategy: {name}")
