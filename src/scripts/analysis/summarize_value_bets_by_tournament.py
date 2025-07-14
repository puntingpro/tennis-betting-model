# src/scripts/analysis/summarize_value_bets_by_tournament.py

from pathlib import Path
import pandas as pd
import argparse

from src.scripts.utils.file_utils import load_dataframes
from src.scripts.utils.logger import log_error, log_success, setup_logging
from src.scripts.utils.config import load_config

def run_summarize_by_tournament(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes backtest results by tournament, calculating total bets, profit, and ROI.

    Args:
        df (pd.DataFrame): The DataFrame containing detailed backtest results.

    Returns:
        pd.DataFrame: A DataFrame summarizing performance for each tournament,
                      sorted by Return on Investment (ROI).
    """
    if df.empty:
        return pd.DataFrame()

    # A bet is considered 'correct' if the player with the higher predicted
    # probability was the actual winner of the match.
    # Note: 'winner' column is 1 if player 1 wins, 0 if player 2 wins.
    # The 'predicted_prob' in the value bets file is always for the player bet on.
    df['is_correct'] = (df['predicted_prob'].round() == df['winner'])

    # Profit is calculated as (odds - 1) for a winning bet, and -1 for a losing bet (assuming a 1-unit stake).
    df['profit'] = df.apply(
        lambda row: (row['odds'] - 1) if row['is_correct'] else -1,
        axis=1
    )
    df['stake'] = 1 # Assume flat unit stake for ROI calculation

    tournament_summary = (
        df.groupby("tourney_name")
        .agg(
            total_bets=("stake", "sum"),
            total_profit=("profit", "sum"),
            num_matches=("match_id", "nunique"),
        )
        .reset_index()
    )
    
    # Calculate Return on Investment (ROI)
    tournament_summary["roi"] = (
        (tournament_summary["total_profit"] / tournament_summary["total_bets"]) * 100
    )
    return tournament_summary.sort_values(by="roi", ascending=False)

def main_cli(args: argparse.Namespace) -> None:
    """
    Main CLI entrypoint for summarizing backtest results by tournament.

    Args:
        args (argparse.Namespace): Arguments parsed from the command line.
    """
    setup_logging()
    config = load_config(args.config)
    paths = config['data_paths']

    df = load_dataframes(paths['backtest_results'])
    summary_df = run_summarize_by_tournament(df)

    if not summary_df.empty:
        log_success("Successfully summarized tournaments.")
        print(summary_df.to_string())

        output_path = Path(paths['tournament_summary'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        log_success(f"Saved tournament summary to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize backtest results by tournament to calculate profit and ROI.")
    parser.add_argument("--config", default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()
    main_cli(args)