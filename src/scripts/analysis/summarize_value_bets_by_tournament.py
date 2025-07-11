# src/scripts/analysis/summarize_value_bets_by_tournament.py

from pathlib import Path
import pandas as pd
import argparse

from src.scripts.utils.file_utils import load_dataframes
from src.scripts.utils.logger import log_error, log_success, setup_logging
from src.scripts.utils.config import load_config

def run_summarize_by_tournament(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes backtest results by tournament with a corrected
    profit and ROI calculation.
    """
    if df.empty:
        return pd.DataFrame()

    # --- FIX: Correctly determine if the bet was a winner ---
    # A bet on P1 wins if predicted_prob > 0.5 and winner == 1.
    # A bet on P2 wins if predicted_prob < 0.5 and winner == 0.
    # This is equivalent to checking if the rounded prediction matches the winner.
    df['is_correct'] = (df['predicted_prob'].round() == df['winner'])

    # Calculate profit based on whether the bet was correct
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
    tournament_summary["roi"] = (
        (tournament_summary["total_profit"] / tournament_summary["total_bets"]) * 100
    )
    return tournament_summary.sort_values(by="roi", ascending=False)

def main_cli(args):
    """
    Main function for summarizing by tournament, driven by the config file.
    """
    setup_logging()
    config = load_config(args.config)
    paths = config['data_paths']

    df = load_dataframes(paths['backtest_results'])
    summary_df = run_summarize_by_tournament(df)

    if not summary_df.empty:
        log_success("Successfully summarized tournaments.")
        print(summary_df.to_string())

        output_path_obj = Path(paths['tournament_summary'])
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path_obj, index=False)
        log_success(f"Saved tournament summary to {output_path_obj}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to the config file.")
    args = parser.parse_args()
    main_cli(args)