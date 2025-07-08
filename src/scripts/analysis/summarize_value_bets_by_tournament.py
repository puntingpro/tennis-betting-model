# src/scripts/analysis/summarize_value_bets_by_tournament.py
import sys
from pathlib import Path
import pandas as pd
import argparse

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.scripts.utils.file_utils import load_dataframes
from src.scripts.utils.logger import log_error, log_success, setup_logging

def run_summarize_by_tournament(df: pd.DataFrame) -> pd.DataFrame:
    tournament_summary = (
        df.groupby("tourney_name")
        .agg(
            total_bets=("total_bets", "sum"),
            total_profit=("total_profit", "sum"),
            num_matches=("match_id", "nunique"),
        )
        .reset_index()
    )
    tournament_summary["roi"] = (
        (tournament_summary["total_profit"] / tournament_summary["total_bets"]) * 100
    )
    return tournament_summary.sort_values(by="roi", ascending=False)

def main_cli(args):
    setup_logging()
    df = load_dataframes(args.input_glob)
    summary_df = run_summarize_by_tournament(df)

    if not summary_df.empty:
        log_success("Successfully summarized tournaments.")
        print(summary_df.to_string())
        
        output_path_obj = Path(args.output_csv)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path_obj, index=False)
        log_success(f"Saved tournament summary to {output_path_obj}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    main_cli(args)