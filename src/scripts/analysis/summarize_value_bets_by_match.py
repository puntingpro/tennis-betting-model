# src/scripts/analysis/summarize_value_bets_by_match.py
import sys
from pathlib import Path
import pandas as pd
import argparse

project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.scripts.utils.file_utils import load_dataframes
from src.scripts.utils.logger import log_error, log_success, setup_logging
from src.scripts.utils.schema import patch_winner_column

def run_summarize_value_bets_by_match(df: pd.DataFrame) -> pd.DataFrame:
    df = patch_winner_column(df)
    if df.empty: return pd.DataFrame()

    df["profit"] = df.apply(lambda row: row["odds"] - 1 if row["winner"] == 1 else -1, axis=1)
    
    summary_df = (
        df.groupby(["tourney_name", "match_id"])
        .agg(total_bets=("match_id", "size"), total_profit=("profit", "sum"))
        .reset_index()
    )
    return summary_df

def main_cli(args):
    setup_logging()
    df = load_dataframes(args.value_bets_glob)
    summary_df = run_summarize_value_bets_by_match(df)

    if not summary_df.empty:
        log_success("Successfully summarized matches.")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save a single summary file now that the source is combined
        output_path = output_dir / "all_matches_summary.csv"
        summary_df.to_csv(output_path, index=False)
        log_success(f"Saved match summary to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--value_bets_glob", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main_cli(args)