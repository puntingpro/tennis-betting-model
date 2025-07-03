import pandas as pd
from pathlib import Path

from scripts.utils.file_utils import load_dataframes
from scripts.utils.logger import log_error, log_info, log_success


def run_summarize_by_tournament(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes match-level data to a tournament-level summary.
    """
    # Add the 'label' column from the file path
    if "label" not in df.columns:
        df["label"] = df["source_file"].apply(lambda x: Path(x).stem.replace('_by_match', ''))

    # Ensure required columns exist before trying to aggregate
    required_cols = ["total_bets", "total_profit", "match_id"]
    if not all(col in df.columns for col in required_cols):
        log_error(f"Missing one or more required columns: {required_cols}")
        return pd.DataFrame()

    # Group by the new 'label' column
    tournament_summary = (
        df.groupby("label")
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
    """
    Main CLI handler for summarizing tournaments.
    Accepts args object from main.py.
    """
    try:
        df = load_dataframes(args.input_glob, add_source_column=True)
        summary_df = run_summarize_by_tournament(df)

        if not summary_df.empty:
            log_success("Successfully summarized tournaments.")
            print(summary_df.to_string())
            
            output_path = getattr(args, 'output_csv', None)
            if output_path:
                output_path_obj = Path(output_path)
                output_path_obj.parent.mkdir(parents=True, exist_ok=True)
                summary_df.to_csv(output_path_obj, index=False)
                log_success(f"Saved tournament summary to {output_path_obj}")
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")