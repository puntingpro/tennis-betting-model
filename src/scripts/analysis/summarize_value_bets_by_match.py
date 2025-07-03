import pandas as pd
from pathlib import Path

from scripts.utils.file_utils import load_dataframes
from scripts.utils.logger import log_error, log_info, log_success
from scripts.utils.schema import patch_winner_column


def run_summarize_value_bets_by_match(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes value bets by match, calculating total bets, profit, and ROI.
    """
    df = patch_winner_column(df)
    if df.empty:
        log_info("DataFrame is empty, cannot summarize.")
        return pd.DataFrame()

    df["is_correct"] = (df["predicted_prob"] > 0.5) == df["winner"]
    df["profit"] = df.apply(
        lambda row: row["odds"] - 1 if row["is_correct"] else -1, axis=1
    )
    
    # Add label to the grouping
    summary_df = (
        df.groupby(["label", "match_id"])
        .agg(
            total_bets=("match_id", "size"),
            total_profit=("profit", "sum"),
            avg_odds=("odds", "mean"),
            avg_predicted_prob=("predicted_prob", "mean"),
        )
        .reset_index()
    )
    summary_df["roi"] = (summary_df["total_profit"] / summary_df["total_bets"]) * 100
    return summary_df.sort_values(by="total_profit", ascending=False)


def main_cli(args):
    """
    Main CLI handler for summarizing value bets.
    Accepts args object from main.py.
    """
    try:
        # Add source file to get the label
        df = load_dataframes(args.value_bets_glob, add_source_column=True)
        if "label" not in df.columns:
            df["label"] = df["source_file"].apply(lambda x: Path(x).stem.replace('_value_bets', ''))
            
        summary_df = run_summarize_value_bets_by_match(df)

        if not summary_df.empty:
            log_success("Successfully summarized matches.")
            
            # Save one summary file per tournament label
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for label, group in summary_df.groupby('label'):
                output_path = output_dir / f"{label}_by_match.csv"
                group.to_csv(output_path, index=False)
                log_success(f"Saved match summary for {label} to {output_path}")

    except FileNotFoundError as e:
        log_error(f"Error loading files: {e}")
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")