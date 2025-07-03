import pandas as pd
import numpy as np

from scripts.utils.logger import log_info, setup_logging
from scripts.utils.schema import enforce_schema


def build_matches_from_snapshots(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    # This function expects a DataFrame from parsed snapshot files.
    
    # Define aggregations, including 'runner_name'
    aggregations = {
        "runner_name": ("runner_name", "first"), # This line preserves the player's name
        "ltp": ("ltp", "last"),
        "timestamp": ("timestamp", "max"),
    }
    
    # Conditionally add the volume aggregation only if the column exists
    if "volume" in snapshot_df.columns:
        aggregations["volume"] = ("volume", "sum")

    grouped = snapshot_df.groupby(["market_id", "selection_id"], as_index=False).agg(**aggregations)

    # If the volume column was missing from the start, create it now and fill with 0
    if "volume" not in grouped.columns:
        grouped["volume"] = 0.0

    # Create the 'match_id' column as a copy of 'market_id' instead of renaming.
    grouped["match_id"] = grouped["market_id"]

    matches_df = enforce_schema(grouped, schema_name="matches")
    return matches_df


def main(args):
    """
    Entry point to build matches from snapshot CSV.
    """
    df = pd.read_csv(args.input_path)
    log_info("Loaded %d snapshots from %s", len(df), args.input_path)
    matches_df = build_matches_from_snapshots(df)
    if args.dry_run:
        log_info("Dry-run mode active; no file written.")
    else:
        matches_df.to_csv(args.output_path, index=False)
        log_info("Matches written to %s", args.output_path)