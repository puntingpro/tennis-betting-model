import pandas as pd

from scripts.utils.logger import log_info, log_warning
from scripts.utils.schema import enforce_schema, normalize_columns


def merge_final_ltps(
    matches_df: pd.DataFrame, snapshots_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Finds the last traded price (LTP) for each selection and merges it into the matches DataFrame.
    """
    matches_df = normalize_columns(matches_df)
    snapshots_df = normalize_columns(snapshots_df)

    if "match_id" not in snapshots_df.columns and "market_id" in snapshots_df.columns:
        snapshots_df["match_id"] = snapshots_df["market_id"]

    # Robustly clean and cast join keys to a standard string format
    key_cols = ['match_id', 'selection_id']
    for col in key_cols:
        if col in matches_df.columns and not matches_df.empty:
            # Convert to string and remove '.0' if it exists from float conversion
            matches_df[col] = matches_df[col].astype(str).str.replace(r'\.0$', '', regex=True)
        if col in snapshots_df.columns and not snapshots_df.empty:
            snapshots_df[col] = snapshots_df[col].astype(str).str.replace(r'\.0$', '', regex=True)

    log_info(f"Preparing to merge. Matches DF shape: {matches_df.shape}")
    log_info(f"Snapshots DF shape: {snapshots_df.shape}")
            
    final_snaps = (
        snapshots_df.sort_values(["match_id", "selection_id", "timestamp"])
        .groupby(["match_id", "selection_id"], as_index=False)
        .last()[["match_id", "selection_id", "ltp"]]
    )

    log_info(f"Created final_snaps DF with shape: {final_snaps.shape}")
    
    df_merged = matches_df.merge(
        final_snaps,
        on=["match_id", "selection_id"],
        how="left",
        suffixes=("", "_final"),
    )
    df_merged.rename(columns={"ltp_final": "final_ltp"}, inplace=True)
    
    ltp_count = df_merged['final_ltp'].notna().sum()
    log_info(f"Merge complete. Found {ltp_count} matching LTPs out of {len(df_merged)} rows.")
    
    if ltp_count == 0 and not df_merged.empty:
        log_warning(f"Merge for a batch starting with match_id {df_merged.iloc[0].get('match_id', 'N/A')} failed to find any matching LTPs.")

    return enforce_schema(df_merged, "merged_matches")


def main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Merge final LTPs into matches")
    parser.add_argument("--matches_csv", required=True)
    parser.add_argument("--snapshots_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    df_matches = pd.read_csv(args.matches_csv)
    df_snaps = pd.read_csv(args.snapshots_csv)
    result = merge_final_ltps(df_matches, df_snaps)
    if not args.dry_run:
        result.to_csv(args.output_csv, index=False)
        log_info(f"Merged matches written to {args.output_csv}")


if __name__ == "__main__":
    main_cli()