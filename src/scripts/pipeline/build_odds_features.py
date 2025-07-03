import numpy as np
import pandas as pd

from scripts.utils.logger import log_info, log_warning
from scripts.utils.schema import enforce_schema, normalize_columns


def build_odds_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds implied probability and bookmaker margin features to a DataFrame.
    This function uses a memory-safe groupby approach to handle large datasets.
    """
    df = normalize_columns(df)
    
    required_cols = ['match_id', 'runner_name', 'final_ltp', 'winner']
    if not all(col in df.columns for col in required_cols):
        log_info(f"Missing one of {required_cols}; cannot build odds features.")
        return enforce_schema(pd.DataFrame(), schema_name="features")

    # List to hold the processed data for each match
    processed_matches = []

    # Group by match and process each one individually
    for match_id, group in df.groupby('match_id'):
        # Skip matches that don't have exactly two players (handles walkovers, bad data)
        if len(group) != 2:
            log_warning(f"Skipping match_id {match_id}: found {len(group)} players instead of 2.")
            continue
            
        # Extract data for player 1 and player 2
        p1_data = group.iloc[0]
        p2_data = group.iloc[1]

        # Create two rows, one for each player's perspective as the primary bet
        row1 = {
            'match_id': match_id,
            'player_1': p1_data['runner_name'],
            'player_2': p2_data['runner_name'],
            'odds_1': p1_data['final_ltp'],
            'odds_2': p2_data['final_ltp'],
            'winner': p1_data['winner']
        }
        
        row2 = {
            'match_id': match_id,
            'player_1': p2_data['runner_name'],
            'player_2': p1_data['runner_name'],
            'odds_1': p2_data['final_ltp'],
            'odds_2': p1_data['final_ltp'],
            'winner': p2_data['winner']
        }
        
        processed_matches.extend([row1, row2])

    if not processed_matches:
        log_warning("No valid 2-player matches found to process.")
        return enforce_schema(pd.DataFrame(), schema_name="features")

    # Create a new DataFrame from the processed list
    features_df = pd.DataFrame(processed_matches)

    # Drop rows where player names are missing before further processing
    features_df.dropna(subset=['player_1', 'player_2'], inplace=True)

    features_df["implied_prob_1"] = 1 / pd.to_numeric(features_df["odds_1"], errors="coerce")
    features_df["implied_prob_2"] = 1 / pd.to_numeric(features_df["odds_2"], errors="coerce")
    features_df["implied_prob_diff"] = features_df["implied_prob_1"] - features_df["implied_prob_2"]
    features_df["odds_margin"] = features_df["implied_prob_1"] + features_df["implied_prob_2"]
    
    log_info("Added implied probability and margin features.")

    return enforce_schema(features_df, schema_name="features")


def main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Build odds features")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    result = build_odds_features(df)
    if not args.dry_run:
        result.to_csv(args.output_csv, index=False)
        log_info(f"Features written to {args.output_csv}")


if __name__ == "__main__":
    main_cli()