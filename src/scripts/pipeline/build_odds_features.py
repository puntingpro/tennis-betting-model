# src/scripts/pipeline/build_odds_features.py

import numpy as np
import pandas as pd

from scripts.utils.logger import log_info, log_warning
from scripts.utils.schema import enforce_schema, normalize_columns

def build_odds_features(df: pd.DataFrame, player_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds implied probability and bookmaker margin features to a DataFrame
    and merges in the pre-calculated player features using a robust mapping strategy.
    """
    df = normalize_columns(df.copy())

    required_cols = ['match_id', 'player_1', 'player_2', 'final_ltp', 'winner', 'tourney_date', 'surface']
    if not all(col in df.columns for col in required_cols):
        log_warning(f"Missing one of {required_cols}; cannot build odds features.")
        return enforce_schema(pd.DataFrame(), schema_name="features")

    # 1. Create a clean, one-row-per-match DataFrame as the base
    matches_base = df[['match_id', 'player_1', 'player_2', 'winner', 'tourney_date', 'surface']].drop_duplicates(subset=['match_id']).reset_index(drop=True)

    # 2. Create a simple dictionary to map odds: (match_id, runner_name) -> final_ltp
    odds_map = df.set_index(['match_id', 'runner_name'])['final_ltp'].to_dict()

    # 3. Map the odds for both players onto the base DataFrame
    matches_base['odds_1'] = matches_base.apply(lambda row: odds_map.get((row['match_id'], row['player_1'])), axis=1)
    matches_base['odds_2'] = matches_base.apply(lambda row: odds_map.get((row['match_id'], row['player_2'])), axis=1)

    # 4. Merge the historical player features
    if not player_features_df.empty:
        log_info("Merging player features...")
        
        matches_base['tourney_date'] = pd.to_datetime(matches_base['tourney_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        player_features_df['tourney_date'] = pd.to_datetime(player_features_df['tourney_date'], errors='coerce').dt.strftime('%Y-%m-%d')

        matches_base = pd.merge(
            matches_base,
            player_features_df,
            on=['tourney_date', 'player_1', 'player_2', 'surface'],
            how='left'
        )
        log_info("Successfully merged new player features.")

    # 5. Now, create the two-perspective DataFrame from the clean base
    matches_base.dropna(subset=['odds_1', 'odds_2'], inplace=True)

    persp1 = matches_base.copy()
    persp2 = matches_base.copy().rename(columns={
        'player_1': 'player_2', 'player_2': 'player_1',
        'odds_1': 'odds_2', 'odds_2': 'odds_1',
        'p1_rolling_win_pct': 'p2_rolling_win_pct', 'p2_rolling_win_pct': 'p1_rolling_win_pct',
        'p1_surface_win_pct': 'p2_surface_win_pct', 'p2_surface_win_pct': 'p1_surface_win_pct',
    })

    features_df = pd.concat([persp1, persp2], ignore_index=True, sort=False)
    
    original_winner = features_df['winner'].copy()
    features_df['winner'] = (features_df['player_1'] == original_winner).astype(int)

    # 6. Calculate final derived features in a single, efficient step
    final_features = {
        "implied_prob_1": 1 / pd.to_numeric(features_df["odds_1"], errors="coerce"),
        "implied_prob_2": 1 / pd.to_numeric(features_df["odds_2"], errors="coerce")
    }
    features_df = features_df.assign(**final_features)
    features_df["implied_prob_diff"] = features_df["implied_prob_1"] - features_df["implied_prob_2"]
    features_df["odds_margin"] = features_df["implied_prob_1"] + features_df["implied_prob_2"]
    
    log_info("Added implied probability and margin features.")
    
    return enforce_schema(features_df, schema_name="features")


def main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Build odds features")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)
    result = build_odds_features(df, pd.DataFrame()) 
    if not args.dry_run:
        result.to_csv(args.output_csv, index=False)
        log_info(f"Features written to {args.output_csv}")

if __name__ == "__main__":
    main_cli()