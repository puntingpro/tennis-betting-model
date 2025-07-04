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
    player_features_df = normalize_columns(player_features_df.copy())

    # --- DATA PREPARATION AND TYPE ENFORCEMENT ---
    log_info("Preparing and cleaning data for feature building...")
    
    key_cols = ['tourney_date', 'player_1', 'player_2', 'surface']
    
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    player_features_df['tourney_date'] = pd.to_datetime(player_features_df['tourney_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    df.dropna(subset=['match_id', 'player_1', 'player_2'], inplace=True)
    
    for col in key_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
        if col in player_features_df.columns:
            player_features_df[col] = player_features_df[col].astype(str)

    # --- FEATURE MERGING ---
    log_info("Merging player features...")
    
    matches_base = df[['match_id', 'player_1', 'player_2', 'winner', 'tourney_date', 'surface']].drop_duplicates(subset=['match_id']).reset_index(drop=True)
    
    odds_map = df.set_index(['match_id', 'runner_name'])['final_ltp'].to_dict()
    matches_base['odds_1'] = matches_base.apply(lambda row: odds_map.get((row['match_id'], row['player_1'])), axis=1)
    matches_base['odds_2'] = matches_base.apply(lambda row: odds_map.get((row['match_id'], row['player_2'])), axis=1)
    
    if not player_features_df.empty:
        matches_base = pd.merge(
            matches_base,
            player_features_df,
            on=['tourney_date', 'player_1', 'player_2', 'surface'],
            how='left'  # Use a left join to keep all matches, even if they have no new features
        )
    log_info("Successfully merged new player features.")

    # --- DATA TRANSFORMATION ---
    # CRITICAL CHANGE: Only drop rows if the core odds are missing. Do NOT drop rows if the new features are missing.
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

    features_df['odds_1'] = pd.to_numeric(features_df['odds_1'], errors='coerce')
    features_df['odds_2'] = pd.to_numeric(features_df['odds_2'], errors='coerce')
    features_df["implied_prob_1"] = 1 / features_df["odds_1"]
    features_df["implied_prob_2"] = 1 / features_df["odds_2"]
    features_df["implied_prob_diff"] = features_df["implied_prob_1"] - features_df["implied_prob_2"]
    features_df["odds_margin"] = features_df["implied_prob_1"] + features_df["implied_prob_2"]
    
    log_info("Added implied probability and margin features.")
    
    return enforce_schema(features_df, schema_name="features")

def main_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Build odds features")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--player_features_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_csv)
    player_features_df = pd.read_csv(args.player_features_csv)
    result = build_odds_features(df, player_features_df)
    
    result.to_csv(args.output_csv, index=False)
    log_info(f"Features written to {args.output_csv}")

if __name__ == "__main__":
    main_cli()