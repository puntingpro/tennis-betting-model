# src/scripts/builders/build_player_features.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Add project root to the Python path ---
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.scripts.utils.config import load_config
from src.scripts.utils.file_utils import load_dataframes
from src.scripts.utils.logger import setup_logging, log_info, log_success

def calculate_player_stats(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates expanding and rolling stats for each player in a vectorized manner.
    """
    id_vars = ['match_id', 'tourney_date', 'surface']
    p1_cols = ['winner_id', 'loser_id']
    p2_cols = ['loser_id', 'winner_id']

    df_p1 = df_matches[id_vars + p1_cols].rename(columns={'winner_id': 'player_id', 'loser_id': 'opponent_id'})
    df_p1['won'] = 1
    
    df_p2 = df_matches[id_vars + p2_cols].rename(columns={'loser_id': 'player_id', 'winner_id': 'opponent_id'})
    df_p2['won'] = 0

    df_player_matches = pd.concat([df_p1, df_p2], ignore_index=True)
    df_player_matches = df_player_matches.sort_values(by='tourney_date')

    gb_player = df_player_matches.groupby('player_id')
    df_player_matches['matches_played'] = gb_player.cumcount()
    df_player_matches['wins'] = gb_player['won'].cumsum() - df_player_matches['won']
    
    gb_surface = df_player_matches.groupby(['player_id', 'surface'])
    df_player_matches['surface_matches'] = gb_surface.cumcount()
    df_player_matches['surface_wins'] = gb_surface['won'].cumsum() - df_player_matches['won']

    df_player_matches['form_last_10'] = gb_player['won'].shift(1).rolling(window=10, min_periods=1).mean().fillna(0)
    df_player_matches['win_perc'] = (df_player_matches['wins'] / df_player_matches['matches_played']).fillna(0)
    df_player_matches['surface_win_perc'] = (df_player_matches['surface_wins'] / df_player_matches['surface_matches']).fillna(0)

    stats_cols = ['match_id', 'player_id', 'win_perc', 'surface_win_perc', 'form_last_10']
    return df_player_matches[stats_cols]

def main(args):
    """
    Main function to build features, driven by the config file passed in args.
    """
    setup_logging()
    config = load_config(args.config)
    paths = config['data_paths']

    log_info("Loading and consolidating raw data...")
    df_matches = load_dataframes(paths['raw_matches_glob'])
    df_rankings = load_dataframes(paths['raw_rankings_glob'])
    df_players = pd.read_csv(paths['raw_players'], encoding='latin-1')
    
    log_info("Preprocessing data...")
    df_matches['tourney_date'] = pd.to_datetime(df_matches['tourney_date'], format='%Y%m%d', errors='coerce')
    df_rankings['ranking_date'] = pd.to_datetime(df_rankings['ranking_date'], format='%Y%m%d', errors='coerce')
    df_matches.dropna(subset=['tourney_date', 'winner_id', 'loser_id'], inplace=True)
    
    # --- FIX: Ensure consistent data types ---
    df_matches['winner_id'] = df_matches['winner_id'].astype('int64')
    df_matches['loser_id'] = df_matches['loser_id'].astype('int64')
    df_rankings['player'] = df_rankings['player'].astype('int64')
    # --- END FIX ---
    
    df_matches['match_id'] = df_matches['tourney_id'].astype(str) + '-' + df_matches['match_num'].astype(str)
    df_matches = df_matches.sort_values(by='tourney_date')

    log_info("Calculating player stats (vectorized)...")
    player_stats_df = calculate_player_stats(df_matches)

    log_info("Merging stats and building final feature set...")
    features_df = df_matches[['match_id', 'tourney_date', 'tourney_name', 'surface', 'winner_id', 'loser_id']].copy()
    features_df = features_df.rename(columns={'winner_id': 'p1_id', 'loser_id': 'p2_id'})
    features_df['winner'] = 1

    features_df = pd.merge(features_df, player_stats_df, left_on=['match_id', 'p1_id'], right_on=['match_id', 'player_id'], how='left').rename(columns={'win_perc': 'p1_win_perc', 'surface_win_perc': 'p1_surface_win_perc', 'form_last_10': 'p1_form_last_10'}).drop('player_id', axis=1)
    features_df = pd.merge(features_df, player_stats_df, left_on=['match_id', 'p2_id'], right_on=['match_id', 'player_id'], how='left').rename(columns={'win_perc': 'p2_win_perc', 'surface_win_perc': 'p2_surface_win_perc', 'form_last_10': 'p2_form_last_10'}).drop('player_id', axis=1)

    player_info = df_players[['player_id', 'hand', 'height']].set_index('player_id')
    features_df = features_df.merge(player_info, left_on='p1_id', right_index=True, how='left').rename(columns={'hand': 'p1_hand', 'height': 'p1_height'})
    features_df = features_df.merge(player_info, left_on='p2_id', right_index=True, how='left').rename(columns={'hand': 'p2_hand', 'height': 'p2_height'})

    log_info("Fetching player rankings (vectorized)...")
    df_rankings_sorted = df_rankings.sort_values(by='ranking_date')
    features_df_sorted = features_df.sort_values(by='tourney_date')

    p1_ranks = pd.merge_asof(
        left=features_df_sorted[['match_id', 'tourney_date', 'p1_id']],
        right=df_rankings_sorted[['ranking_date', 'player', 'rank']],
        left_on='tourney_date',
        right_on='ranking_date',
        left_by='p1_id',
        right_by='player',
        direction='backward'
    ).rename(columns={'rank': 'p1_rank'})[['match_id', 'p1_rank']]

    p2_ranks = pd.merge_asof(
        left=features_df_sorted[['match_id', 'tourney_date', 'p2_id']],
        right=df_rankings_sorted[['ranking_date', 'player', 'rank']],
        left_on='tourney_date',
        right_on='ranking_date',
        left_by='p2_id',
        right_by='player',
        direction='backward'
    ).rename(columns={'rank': 'p2_rank'})[['match_id', 'p2_rank']]

    features_df = pd.merge(features_df, p1_ranks, on='match_id', how='left')
    features_df = pd.merge(features_df, p2_ranks, on='match_id', how='left')
    features_df['rank_diff'] = features_df['p1_rank'] - features_df['p2_rank']
    
    features_df['h2h_p1_wins'] = 0
    features_df['h2h_p2_wins'] = 0

    output_path = Path(paths['consolidated_features'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_info(f"Saving features to {output_path}...")
    features_df.to_csv(output_path, index=False)
    
    log_success(f"✅ Successfully created feature library at {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config file.")
    args = parser.parse_args()
    main(args)