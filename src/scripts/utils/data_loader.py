# src/scripts/utils/data_loader.py

import pandas as pd
from .logger import log_info
from .file_utils import load_dataframes

def load_pipeline_data(paths: dict) -> tuple:
    """Loads all necessary data sources for the pipeline."""
    log_info("Loading model and all required data sources...")
    
    # --- MODIFIED: Use the correct consolidated file paths ---
    df_matches = pd.read_csv(paths['consolidated_matches'])
    df_matches['tourney_date'] = pd.to_datetime(df_matches['tourney_date'], errors='coerce').dt.tz_localize('UTC')
    df_matches.dropna(subset=['tourney_date', 'winner_id', 'loser_id'], inplace=True)
    df_matches['winner_id'] = df_matches['winner_id'].astype(int)
    df_matches['loser_id'] = df_matches['loser_id'].astype(int)

    # Load player lookup
    df_players = pd.read_csv(paths['raw_players'], encoding='latin-1')
    df_players = df_players.drop_duplicates(subset=['player_id'], keep='first')
    player_info_lookup = df_players.set_index('player_id').to_dict('index')

    # Load consolidated rankings data
    df_rankings = pd.read_csv(paths['consolidated_rankings'])
    df_rankings['ranking_date'] = pd.to_datetime(df_rankings['ranking_date'], utc=True)
    df_rankings = df_rankings.sort_values(by='ranking_date')
    
    log_info("✅ All data loaded successfully.")
    return player_info_lookup, df_rankings, df_matches