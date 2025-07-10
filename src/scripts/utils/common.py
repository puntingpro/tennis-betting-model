# src/scripts/utils/common.py
import numpy as np
import pandas as pd

def get_most_recent_ranking(df_rankings: pd.DataFrame, player_id: int, date: pd.Timestamp) -> float:
    """
    Retrieves the most recent ranking for a player before a given date.
    """
    player_rankings = df_rankings[df_rankings['player'] == player_id]
    last_ranking_idx = player_rankings['ranking_date'].searchsorted(date, side='right') - 1
    if last_ranking_idx >= 0:
        return player_rankings.iloc[last_ranking_idx]['rank']
    return np.nan

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes all column names to lowercase and replaces spaces with underscores.
    """
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    return df

def patch_winner_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles the 'winner' column, ensuring it's numeric.
    It checks for a 'result' column and converts it if 'winner' is missing.
    """
    if 'winner' not in df.columns and 'result' in df.columns:
        df['winner'] = pd.to_numeric(df['result'], errors='coerce')
    elif 'winner' in df.columns:
        df['winner'] = pd.to_numeric(df['winner'], errors='coerce')
        
    return df