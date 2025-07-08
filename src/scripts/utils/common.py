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