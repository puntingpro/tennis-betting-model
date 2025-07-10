# src/scripts/pipeline/feature_engineering.py

import pandas as pd

def get_h2h_stats(df_matches: pd.DataFrame, p1_id: int, p2_id: int, match_date: pd.Timestamp) -> tuple[int, int]:
    """Calculates point-in-time Head-to-Head stats for two players."""
    # Filter for past matches between the two players before the current match date
    h2h_matches = df_matches[
        ((df_matches['winner_id'] == p1_id) & (df_matches['loser_id'] == p2_id) |
         (df_matches['winner_id'] == p2_id) & (df_matches['loser_id'] == p1_id)) &
        (df_matches['tourney_date'] < match_date)
    ]
    
    p1_wins = len(h2h_matches[h2h_matches['winner_id'] == p1_id])
    p2_wins = len(h2h_matches[h2h_matches['winner_id'] == p2_id])
    
    return p1_wins, p2_wins

def get_player_form_and_win_perc(df_matches: pd.DataFrame, player_id: int, surface: str, match_date: pd.Timestamp) -> tuple[float, float, float]:
    """Calculates point-in-time win percentages and form."""
    player_matches = df_matches[
        ((df_matches['winner_id'] == player_id) | (df_matches['loser_id'] == player_id)) &
        (df_matches['tourney_date'] < match_date)
    ].sort_values('tourney_date')

    if player_matches.empty:
        return 0.0, 0.0, 0.0
        
    wins = (player_matches['winner_id'] == player_id).sum()
    win_perc = wins / len(player_matches)
    
    surface_matches = player_matches[player_matches['surface'] == surface]
    if not surface_matches.empty:
        surface_wins = (surface_matches['winner_id'] == player_id).sum()
        surface_win_perc = surface_wins / len(surface_matches)
    else:
        surface_win_perc = 0.0

    # For form, look at the last 10 games
    last_10 = player_matches.tail(10)
    form_last_10 = (last_10['winner_id'] == player_id).mean() if not last_10.empty else 0.0

    return win_perc, surface_win_perc, form_last_10