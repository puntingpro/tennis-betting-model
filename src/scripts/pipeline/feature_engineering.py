# src/scripts/pipeline/feature_engineering.py

import pandas as pd

def get_h2h_stats(df_matches: pd.DataFrame, p1_id: int, p2_id: int, match_date: pd.Timestamp) -> tuple[int, int]:
    """
    Calculates point-in-time Head-to-Head (H2H) stats for two players.

    Args:
        df_matches (pd.DataFrame): DataFrame of all historical matches.
        p1_id (int): The ID of player 1.
        p2_id (int): The ID of player 2.
        match_date (pd.Timestamp): The date of the current match, to exclude future games.

    Returns:
        tuple[int, int]: A tuple containing the number of wins for player 1 and player 2.
    """
    h2h_matches = df_matches[
        ((df_matches['winner_id'] == p1_id) & (df_matches['loser_id'] == p2_id) |
         (df_matches['winner_id'] == p2_id) & (df_matches['loser_id'] == p1_id)) &
        (df_matches['tourney_date'] < match_date)
    ]
    
    p1_wins = len(h2h_matches[h2h_matches['winner_id'] == p1_id])
    # --- FIXED: Corrected the variable name on the next line ---
    p2_wins = len(h2h_matches[h2h_matches['winner_id'] == p2_id])
    
    return p1_wins, p2_wins

def get_player_form_and_win_perc(df_matches: pd.DataFrame, player_id: int, surface: str, match_date: pd.Timestamp) -> tuple[float, float, float]:
    """
    Calculates point-in-time win percentages and recent form for a single player.

    Args:
        df_matches (pd.DataFrame): DataFrame of all historical matches.
        player_id (int): The ID of the player to calculate stats for.
        surface (str): The surface of the current match (e.g., 'Hard', 'Clay').
        match_date (pd.Timestamp): The date of the current match.

    Returns:
        tuple[float, float, float]: A tuple containing overall win percentage,
        surface-specific win percentage, and form over the last 10 matches.
    """
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