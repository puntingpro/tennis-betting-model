# src/tennis_betting_model/builders/feature_logic.py
from typing import Tuple
import pandas as pd


def get_win_percentages(
    df_matches: pd.DataFrame, player_id: int, surface: str, match_date: pd.Timestamp
) -> Tuple[float, float, float]:
    """
    Calculates overall, surface-specific, and recent win percentages for a player.
    """
    player_matches = df_matches[
        (df_matches["winner_historical_id"] == player_id)
        | (df_matches["loser_historical_id"] == player_id)
    ].copy()

    if player_matches.empty:
        return 0.0, 0.0, 0.0

    # Explicitly convert to datetime to ensure correct type for operations
    player_matches["tourney_date"] = pd.to_datetime(
        player_matches["tourney_date"], utc=True
    )
    player_matches_before_date = player_matches[
        player_matches["tourney_date"] < match_date
    ]

    if player_matches_before_date.empty:
        return 0.0, 0.0, 0.0

    total_wins = (player_matches_before_date["winner_historical_id"] == player_id).sum()
    total_matches = len(player_matches_before_date)
    win_perc = total_wins / total_matches if total_matches > 0 else 0.0

    surface_matches = player_matches_before_date[
        player_matches_before_date["surface"] == surface
    ]
    surface_wins = (surface_matches["winner_historical_id"] == player_id).sum()
    total_surface_matches = len(surface_matches)
    surface_win_perc = (
        surface_wins / total_surface_matches if total_surface_matches > 0 else 0.0
    )

    return win_perc, surface_win_perc, 0.0


def get_h2h_stats_optimized(
    df_matches: pd.DataFrame, p1_id: int, p2_id: int, match_date: pd.Timestamp
) -> Tuple[int, int]:
    """
    Calculates head-to-head wins between two players before a specific match date.
    """
    if df_matches.empty:
        return 0, 0

    h2h_matches = df_matches[
        (
            (
                (df_matches["winner_historical_id"] == p1_id)
                & (df_matches["loser_historical_id"] == p2_id)
            )
            | (
                (df_matches["winner_historical_id"] == p2_id)
                & (df_matches["loser_historical_id"] == p1_id)
            )
        )
    ].copy()

    if h2h_matches.empty:
        return 0, 0

    h2h_matches["tourney_date"] = pd.to_datetime(h2h_matches["tourney_date"], utc=True)
    h2h_matches_before = h2h_matches[h2h_matches["tourney_date"] < match_date]

    if h2h_matches_before.empty:
        return 0, 0

    p1_wins = h2h_matches_before[
        h2h_matches_before["winner_historical_id"] == p1_id
    ].shape[0]
    p2_wins = h2h_matches_before[
        h2h_matches_before["winner_historical_id"] == p2_id
    ].shape[0]

    return p1_wins, p2_wins


def get_fatigue_features(
    df_matches: pd.DataFrame, player_id: int, match_date: pd.Timestamp
) -> Tuple[int, int]:
    """
    Calculates fatigue metrics: sets played in the last 7 and 14 days.
    """
    player_matches = df_matches[
        (df_matches["winner_historical_id"] == player_id)
        | (df_matches["loser_historical_id"] == player_id)
    ].copy()

    if player_matches.empty:
        return 0, 0

    # Ensure the date column is the correct type before calculations
    tourney_dates = pd.to_datetime(player_matches["tourney_date"], utc=True)

    recent_mask = (match_date - tourney_dates).dt.days <= 14
    recent_matches = player_matches[recent_mask]

    if recent_matches.empty:
        return 0, 0

    # Re-create the date series for the filtered dataframe
    recent_tourney_dates = pd.to_datetime(recent_matches["tourney_date"], utc=True)

    last_7_days_mask = (match_date - recent_tourney_dates).dt.days <= 7
    last_7_days_matches = recent_matches[last_7_days_mask]

    sets_last_7_days = int(last_7_days_matches["sets_played"].sum())
    sets_last_14_days = int(recent_matches["sets_played"].sum())

    return sets_last_7_days, sets_last_14_days


def get_recent_form(
    df_matches: pd.DataFrame, player_id: int, match_date: pd.Timestamp
) -> Tuple[int, int]:
    """
    Calculates recent form: matches played in the last 7 and 14 days.
    """
    player_matches = df_matches[
        (df_matches["winner_historical_id"] == player_id)
        | (df_matches["loser_historical_id"] == player_id)
    ].copy()

    if player_matches.empty:
        return 0, 0

    # Ensure the date column is the correct type before calculations
    tourney_dates = pd.to_datetime(player_matches["tourney_date"], utc=True)

    recent_mask = (match_date - tourney_dates).dt.days <= 14
    recent_matches = player_matches[recent_mask]

    if recent_matches.empty:
        return 0, 0

    matches_last_14_days = recent_matches.shape[0]

    # Re-create the date series for the filtered dataframe
    recent_tourney_dates = pd.to_datetime(recent_matches["tourney_date"], utc=True)

    last_7_days_mask = (match_date - recent_tourney_dates).dt.days <= 7
    matches_last_7_days = recent_matches[last_7_days_mask].shape[0]

    return matches_last_7_days, matches_last_14_days
