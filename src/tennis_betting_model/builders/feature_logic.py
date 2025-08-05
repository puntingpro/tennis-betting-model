# src/tennis_betting_model/builders/feature_logic.py

import pandas as pd


def get_h2h_stats(
    df_matches: pd.DataFrame, p1_id: int, p2_id: int, match_date: pd.Timestamp
) -> tuple[int, int]:
    """
    Calculates point-in-time Head-to-Head (H2H) stats for two players.
    """
    # Ensure match_date is timezone-aware for safe comparison
    if match_date.tzinfo is None:
        match_date = match_date.tz_localize("UTC")

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
        & (df_matches["tourney_date"] < match_date)
    ]

    p1_wins = len(h2h_matches[h2h_matches["winner_historical_id"] == p1_id])
    p2_wins = len(h2h_matches[h2h_matches["winner_historical_id"] == p2_id])

    return p1_wins, p2_wins


def get_player_stats(
    df_matches: pd.DataFrame, player_id: int, surface: str, match_date: pd.Timestamp
) -> tuple[float, float, float, int, int]:
    """
    Calculates point-in-time win percentages, recent form, and fatigue for a single player.
    """
    # Ensure match_date is timezone-aware for safe comparison
    if match_date.tzinfo is None:
        match_date = match_date.tz_localize("UTC")

    player_matches = df_matches[
        (
            (df_matches["winner_historical_id"] == player_id)
            | (df_matches["loser_historical_id"] == player_id)
        )
        & (df_matches["tourney_date"] < match_date)
    ].sort_values("tourney_date")

    if player_matches.empty:
        return 0.0, 0.0, 0.0, 0, 0

    wins = (player_matches["winner_historical_id"] == player_id).sum()
    win_perc = wins / len(player_matches)

    surface_matches = player_matches[player_matches["surface"] == surface]
    if not surface_matches.empty:
        surface_wins = (surface_matches["winner_historical_id"] == player_id).sum()
        surface_win_perc = surface_wins / len(surface_matches)
    else:
        surface_win_perc = 0.0

    last_10 = player_matches.tail(10)
    form_last_10 = (
        (last_10["winner_historical_id"] == player_id).mean()
        if not last_10.empty
        else 0.0
    )

    recent_match_dates = player_matches["tourney_date"]

    # Calculate time differences safely
    time_since_matches = match_date - recent_match_dates  # type: ignore
    matches_last_14_days = (time_since_matches.dt.days <= 14).sum()
    matches_last_7_days = (time_since_matches.dt.days <= 7).sum()

    return (
        win_perc,
        surface_win_perc,
        form_last_10,
        int(matches_last_7_days),
        int(matches_last_14_days),
    )
